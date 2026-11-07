"""CLI: ``python -m tinyml_studio.cli profile <csv>`` (misma lógica que la API)."""

from __future__ import annotations

import argparse
import json
import sys

from tinyml_studio.i18n import SUPPORTED, set_lang, tr
from tinyml_studio.io import read_influx_csv
from tinyml_studio.profile import profile_dataframe


def _add_keras_compute_flags(sp) -> None:
    from tinyml_studio.train import backend as B

    sp.add_argument(
        "--optimizer",
        choices=B.OPTIMIZERS,
        default="adam",
        help=tr("optimizador (por defecto adam)", "optimizer (default adam)"),
    )
    sp.add_argument(
        "--weight-decay",
        type=float,
        default=0.004,
        help=tr("solo adamw", "adamw only"),
    )
    sp.add_argument(
        "--nesterov",
        action="store_true",
        help=tr("solo sgd (momento 0.9)", "sgd only (momentum 0.9)"),
    )
    sp.add_argument(
        "--lr-schedule",
        choices=B.SCHEDULES,
        default="constant",
        help=tr("programa de la tasa de aprendizaje", "learning-rate schedule"),
    )
    sp.add_argument(
        "--lr-final-frac",
        type=float,
        default=0.01,
        help=tr(
            "tasa final / inicial (cosine, exp_decay)", "final / initial rate (cosine, exp_decay)"
        ),
    )
    sp.add_argument(
        "--backend",
        choices=B.BACKENDS,
        default="tensorflow",
        help=tr(
            "backend de Keras; 'torch' entrena con PyTorch y luego pasa los pesos a TensorFlow",
            "Keras backend; 'torch' trains with PyTorch then copies the weights to TensorFlow",
        ),
    )
    sp.add_argument(
        "--device",
        choices=B.DEVICES,
        default="auto",
        help=tr(
            "'gpu': Usa la GPU si algún framework la ve; si no, CPU con aviso",
            "'gpu': Use the GPU if any framework sees one; otherwise CPU with a warning",
        ),
    )


def _keras_compute_kwargs(args) -> dict:
    return dict(
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        lr_schedule=args.lr_schedule,
        lr_final_frac=args.lr_final_frac,
        backend=args.backend,
        device=args.device,
    )




def _apply_lang(argv: list[str] | None) -> None:
    """``--lang`` debe conocerse antes de construir el parser (los ``help`` se traducen)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=list(SUPPORTED), default=None)
    known, _ = pre.parse_known_args(argv)
    if known.lang:
        set_lang(known.lang)


def main(argv: list[str] | None = None) -> int:
    _apply_lang(argv)
    ap = argparse.ArgumentParser(prog="tinyml-studio")
    ap.add_argument(
        "--lang",
        choices=list(SUPPORTED),
        default=None,
        help=tr(
            "idioma de los mensajes (por defecto: Variable TINYML_LANG o es)",
            "language of the messages (default: TINYML_LANG variable or es)",
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    doc = sub.add_parser(
        "doctor",
        help=tr(
            "revisa qué le falta o qué versión no es compatible a un entorno Python",
            "check what a Python environment is missing or has incompatible versions of",
        ),
    )
    doc.add_argument(
        "--python",
        help=tr("intérprete a revisar (venv/conda); por defecto el actual", "interpreter to check"),
    )
    doc.add_argument(
        "--components",
        default="base",
        help=tr(
            "base,tui,keras,torch,tflm (separados por coma)",
            "base,tui,keras,torch,tflm (comma-separated)",
        ),
    )
    doc.add_argument("--json", action="store_true", help=tr("salida JSON", "JSON output"))
    rep = sub.add_parser(
        "report",
        help=tr(
            "gráficas (PNG/JPG/SVG) e informe PDF de un entrenamiento ya hecho",
            "charts (PNG/JPG/SVG) and PDF report of a finished training run",
        ),
    )
    rep.add_argument(
        "run_dir", help=tr("carpeta del entrenamiento (results.json)", "training folder")
    )
    rep.add_argument("--out", required=True, help=tr("carpeta de destino", "destination folder"))
    rep.add_argument("--formats", default="png", help=tr("png,jpg,svg", "png,jpg,svg"))
    rep.add_argument(
        "--charts", default="", help=tr("ids separados por coma (vacío: Todas)", "ids")
    )
    rep.add_argument(
        "--pdf",
        action="store_true",
        help=tr("generar también informe.pdf", "also build informe.pdf"),
    )
    tui = sub.add_parser(
        "tui",
        help=tr(
            "interfaz de terminal (asistente completo, sin ventana gráfica)",
            "terminal interface (full wizard, no graphical window)",
        ),
    )
    tui.add_argument(
        "paths", nargs="*", help=tr("CSV o carpetas a abrir", "CSVs or folders to open")
    )
    tui.add_argument(
        "--python",
        default=None,
        metavar="RUTA",
        help=tr(
            "python.exe/python de TU entorno (venv, conda): Se revisa con el doctor y la TUI "
            "se lanza allí",
            "python.exe/python of YOUR environment (venv, conda): Checked with the doctor and "
            "the TUI is launched there",
        ),
    )
    p = sub.add_parser(
        "profile",
        help=tr(
            "perfila un CSV (Influx o simple) y escribe JSON",
            "profile a CSV (Influx or plain) and write JSON",
        ),
    )
    p.add_argument("csv")
    p.add_argument("--label-column", default=None)
    i = sub.add_parser(
        "inspect",
        help=tr(
            "entiende uno o varios CSV (o una carpeta): Esquema, roles, alertas",
            "understand one or more CSVs (or a folder): Schema, roles, alerts",
        ),
    )
    i.add_argument("paths", nargs="+")
    i.add_argument(
        "--json", action="store_true", help=tr("informe completo en JSON", "full report as JSON")
    )
    i.add_argument(
        "--recipe-out",
        default=None,
        help=tr(
            "guarda la receta editable en este archivo", "save the editable recipe to this file"
        ),
    )
    a = sub.add_parser(
        "apply",
        help=tr(
            "aplica una receta JSON y escribe el CSV resultante",
            "apply a JSON recipe and write the resulting CSV",
        ),
    )
    a.add_argument("recipe")
    a.add_argument("--out", required=True)
    c = sub.add_parser(
        "clean",
        help=tr(
            "organiza, parte y limpia según una receta; escribe train/val/test",
            "organize, split and clean according to a recipe; writes train/val/test",
        ),
    )
    c.add_argument("recipe")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument(
        "--html",
        action="store_true",
        help=tr("gráficas antes/después (Plotly, HTML)", "before/after charts (Plotly, HTML)"),
    )
    vg = sub.add_parser(
        "validate-groups",
        help=tr(
            "valida dejando una sesión (archivo) fuera: ¿funciona con sesiones nuevas?",
            "leave-one-session-out validation: Does it work on new sessions?",
        ),
    )
    vg.add_argument("recipe")
    vg.add_argument("--model", default="random_forest")
    vg.add_argument(
        "--group",
        default=None,
        help=tr("columna de sesión (por defecto source)", "session column (default: Source)"),
    )
    vg.add_argument("--exclude", default="")
    vg.add_argument("--seed", type=int, default=42)
    vg.add_argument(
        "--json", default=None, help=tr("guardar el resultado en JSON", "save the result as JSON")
    )
    tn = sub.add_parser(
        "tune",
        help=tr(
            "busca hiperparámetros que quepan en el flash/RAM de la placa",
            "search hyperparameters that fit the board's flash/RAM",
        ),
    )
    tn.add_argument("recipe")
    tn.add_argument("--model", default="random_forest")
    tn.add_argument("--flash-kb", type=float, default=None)
    tn.add_argument("--ram-kb", type=float, default=None)
    tn.add_argument("--trials", type=int, default=24)
    tn.add_argument("--exclude", default="")
    tn.add_argument("--seed", type=int, default=42)
    tn.add_argument("--json", default=None)
    im = sub.add_parser(
        "importance",
        help=tr(
            "importancia de variables por permutación (sobre validación)",
            "permutation feature importance (on validation)",
        ),
    )
    im.add_argument("run_dir")
    im.add_argument("--model", default=None)
    im.add_argument("--repeats", type=int, default=5)
    im.add_argument("--json", default=None)
    la = sub.add_parser(
        "latency",
        help=tr(
            "latencia ESTIMADA por predicción en cada placa (no es una medición)",
            "ESTIMATED latency per prediction on each board (not a measurement)",
        ),
    )
    la.add_argument("run_dir")
    la.add_argument("--model", default=None)
    la.add_argument("--board", default="all")
    la.add_argument("--target-ms", type=float, default=None)
    rj = sub.add_parser(
        "reject",
        help=tr(
            "calibra el umbral para contestar 'desconocido' cuando el modelo duda",
            "calibrate the threshold to answer 'unknown' when the model is unsure",
        ),
    )
    rj.add_argument("run_dir")
    rj.add_argument("--model", default=None)
    rj.add_argument("--target-accuracy", type=float, default=0.95)
    rj.add_argument("--no-holdout", action="store_true")
    rj.add_argument("--json", default=None)
    an = sub.add_parser(
        "anomaly",
        help=tr(
            "detector de anomalías: Aprende lo normal y avisa de lo que no se parece",
            "anomaly detector: Learns what is normal and flags what is not",
        ),
    )
    an.add_argument("recipe")
    an.add_argument("--model", default="isolation_forest")
    an.add_argument(
        "--normal",
        default="",
        help=tr(
            "clases normales separadas por coma (las demás solo se usan para evaluar)",
            "comma-separated normal classes (the others are only used to evaluate)",
        ),
    )
    an.add_argument("--false-alarm", type=float, default=0.01)
    an.add_argument("--exclude", default="")
    an.add_argument("--seed", type=int, default=42)
    an.add_argument("--out-dir", default=None)
    pt = sub.add_parser(
        "train",
        help=tr(
            "entrena modelos clásicos sobre los splits de una receta",
            "train classical models on a recipe's splits",
        ),
    )
    pt.add_argument("recipe")
    pt.add_argument(
        "--out-dir",
        default=None,
        help=tr(
            "guarda results.json, encoder y mejor modelo",
            "save results.json, the encoder and the best model",
        ),
    )
    pt.add_argument(
        "--models",
        default=None,
        help=tr(
            "lista separada por comas (por defecto todos)", "comma-separated list (default: All)"
        ),
    )
    pt.add_argument(
        "--exclude",
        default="",
        help=tr("columnas a excluir de las features", "columns to exclude from the features"),
    )
    pt.add_argument("--seed", type=int, default=42)
    ex = sub.add_parser(
        "export",
        help=tr(
            "exporta un modelo clásico entrenado a C y lo verifica",
            "export a trained classical model to C and verify it",
        ),
    )
    ex.add_argument(
        "run_dir",
        help=tr("carpeta creada por 'train --out-dir'", "folder created by 'train --out-dir'"),
    )
    ex.add_argument("--out-dir", required=True)
    ex.add_argument(
        "--model",
        default=None,
        help=tr("por defecto el mejor por validación", "default: The best on validation"),
    )
    ex.add_argument("--board", default="esp32-s3")
    ex.add_argument(
        "--flash-kb",
        type=float,
        default=None,
        help=tr("flash disponible para el modelo", "flash available for the model"),
    )
    ex.add_argument(
        "--ram-kb",
        type=float,
        default=None,
        help=tr("RAM disponible para el modelo", "RAM available for the model"),
    )
    tk = sub.add_parser(
        "train-keras",
        help=tr(
            "entrena un MLP en Keras (TensorFlow) sobre los splits",
            "train a Keras (TensorFlow) MLP on the splits",
        ),
    )
    tk.add_argument("recipe")
    tk.add_argument("--out-dir", required=True)
    tk.add_argument(
        "--hidden",
        default="32,16",
        help=tr(
            "neuronas por capa oculta, separadas por coma",
            "neurons per hidden layer, comma-separated",
        ),
    )
    tk.add_argument("--epochs", type=int, default=100)
    tk.add_argument("--batch-size", type=int, default=128)
    tk.add_argument("--lr", type=float, default=1e-3)
    tk.add_argument("--patience", type=int, default=8)
    tk.add_argument("--exclude", default="")
    tk.add_argument("--seed", type=int, default=42)
    _add_keras_compute_flags(tk)
    tc = sub.add_parser(
        "train-cnn1d",
        help=tr(
            "1D-CNN (Keras) sobre ventanas de filas consecutivas",
            "1D-CNN (Keras) on windows of consecutive rows",
        ),
    )
    tc.add_argument("recipe")
    tc.add_argument("--out-dir", required=True)
    tc.add_argument(
        "--window", type=int, default=16, help=tr("filas por ventana", "rows per window")
    )
    tc.add_argument(
        "--stride",
        type=int,
        default=4,
        help=tr("paso entre ventanas de train", "step between train windows"),
    )
    tc.add_argument(
        "--filters",
        default="16,32",
        help=tr("filtros por capa Conv1D", "filters per Conv1D layer"),
    )
    tc.add_argument("--kernel", type=int, default=3)
    tc.add_argument("--head", choices=["gap", "flatten"], default="gap")
    tc.add_argument("--epochs", type=int, default=60)
    tc.add_argument("--batch-size", type=int, default=128)
    tc.add_argument("--lr", type=float, default=1e-3)
    tc.add_argument("--patience", type=int, default=6)
    tc.add_argument("--exclude", default="")
    tc.add_argument("--seed", type=int, default=42)
    _add_keras_compute_flags(tc)
    eo = sub.add_parser(
        "export-onnx",
        help=tr(
            "exporta un modelo clásico entrenado a ONNX (con el preprocesado dentro)",
            "export a trained classical model to ONNX (preprocessing included)",
        ),
    )
    eo.add_argument("run_dir")
    eo.add_argument("--out-dir", required=True)
    eo.add_argument("--model", default=None)
    eo.add_argument(
        "--reject-below",
        type=float,
        default=None,
        help=tr(
            "responder 'desconocido' si la probabilidad máxima es menor (ver 'reject')",
            "answer 'unknown' if the top probability is lower (see 'reject')",
        ),
    )
    et = sub.add_parser(
        "export-tflite",
        help=tr(
            "MLP Keras -> .tflite float/int8 + C para TFLite Micro",
            "Keras MLP -> .tflite float/int8 + C for TFLite Micro",
        ),
    )
    et.add_argument(
        "run_dir", help=tr("carpeta creada por 'train-keras'", "folder created by 'train-keras'")
    )
    et.add_argument("--out-dir", required=True)
    et.add_argument(
        "--int8-io",
        action="store_true",
        help=tr("entrada/salida int8 (por defecto float)", "int8 input/output (default: Float)"),
    )
    et.add_argument("--board", default="esp32-s3")
    et.add_argument("--flash-kb", type=float, default=None)
    et.add_argument("--ram-kb", type=float, default=None)
    bm = sub.add_parser(
        "benchmark",
        help=tr(
            "compara muchos modelos sobre los mismos splits y recomienda uno (1-SE, IC, McNemar)",
            "compare many models on the same splits and recommend one (1-SE, CIs, McNemar)",
        ),
    )
    bm.add_argument("recipe")
    bm.add_argument("--out-dir", default=None)
    bm.add_argument(
        "--models", default=None, help="decision_tree,random_forest,... (por defecto: todos)"
    )
    bm.add_argument(
        "--keras",
        default=None,
        help=tr(
            "MLP Keras: Presets separados por ';', cada uno con capas ocultas por coma "
            "(p. ej. '32,16;64,32')",
            "Keras MLPs: Presets separated by ';', each with comma-separated hidden layers "
            "(e.g. '32,16;64,32')",
        ),
    )
    bm.add_argument("--keras-epochs", type=int, default=60)
    bm.add_argument(
        "--cv-folds", type=int, default=0, help=tr("0 = sin CV; 2..10", "0 = no CV; 2..10")
    )
    bm.add_argument("--selection", default="f1_macro")
    bm.add_argument("--max-flash-kb", type=float, default=None)
    bm.add_argument("--n-boot", type=int, default=200)
    bm.add_argument("--exclude", default="")
    bm.add_argument("--seed", type=int, default=0)
    bm.add_argument(
        "--json",
        default=None,
        help=tr("escribe el resultado completo en este JSON", "write the full result to this JSON"),
    )
    args = ap.parse_args(argv)
    if args.lang:
        set_lang(args.lang)

    if args.cmd == "doctor":
        from tinyml_studio.doctor import doctor, format_report

        try:
            rep = doctor(args.python, [c for c in args.components.split(",") if c])
        except (OSError, RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            json.dump(rep, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(format_report(rep))
        return 0 if rep["ok"] else 1
    if args.cmd == "report":
        from pathlib import Path

        from tinyml_studio.render import ReportData, available_charts, build_pdf, save_charts

        try:
            results = json.loads((Path(args.run_dir) / "results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        d = ReportData(results=results, dataset_name=Path(args.run_dir).name)
        only = [c for c in args.charts.split(",") if c] or None
        try:
            for p in save_charts(d, args.out, [f for f in args.formats.split(",") if f], only):
                print(p)
            if args.pdf:
                print(build_pdf(d, Path(args.out) / "informe.pdf", only))
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(
            tr(
                "Gráficas disponibles: {c}",
                "Available charts: {c}",
                c=", ".join(c for c, _ in available_charts(d)),
            )
        )
        return 0
    if args.cmd == "tui":
        if args.python:
            from tinyml_studio.relaunch import run_in_environment

            try:
                return run_in_environment(args.python, args.paths)
            except (OSError, RuntimeError) as e:
                print(str(e), file=sys.stderr)
                return 1
        from tinyml_studio.tui.app import run

        return run(args.paths)
    if args.cmd == "profile":
        res = read_influx_csv(args.csv)
        report = profile_dataframe(res.df, label_column=args.label_column)
        report["source"] = {
            "annotation_rows": res.annotation_rows,
            "separator": res.separator,
            "influx_types": res.influx_types,
        }
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    elif args.cmd == "inspect":
        from tinyml_studio.dataset import inspect_files

        report = inspect_files(args.paths)
        if args.recipe_out:
            with open(args.recipe_out, "w", encoding="utf-8") as fh:
                json.dump(report["recipe"], fh, ensure_ascii=False, indent=2)
        if args.json:
            json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(report.get("summary") or report["summary_es"])
    elif args.cmd == "apply":
        from tinyml_studio.dataset import apply_recipe

        with open(args.recipe, encoding="utf-8") as fh:
            df = apply_recipe(json.load(fh))
        df.to_csv(args.out, index=False)
        print(
            tr(
                "{n} filas x {c} columnas -> {o}",
                "{n} rows x {c} columns -> {o}",
                n=len(df),
                c=df.shape[1],
                o=args.out,
            )
        )
    elif args.cmd == "clean":
        _clean(args)
    elif args.cmd == "train-cnn1d":
        from tinyml_studio.train.cnn1d import train_keras_cnn1d
        from tinyml_studio.train.report import cnn1d_summary

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = train_keras_cnn1d(
            recipe,
            window=args.window,
            stride=args.stride,
            filters=tuple(int(f) for f in args.filters.split(",") if f),
            kernel=args.kernel,
            head=args.head,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
            out_dir=args.out_dir,
            **_keras_compute_kwargs(args),
        )
        print(cnn1d_summary(res))
    elif args.cmd == "train-keras":
        from tinyml_studio.train.keras_mlp import train_keras_mlp
        from tinyml_studio.train.report import keras_summary

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = train_keras_mlp(
            recipe,
            hidden=tuple(int(h) for h in args.hidden.split(",") if h),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
            out_dir=args.out_dir,
            **_keras_compute_kwargs(args),
        )
        print(keras_summary(res))
    elif args.cmd == "benchmark":
        from tinyml_studio.train.bench import run_benchmark

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        presets = [
            {"hidden": [int(h) for h in grp.split(",") if h], "epochs": args.keras_epochs}
            for grp in (args.keras or "").split(";")
            if grp.strip()
        ]
        res = run_benchmark(
            recipe,
            models=[m for m in args.models.split(",") if m] if args.models is not None else None,
            keras=presets or None,
            cv_folds=args.cv_folds,
            seed=args.seed,
            selection=args.selection,
            max_flash_kb=args.max_flash_kb,
            out_dir=args.out_dir,
            n_boot=args.n_boot,
            exclude=tuple(c for c in args.exclude.split(",") if c),
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        print(res["summary"])
    elif args.cmd == "export-onnx":
        from tinyml_studio.export.onnx_export import export_onnx

        rep = export_onnx(args.run_dir, args.out_dir, args.model, args.reject_below)
        v = rep["verification"]
        print(
            tr(
                "ONNX: {m} · {k} KB · coincidencia con scikit-learn {a:.2%} ({r} filas) · {ok}",
                "ONNX: {m} · {k} KB · match with scikit-learn {a:.2%} ({r} rows) · {ok}",
                m=rep["model"],
                k=round(rep["size_bytes"] / 1024, 1),
                a=v["label_match"],
                r=v["rows"],
                ok="OK" if v["ok"] else "FALLA",
            )
        )
        if not v["ok"]:
            return 1
    elif args.cmd == "export-tflite":
        from tinyml_studio.export.report import tflite_summary
        from tinyml_studio.export.tflite import export_tflite

        rep = export_tflite(
            args.run_dir,
            args.out_dir,
            int8_io=args.int8_io,
            flash_kb=args.flash_kb,
            ram_kb=args.ram_kb,
            board=args.board,
        )
        print(tflite_summary(rep, args.out_dir))
    elif args.cmd == "export":
        from tinyml_studio.export import export_run
        from tinyml_studio.export.report import summary as export_summary

        rep = export_run(
            args.run_dir,
            args.out_dir,
            args.model,
            flash_kb=args.flash_kb,
            ram_kb=args.ram_kb,
            board=args.board,
        )
        print(export_summary(rep, args.out_dir))
    elif args.cmd == "validate-groups":
        from tinyml_studio.train.groups import group_validation
        from tinyml_studio.train.groups import summary as group_summary

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = group_validation(
            recipe,
            model=args.model,
            group=args.group,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        print(group_summary(res))
    elif args.cmd == "importance":
        from tinyml_studio.train.importance import permutation_importance

        res = permutation_importance(args.run_dir, args.model, n_repeats=args.repeats)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        print(res["summary"])
    elif args.cmd == "latency":
        from tinyml_studio.export.latency import estimate_latency

        print(estimate_latency(args.run_dir, args.model, args.board, args.target_ms)["summary"])
    elif args.cmd == "reject":
        from tinyml_studio.train.reject import calibrate_rejection

        res = calibrate_rejection(
            args.run_dir,
            args.model,
            target_accuracy=args.target_accuracy,
            holdout=not args.no_holdout,
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        print(res["summary"])
    elif args.cmd == "anomaly":
        from tinyml_studio.train.anomaly import anomaly_detection

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = anomaly_detection(
            recipe,
            model=args.model,
            normal_classes=[c for c in args.normal.split(",") if c] or None,
            false_alarm=args.false_alarm,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
            out_dir=args.out_dir,
        )
        print(res["summary"])
    elif args.cmd == "tune":
        from tinyml_studio.train.tune import tune

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = tune(
            recipe,
            args.model,
            flash_kb=args.flash_kb,
            ram_kb=args.ram_kb,
            n_trials=args.trials,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        print(res["summary"])
    elif args.cmd == "train":
        from tinyml_studio.train import train_models
        from tinyml_studio.train.report import summary as train_summary

        with open(args.recipe, encoding="utf-8") as fh:
            recipe = json.load(fh)
        res = train_models(
            recipe,
            args.models.split(",") if args.models else None,
            seed=args.seed,
            exclude=tuple(c for c in args.exclude.split(",") if c),
            out_dir=args.out_dir,
        )
        print(train_summary(res))
    return 0


def _clean(args) -> None:
    from pathlib import Path

    from tinyml_studio.clean import prepare_splits
    from tinyml_studio.clean.compare import compare_stats

    with open(args.recipe, encoding="utf-8") as fh:
        recipe = json.load(fh)
    prep = prepare_splits(recipe, seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in prep.clean.items():
        df.to_csv(out / f"{name}.csv", index=False)
    fitted = {**recipe, "cleaning": prep.pipeline.to_list(), "seed": args.seed}
    (out / "recipe_fitted.json").write_text(
        json.dumps(fitted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cols = [c for s in prep.pipeline.steps for c in s.params.get("columns", [])]
    from tinyml_studio.clean.compare import numeric_columns, unit_view

    before, after = unit_view(prep)  # mismas unidades: sin el escalado
    cols = numeric_columns(before, list(dict.fromkeys(cols)))
    stats = compare_stats(before, after, cols)
    (out / "before_after_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.html:
        from tinyml_studio.clean.compare import figure

        figure(before, after, cols).write_html(out / "before_after.html", include_plotlyjs=True)
    sizes = ", ".join(f"{k}={len(v)}" for k, v in prep.clean.items())
    print(tr("filas: {s} -> {o}", "rows: {s} -> {o}", s=sizes, o=out))


if __name__ == "__main__":
    raise SystemExit(main())
