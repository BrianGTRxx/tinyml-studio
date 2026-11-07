"""Resumen legible de una exportación (``*_summary_es`` y ``*_summary`` en el idioma activo)."""

from __future__ import annotations

from tinyml_studio.i18n import tr


def _verdict(b: dict) -> str:
    return b.get("verdict_label", b["verdict"])


def summary(r: dict, out_dir: str) -> str:
    """Resumen en el idioma activo (``tr``)."""
    v = r["verification"]
    L = [
        tr(
            "Exportado a {d} (placa: {b})",
            "Exported to {d} (board: {b})",
            d=out_dir,
            b=r["board"],
        ),
        tr(
            "Tablas del modelo (flash): {a} B · RAM de trabajo: ~{b} B",
            "Model tables (flash): {a} B · working RAM: ~{b} B",
            a=r["data_bytes"],
            b=r["ram_bytes_estimate"],
        ),
    ]
    for k, label in (("flash_data", "flash"), ("ram_working", "RAM")):
        b = r["budget"][k]
        if b["budget_kb"] is None:
            L.append(
                tr(
                    "  {label}: Sin presupuesto (usa --{o}-kb)",
                    "  {label}: No budget (use --{o}-kb)",
                    label=label,
                    o="flash" if k == "flash_data" else "ram",
                )
            )
        else:
            L.append(
                tr(
                    "  {label}: {v} ({f:.1%} de {kb:g} KB)",
                    "  {label}: {v} ({f:.1%} of {kb:g} KB)",
                    label=label,
                    v=_verdict(b),
                    f=b["fraction"],
                    kb=b["budget_kb"],
                )
            )
    L.append("  " + r["budget"]["note"])
    L.append("")
    if not v.get("available"):
        L.append(
            tr(
                "Verificación en C NO ejecutada: {r}",
                "C verification NOT run: {r}",
                r=v.get("reason"),
            )
        )
    elif not v.get("compile_ok"):
        L.append(
            tr("¡El C generado NO compila!\n", "The generated C does NOT compile!\n")
            + v.get("compile_error", "")
        )
    elif "vectors" not in v:
        L.append(
            tr("¡El verificador falló al ejecutarse!\n", "The verifier failed to run!\n")
            + v.get("run_error", "")
        )
    else:
        L.append(
            tr(
                "Verificación Python↔C en PC con {n} vectores del test:",
                "Python↔C verification on PC with {n} test vectors:",
                n=v["vectors"],
            )
        )
        L.append(
            tr(
                "  features: Diferencia relativa máx. {d:.2g}",
                "  features: Max. relative difference {d:.2g}",
                d=v["features_max_rel_diff"],
            )
        )
        L.append(
            tr(
                "  modelo C == scikit-learn (sobre las mismas features): {m:.2%}",
                "  C model == scikit-learn (on the same features): {m:.2%}",
                m=v["model_match"],
            )
        )
        L.append(
            tr(
                "  crudo→resultado en C == scikit-learn sobre features emuladas: {m:.2%}",
                "  raw→result in C == scikit-learn on emulated features: {m:.2%}",
                m=v["raw_to_class_match"],
            )
        )
        L.append(
            tr(
                "  crudo→resultado en C == pipeline completo de Python: {m:.2%}",
                "  raw→result in C == full Python pipeline: {m:.2%}",
                m=v["vs_full_python_pipeline"],
            )
        )
        if v.get("compile_warnings"):
            L.append(
                tr(
                    "  avisos del compilador: {n}",
                    "  compiler warnings: {n}",
                    n=len(v["compile_warnings"]),
                )
            )
        if "host_x86_64_Os_bytes" in v:
            L.append(
                tr(
                    "  código (orientativo, x86-64 -Os, NO Xtensa): ~{n} B",
                    "  code (indicative, x86-64 -Os, NOT Xtensa): ~{n} B",
                    n=v["host_x86_64_Os_bytes"]["text"],
                )
            )
    for w in r["warnings"]:
        L.append(tr("Aviso: {w}", "Warning: {w}", w=w))
    L.append(
        tr(
            "Esto se verificó en PC. Falta probar en el ESP32-S3 con test_vectors.csv.",
            "This was verified on a PC. It still needs testing on the ESP32-S3 with "
            "test_vectors.csv.",
        )
    )
    return "\n".join(L)


def summary_es(r: dict, out_dir: str) -> str:
    return summary(r, out_dir)


def tflite_summary(r: dict, out_dir: str) -> str:
    cls = "f1_macro" in next(iter(r["metrics_test"].values()))
    key = "f1_macro" if cls else "r2"
    multi = r.get("threshold") is not None
    L = [
        tr(
            "Exportado a {d} (placa: {b}) · E/S del modelo: {io}",
            "Exported to {d} (board: {b}) · model I/O: {io}",
            d=out_dir,
            b=r["board"],
            io=r["io"],
        )
    ]
    s = r["sizes_bytes"]
    L.append(
        tr(
            "Tamaño: Float {a} B → int8 {b} B (cuantización completa)",
            "Size: Float {a} B → int8 {b} B (full quantization)",
            a=s["tflite_float"],
            b=s["tflite_int8"],
        )
    )
    if "tflite_float16" in s:
        L.append(
            tr(
                "float16: {a} B (pesos a la mitad; para Linux/móvil, un micro con TFLM suele "
                "necesitar int8)",
                "float16: {a} B (half-size weights; for Linux/mobile, an MCU with TFLM usually "
                "needs int8)",
                a=s["tflite_float16"],
            )
        )
    L.append(
        tr("Ops de TFLM a registrar: ", "TFLM ops to register: ")
        + ", ".join(r["tflm_resolver"] or [tr("(desconocidas)", "(unknown)")])
    )
    a = r["arena"]
    if a.get("measured"):
        L.append(
            tr(
                "Arena medida en TFLM de PC: {a} B (sugerida con margen: {s} B)",
                "Arena measured in PC TFLM: {a} B (suggested with margin: {s} B)",
                a=a["arena_bytes"],
                s=r["arena_suggested"],
            )
        )
    for k, label in (("flash_data", "flash"), ("ram_working", "RAM (arena)")):
        b = r["budget"][k]
        if b["budget_kb"] is not None:
            L.append(
                tr(
                    "  {label}: {v} ({f:.1%} de {kb:g} KB)",
                    "  {label}: {v} ({f:.1%} of {kb:g} KB)",
                    label=label,
                    v=_verdict(b),
                    f=b["fraction"],
                    kb=b["budget_kb"],
                )
            )
    L.append("")
    acc_name = tr("exactitud exacta", "exact accuracy") if multi else tr("exactitud", "accuracy")
    thr = tr("; umbral ", "; threshold ") + str(r["threshold"]) if multi else ""
    L.append(
        tr(
            "Verificación en el test ({key} / {acc}{thr}):",
            "Verification on the test set ({key} / {acc}{thr}):",
            key=key,
            acc=acc_name,
            thr=thr,
        )
    )
    for name, m in r["metrics_test"].items():
        acc = m.get("accuracy", m.get("subset_accuracy", m.get("mae")))
        L.append(f"  {name:<13}{m[key]:.4f}  {acc:.4f}")
    c = r["comparison"]
    L.append(
        tr(
            "  salidas float vs Keras: Dif. máx. {d:.2g}",
            "  float vs Keras outputs: Max. diff. {d:.2g}",
            d=c["float_vs_keras_outputs"]["max_abs"],
        )
    )
    L.append(
        tr(
            "  salidas int8 vs float: Dif. media {m:.2g}, máx. {x:.2g}",
            "  int8 vs float outputs: Mean diff. {m:.2g}, max. {x:.2g}",
            m=c["int8_vs_float_outputs"]["mean_abs"],
            x=c["int8_vs_float_outputs"]["max_abs"],
        )
    )
    if cls:
        L.append(
            tr(
                "  coinciden int8 y Keras en {a:.2%} de las filas",
                "  int8 and Keras agree on {a:.2%} of the rows",
                a=c["agreement_int8_vs_keras"],
            )
        )
        if "agreement_tflm_vs_tflite_int8" in c:
            L.append(
                tr(
                    "  TFLite Micro vs tf.lite int8: {a:.2%}",
                    "  TFLite Micro vs tf.lite int8: {a:.2%}",
                    a=c["agreement_tflm_vs_tflite_int8"],
                )
            )
    L.append(
        tr(
            "  pérdida de {key} por int8: {d:+.4f}",
            "  {key} loss from int8: {d:+.4f}",
            key=key,
            d=r["int8_f1_drop"],
        )
    )
    wv = r.get("window_c_verification")
    if wv and wv.get("compile_ok") and "windows" in wv:
        L.append(
            tr(
                "  C de la ventana (tm_window) vs Python en {n} ventanas: Dif. relativa "
                "máx. {d:.2g}, indicador 'lista' correcto: {ok}",
                "  window C (tm_window) vs Python on {n} windows: Max. relative diff. "
                "{d:.2g}, 'ready' flag correct: {ok}",
                n=wv["windows"],
                d=wv["features_max_rel_diff"],
                ok=wv["ready_flag_ok"],
            )
        )
    elif wv:
        L.append(
            tr(
                "  C de la ventana: Verificación no completada ({wv})",
                "  window C: Verification not completed ({wv})",
                wv=wv,
            )
        )
    for w in r["warnings"]:
        L.append(tr("Aviso: {w}", "Warning: {w}", w=w))
    L.append(
        tr(
            "Verificado en PC. Falta probar inference_example.ino en el ESP32-S3 (no se compiló).",
            "Verified on a PC. inference_example.ino still needs testing on the ESP32-S3 "
            "(it was not compiled).",
        )
    )
    return "\n".join(L)


def tflite_summary_es(r: dict, out_dir: str) -> str:
    return tflite_summary(r, out_dir)
