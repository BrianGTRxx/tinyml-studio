"""Builds the English user manual (PDF) of TinyML Studio.

Usage:
    python tools/manual/build_manual.py SHOTS_DIR OUT.pdf

SHOTS_DIR holds the screenshots taken by the app itself (TINYML_SMOKE_MANUAL=1, see
tools/manual/README.txt). The text is written for people who have never trained a model:
every concept is explained before it is used, and every control says what happens when you
choose it and what happens when you do not.
"""

# ruff: noqa: E501  (the manual text is long prose on purpose)
from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "tinyml_studio" / "render" / "fonts"
VERSION = "0.0.1"
MARGIN = 20
BLUE = (31, 78, 160)
GREY = (90, 98, 110)


class Manual(FPDF):
    def __init__(self, shots: Path):
        super().__init__(format="A4")
        self.shots = shots
        self.set_margins(MARGIN, 22, MARGIN)
        self.set_auto_page_break(True, 20)
        self.add_font("D", "", str(FONTS / "DejaVuSans.ttf"))
        self.add_font("D", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
        self.set_title(f"TinyML Studio {VERSION} User Manual")
        self.set_author("BrianGTRxx")
        self.set_creator("TinyML Studio manual builder")
        self.fig = 0
        self.cover_done = False
        self.fresh = False

    # ---- page furniture -------------------------------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("D", "", 8)
        self.set_text_color(*GREY)
        self.set_y(10)
        self.cell(0, 5, f"TinyML Studio {VERSION}  |  User Manual", align="L")
        self.set_draw_color(210, 214, 220)
        self.line(MARGIN, 16, 210 - MARGIN, 16)
        self.set_y(22)
        self.set_text_color(0, 0, 0)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("D", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    # ---- text blocks ----------------------------------------------------------------------
    def chapter(self, number: int, title: str):
        # a new chapter starts on a fresh page only when little room is left, so short tails of the
        # previous chapter never end up alone on a page
        if self.fresh:
            self.fresh = False
        elif self.page_break_trigger - self.get_y() < 75:
            self.add_page()
        else:
            self.ln(10)
        self.start_section(f"{number}. {title}", 0)
        self.set_font("D", "B", 20)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def h2(self, text: str):
        if self.get_y() > 245:
            self.add_page()
        self.start_section(text, 1)
        self.ln(2)
        self.set_font("D", "B", 13)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def p(self, text: str):
        self.set_font("D", "", 10)
        self.multi_cell(0, 5.4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2.2)

    def note(self, label: str, text: str):
        """A shaded paragraph: Note, Warning, Tip or If you..."""
        self.set_font("D", "", 9.5)
        self.set_fill_color(238, 243, 252)
        self.set_draw_color(*BLUE)
        x = self.get_x()
        self.set_x(x + 3)
        self.multi_cell(
            0, 5.2, f"{label}  {text}", fill=True, border="L", new_x="LMARGIN", new_y="NEXT"
        )
        self.ln(2.5)
        self.set_x(x)

    def numbered(self, items: list[str]):
        self.set_font("D", "", 10)
        for i, text in enumerate(items, 1):
            self.set_x(MARGIN + 2)
            self.multi_cell(0, 5.4, f"{i}.  {text}", new_x="LMARGIN", new_y="NEXT")
            self.ln(0.8)
        self.ln(1.5)

    def term(self, name: str, text: str):
        """Definition style line: bold term, then the explanation."""
        self.set_font("D", "B", 10)
        self.multi_cell(0, 5.4, name, new_x="LMARGIN", new_y="NEXT")
        self.set_font("D", "", 10)
        self.set_x(MARGIN + 4)
        self.multi_cell(0, 5.4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.8)

    def table(self, header: list[str], rows: list[list[str]], widths: list[float]):
        self.set_font("D", "B", 9)
        self.set_fill_color(225, 232, 245)
        for h, w in zip(header, widths, strict=True):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_font("D", "", 9)
        for row in rows:
            lines = max(
                self.multi_cell(w, 5, c, dry_run=True, output="LINES").__len__()
                for c, w in zip(row, widths, strict=True)
            )
            h = 5 * lines + 2
            if self.get_y() + h > self.page_break_trigger:
                self.add_page()
            y0 = self.get_y()
            x = MARGIN
            for c, w in zip(row, widths, strict=True):
                self.set_xy(x, y0)
                self.multi_cell(w, 5, c, border=0, new_x="RIGHT", new_y="TOP")
                x += w
            x = MARGIN
            for w in widths:
                self.rect(x, y0, w, h)
                x += w
            self.set_y(y0 + h)
        self.ln(3)

    def shot(self, name: str, caption: str, width: float = 150.0):
        path = self.shots / f"{name}.png"
        if not path.is_file():
            raise FileNotFoundError(f"missing screenshot {path}")
        jpg = self.shots / f"{name}.jpg"
        if not jpg.exists() or jpg.stat().st_mtime < path.stat().st_mtime:
            Image.open(path).convert("RGB").save(jpg, quality=86, optimize=True)
        w_px, h_px = Image.open(jpg).size
        height = width * h_px / w_px
        if self.get_y() + height + 12 > self.page_break_trigger:
            self.add_page()
        x = (210 - width) / 2
        self.fig += 1
        self.image(str(jpg), x=x, y=self.get_y(), w=width)
        self.set_y(self.get_y() + height + 1.5)
        self.set_font("D", "", 8.5)
        self.set_text_color(*GREY)
        self.multi_cell(
            0, 4.5, f"Figure {self.fig}. {caption}", align="C", new_x="LMARGIN", new_y="NEXT"
        )
        self.set_text_color(0, 0, 0)
        self.ln(4)


def toc_renderer(pdf: FPDF, outline):
    pdf.set_x(MARGIN)
    pdf.set_font("D", "B", 20)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 12, "Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)
    for sec in outline:
        if sec.level > 0:
            continue
        pdf.set_font("D", "B" if sec.level == 0 else "", 10 if sec.level == 0 else 9)
        pdf.set_x(MARGIN + 6 * sec.level)
        link = pdf.add_link(page=sec.page_number)
        pdf.cell(150 - 6 * sec.level, 6, sec.name, link=link)
        pdf.cell(0, 6, str(sec.page_number), align="R", link=link, new_x="LMARGIN", new_y="NEXT")


def build(shots: Path, out: Path) -> None:
    m = Manual(shots)
    m.add_page()
    # ---- cover -----------------------------------------------------------------------------
    m.ln(30)
    m.set_font("D", "B", 34)
    m.set_text_color(*BLUE)
    m.multi_cell(0, 15, "TinyML Studio", align="C", new_x="LMARGIN", new_y="NEXT")
    m.set_font("D", "", 16)
    m.set_text_color(0, 0, 0)
    m.multi_cell(0, 9, "User Manual", align="C", new_x="LMARGIN", new_y="NEXT")
    m.ln(4)
    m.set_font("D", "", 11)
    m.set_text_color(*GREY)
    m.multi_cell(0, 6, f"Version {VERSION}", align="C", new_x="LMARGIN", new_y="NEXT")
    m.ln(10)
    m.set_text_color(0, 0, 0)
    m.shot("m-step6", "The model and training step, where most of the work happens.", width=118)
    m.set_font("D", "", 9.5)
    m.multi_cell(
        0,
        5,
        "Turn sensor data into a model that fits on your microcontroller.\n"
        "Free software under the GNU General Public License v3.0.\n"
        "Copyright (C) 2026 BrianGTRxx. Developed with the assistance of Claude (Anthropic).",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    m.cover_done = True
    m.fig = 0

    m.add_page()
    m.insert_toc_placeholder(toc_renderer, pages=1)
    m.fresh = True  # the placeholder already moved to a new page

    # =========================================================================================
    m.chapter(1, "What TinyML Studio is")
    m.p(
        "TinyML Studio is a desktop application that guides you from a folder of sensor recordings to a small "
        "machine learning model that runs on a microcontroller such as the ESP32-S3. It takes care of the parts "
        "that are easy to get wrong: cleaning the data, splitting it without leaking information, training several "
        "models, shrinking the result, and checking that it fits in the memory of your board before you flash anything."
    )
    m.h2("1.1 Concepts you need first")
    m.term(
        "Machine learning model",
        "A mathematical function that was fitted to examples. You show it many rows of "
        "inputs together with the correct answer, and it learns to produce the answer for inputs it has never seen.",
    )
    m.term(
        "TinyML",
        "Machine learning on devices with very little memory and no operating system, such as a "
        "microcontroller. The model must be small (kilobytes), fast, and must not need a network connection.",
    )
    m.term(
        "Feature",
        "One input column the model reads, for example the current, the active power or the power factor.",
    )
    m.term(
        "Target",
        "The thing you want to predict. If it is a category (the name of an appliance) the problem is "
        "classification. If it is a number (an exam score) the problem is regression.",
    )
    m.term(
        "Flash and RAM",
        "A microcontroller keeps the program and the model constants in flash (permanent, large) "
        "and the working values in RAM (temporary, small). A model that does not fit in either cannot run on the board.",
    )
    m.term(
        "Training, validation and test sets",
        "The data is divided into three parts. The model learns from the "
        "training part, you compare models and tune settings on the validation part, and the test part is looked "
        "at once, at the end, to report an honest result. Looking at the test part to make decisions makes the "
        "result look better than it is.",
    )
    m.term(
        "Data leakage",
        "Any way in which information from the validation or test data reaches the training "
        "process. Leakage produces beautiful numbers that disappear in the real world. TinyML Studio is built "
        "to avoid the common forms of it, and warns you about the ones it cannot prevent.",
    )
    m.h2("1.2 The seven steps")
    m.p(
        "The left sidebar shows the workflow. A step is unlocked when the one before it is complete."
    )
    m.table(
        ["Step", "Name", "What you do there"],
        [
            ["1", "Dataset", "Open your CSV files and let the app understand them."],
            ["2", "Task", "Say what to predict and what each column is."],
            ["3", "Cleaning", "Fix missing and absurd values."],
            ["4", "Augmentation", "Optional: create new training rows from your real rows."],
            ["5", "Split", "Divide the data into training, validation and test."],
            ["6", "Model and training", "Train one model or compare many, and check the quality."],
            ["7", "Board and export", "Pick your board, check that it fits and take the code."],
        ],
        [14, 42, 114],
    )
    m.h2("1.3 What it produces")
    m.p(
        "At the end you obtain C code you can compile into your firmware, optionally a TensorFlow Lite file, an ONNX "
        "file, and a ready to open Arduino sketch. Every export is compared with the original model on your test "
        "data, so you know the version on the board gives the same answers as the one you trained."
    )
    m.note(
        "Note.",
        "Everything runs on your computer. Your data is never uploaded anywhere. The only network use is "
        "downloading Python packages when you ask the app to install them.",
    )

    # =========================================================================================
    m.chapter(2, "Installing and first run")
    m.h2("2.1 Installing")
    m.p(
        "Download the installer from the Releases page of the project. On Windows run the Setup program, or unzip the "
        "zip version and start TinyML Studio.exe. On Linux make the AppImage executable and run it, or install the "
        "deb package. Python and Node.js are not required to use the installed application."
    )
    m.note(
        "Warning.",
        "The Windows installer is not code signed. Windows SmartScreen may show a warning the first "
        "time. Choose More info, then Run anyway, if you trust the download.",
    )
    m.h2("2.2 New versions")
    m.p(
        "The first time you start the application it asks once whether you want to be told when a new version is "
        "available. If you say yes, it asks GitHub, about once a day, whether a newer release exists. Nothing about "
        "you or your files is sent. If a newer version exists, a notice appears at the top with three choices: See "
        "what is new and download, which opens the download page in your browser, Not now, and Skip this version. "
        "The application never downloads or installs anything by itself, so you always decide."
    )
    m.p(
        "The page Updates, in the sidebar, shows your version and a button Check for updates that you can press at any "
        "time, with or without the automatic notice. To update, run the new installer over the old one. It replaces "
        "the program and keeps your settings. There is no need to uninstall first."
    )
    m.h2("2.3 The Python environment")
    m.p(
        "The application is small because it does not include the scientific Python packages. The first time you "
        "start it, it needs a Python environment. There are two ways to get one, and the Environment page (in the "
        "sidebar, under Tools) explains both."
    )
    m.term(
        "Option 1, install for me (recommended)",
        "The app downloads its own private copy of Python and only the "
        "components you tick: the core (always), the terminal interface, Keras with TensorFlow, PyTorch, and TFLite "
        "Micro. Nothing else on your computer is touched. Keras and PyTorch with GPU support are large, about two "
        "gigabytes each, so tick them only if you need them.",
    )
    m.term(
        "Option 2, use my own environment",
        "You point the app to a Python you already have (a virtual "
        "environment, conda or miniconda). The app checks what is missing and, after asking you, installs only "
        "the missing pieces. Packages that already satisfy the required range are not touched.",
    )
    m.shot(
        "m-env",
        "The Environment page. Look for Python on this computer lists the interpreters already installed.",
    )
    m.p(
        "The button Look for Python on this computer scans your PATH and the usual install locations, and shows "
        "each Python with a verdict. Versions 3.11 and 3.12 are recommended, 3.10 works, anything older than 3.10 "
        "cannot be used, and 3.13 or newer is marked as newer than tested because some libraries, TensorFlow in "
        "particular, may not support it yet."
    )
    m.note(
        "Important.",
        "The app never upgrades or changes your Python. Adding packages to an environment you "
        "already use can change library versions that your other projects depend on. A clean virtual "
        "environment, or the private Python that the app installs, avoids that risk.",
    )
    m.p(
        "What if you choose both? Only one environment runs the core at a time. Choosing your own environment "
        "replaces the private one until you press Stop using my environment."
    )

    # =========================================================================================
    m.chapter(3, "The main window")
    m.shot(
        "m-step1",
        "The main window: workflow on the left, the current step in the center, navigation at the bottom.",
    )
    m.p(
        "The sidebar lists the seven steps. Below it are the tools: History (past runs), Environment, PDF report and "
        "Tutorial (the guided tour). The top bar has Load recipe and Save recipe, the language switch (English or "
        "Spanish), and the theme switch (light, dark or follow the system). The bar at the bottom moves between steps."
    )
    m.h2("3.1 Recipes")
    m.p(
        "Everything you configure is stored in a recipe, a small JSON file. Save recipe writes it, and Load recipe "
        "restores it, so you or another person can repeat exactly the same run on the same or another dataset. "
        "Recipes contain settings and file paths, not your data."
    )
    m.h2("3.2 Language and theme")
    m.p(
        "You can switch language at any time. Messages produced by the core, such as warnings and reports, are "
        "translated together with the interface, and results that are already on screen are translated without "
        "running them again. Names of columns and your own data are never translated."
    )

    # =========================================================================================
    m.chapter(4, "Step 1: Dataset")
    m.shot(
        "m-step1-open",
        "Step 1 after opening a folder of recordings, with the file table and quality alerts.",
    )
    m.p(
        "This step loads your CSV files. The application reads files exported from InfluxDB, files that use a comma "
        "as the decimal mark, and files in the latin-1 encoding, and it detects the separator and the encoding by itself."
    )
    m.h2("4.1 Opening data")
    m.term(
        "Open CSV",
        "Opens the system file dialog. Select one file, or several with Ctrl or Shift. Use this when "
        "you want exactly some files out of a folder. Once data is loaded the same button is called Change CSV files.",
    )
    m.term(
        "Open folder",
        "Takes every CSV in the folder. Use this when all the files belong to the project.",
    )
    m.term(
        "Recent",
        "Under the buttons, the last data sets you opened appear as small buttons. One click opens the same files "
        "again, and the app checks that they still exist.",
    )
    m.p(
        "If you pick files that have different columns, the app merges columns with the same or similar names, and "
        "shows the groups it merged so you can check them. A column that exists in only some of the files is marked "
        "as missing for the others."
    )
    m.h2("4.2 What you see")
    m.numbered(
        [
            "The number of files, rows and unified columns.",
            "A card that tells you whether the files belong together. Same structure means they merge cleanly. Similar "
            "means some columns exist only in some files. Different kinds of data (for example electrical measurements and "
            "soil quality) are reported as groups, and you can continue with one group only.",
            "The file table with rows, columns, encoding and separator of each file.",
            "Quality alerts sorted by severity: columns with many missing values, absurd values, sentinel values, "
            "duplicated rows, and columns that identify the file of origin. An alert also appears when two files "
            "seem measured differently: a column with a very different scale (another calibration or other units) "
            "or with the opposite sign (another sign convention).",
        ]
    )
    m.note(
        "If you ignore the compatibility card.",
        "Nothing is blocked. But a model trained on a mix of "
        "unrelated measurements usually learns the difference between the files and not what you care about. "
        "Mix files only when the study needs it.",
    )
    m.note(
        "Tip.",
        "A column that identifies the recording, such as a sensor name that is different in every file, "
        "is flagged. If it stays as an input, the model can recognise the file instead of the appliance, and the "
        "results look better than they are.",
    )

    # =========================================================================================
    m.chapter(5, "Step 2: Task")
    m.shot(
        "m-step2",
        "Step 2: the task type, the label source and the table of columns with their roles.",
    )
    m.p(
        "Here you tell the application what to predict and what each column means. Everything shown is a suggestion "
        "that you can change."
    )
    m.h2("5.1 Task type")
    m.term(
        "Classification",
        "The answer is a category, such as kettle, fan or lamp. The model outputs the "
        "category. Choose this to recognise appliances.",
    )
    m.term(
        "Regression",
        "The answer is a number, such as an exam score or a temperature. Choose this to estimate a value.",
    )
    m.term(
        "No label",
        "There is nothing to learn from, so supervised models cannot be trained. Use the anomaly "
        "detector in step 6, or choose a target.",
    )
    m.p(
        "If you choose classification but the target column holds a continuous number, the classes would be thousands "
        "of different values and the result would be meaningless. If you choose regression for a category stored as "
        "text, the app cannot use it as a number. When in doubt, follow the suggestion at the top of the page."
    )
    m.h2("5.2 Label source")
    m.p(
        "Turn on The label is the file name when each file contains one class, for example one file per appliance. "
        "Numbers and suffixes in the file name are removed, so kettle_1.csv and kettle_2.csv both become the class "
        "kettle. Leave it off when a column of the table holds the class, and pick that column as the dependent variable."
    )
    m.h2("5.3 Roles of the columns")
    m.table(
        ["Role", "Meaning", "Effect"],
        [
            [
                "Independent, input (X)",
                "A column the model reads.",
                "Becomes an input of the model and of the firmware.",
            ],
            ["Dependent, target (y)", "The column to predict.", "Only one per model."],
            [
                "Group",
                "Identifies a recording or session.",
                "Used to split by group and to validate on unseen sessions.",
            ],
            ["Time", "A timestamp.", "Used for temporal splits. Never an input."],
            ["Metadata (unused)", "Information you keep but do not use.", "Ignored by the model."],
            ["Discard", "Remove the column.", "Ignored and removed from the data."],
        ],
        [42, 58, 70],
    )
    m.p(
        "The quick buttons help with many columns. Use all usable columns as inputs makes every numeric and "
        "categorical column an input. Numeric columns only keeps numbers. Back to suggestions restores the proposal."
    )
    m.shot("m-step2-open", "Choose exactly which columns to use: tick only the inputs you want.")
    m.p(
        "Choose exactly which columns to use opens a list of checkboxes. Tick the columns the model must read and "
        "press Use only the ticked columns. Every other usable column becomes unused, and the target is not touched."
    )
    m.note(
        "If you select a column but not its relatives.",
        "Every input you keep must also be available on the "
        "board. If an input is computed by your firmware from other readings, make sure you can compute the "
        "same value there. If you remove an input, the exported code no longer asks for it.",
    )
    m.h2("5.4 Encoding of categories")
    m.p(
        "A model reads numbers. A column of categories (for example Low, Medium, High) is converted. One-hot creates "
        "one yes or no column per category. Dummy is the same without one redundant column. Ordinal uses one number "
        "per category and suits categories that have a natural order. The limit of categories per column protects "
        "you from turning an identifier into hundreds of columns."
    )

    # =========================================================================================
    m.chapter(6, "Step 3: Cleaning")
    m.shot(
        "m-step3", "Step 3: each cleaning step comes from an alert and can be unticked or adjusted."
    )
    m.p(
        "Real recordings contain gaps and impossible values. This step proposes a list of cleaning operations, each "
        "one created from a quality alert, and lets you change them."
    )
    m.table(
        ["Operation", "What it does", "When to keep it"],
        [
            [
                "Absurd values to missing",
                "Marks values far outside the usual range as missing.",
                "Almost always. A power factor of 9 is a reading error.",
            ],
            [
                "Fill missing values",
                "Replaces gaps by the median, mean, mode or the previous value.",
                "When a few values are missing. Median is robust.",
            ],
            [
                "Scale",
                "Rescales columns (z-score, min-max or robust).",
                "For neural networks and support vector machines. Trees do not need it.",
            ],
            [
                "Clip outliers",
                "Limits values to a range.",
                "When rare extreme values distort the model.",
            ],
            [
                "Replace sentinel values",
                "Turns codes such as -999 into missing.",
                "When your instrument writes such codes.",
            ],
            [
                "Remove duplicates",
                "Deletes repeated rows, in training only.",
                "When the logger repeats rows.",
            ],
        ],
        [46, 70, 54],
    )
    m.h2("6.1 No leakage")
    m.p(
        "Every number a step needs (the median, the mean, the limits) is computed from the training set only, and "
        "then applied unchanged to validation and test. This is why you can trust the validation numbers. The same "
        "numbers are exported to the firmware, so the board cleans its inputs exactly as the training did."
    )
    m.note(
        "Important.",
        "The column you want to predict is never scaled. In a regression, the model and the firmware "
        "answer in real units (points, degrees), not in standardized values.",
    )
    m.h2("6.2 If you change roles after the suggestion")
    m.p(
        "When you change a role in step 2, a warning says the suggestion may be outdated. Press Regenerate suggestion "
        "to recompute it. If you do not, steps may still refer to columns you no longer use."
    )
    m.p(
        "Apply and see before and after runs the cleaning and shows charts for each column. Every chart can be "
        "downloaded as PNG, JPG or SVG with the button above it. Statistics are shown in original units."
    )

    # =========================================================================================
    m.chapter(7, "Step 4: Augmentation")
    m.shot("m-step4", "Step 4 is optional: it creates new training rows from your real rows, at your own responsibility.")
    m.p(
        "Augmentation means making new rows from the rows you already have, when you have few. It is not the "
        "same as filling missing values in step 3: there the gaps of existing rows are filled, here whole new "
        "rows are made."
    )
    m.note(
        "Under your responsibility.",
        "Creating data is not recommended in general. If you use it, you are the one who decides that the "
        "result is valid. TinyML Studio does not check it for you.",
    )
    m.h2("7.1 The two methods")
    m.term(
        "Copies with a little noise",
        "Each new row is a real row plus a small random noise in every numeric column. The noise is a fraction "
        "of the deviation of that column. The new rows stay very close to real ones.",
    )
    m.term(
        "Between neighbouring rows",
        "Each new row lies between a real row and one of its closest neighbours of the same class. For a "
        "regression, the target is interpolated too. The idea is the same as SMOTE.",
    )
    m.p(
        "You can ask for a number of new rows per real row, or, for classification, only as many as are needed to "
        "equalize the classes. Only the training set grows. Validation and test always keep the real rows."
    )
    m.h2("7.2 An example")
    m.p(
        "You have 3 real students (study hours, score): A (10, 60), B (12, 66) and C (30, 90). With noise, A can "
        "become (10.3, 59.4) and then (9.8, 60.5). Between neighbours, A and B are close, so a new student (11, 63) "
        "appears between them, while C is far from both and is not mixed with them."
    )
    m.h2("7.3 The risks")
    m.numbered(
        [
            "New rows add no information that is not already in your data.",
            "They can make a model look better than it is. Trust only validation and test, which stay real. "
            "Cross-validation inside training becomes optimistic.",
            "Noise or interpolation can create rows that could never happen in real life.",
            "Outliers and mistakes in your data are copied too. Clean first, in step 3.",
            "Interpolating between classes that overlap blurs the border between them.",
            "When you can, measuring more real data is always better than creating new rows.",
            "It helps most with very few rows, strongly unbalanced classes, or a model that memorizes the "
            "training set.",
        ]
    )
    m.p("What if you skip it? Nothing is lost. The rest of the workflow is the same, with real rows only.")

    # =========================================================================================
    m.chapter(8, "Step 5: Split")
    m.shot("m-step5", "Step 5: the split strategy and the size of each part.")
    m.p(
        "The data is divided into training (80 percent by default), validation (10 percent) and test (10 percent). "
        "How you divide matters more than most settings."
    )
    m.table(
        ["Strategy", "How it divides", "Use it when"],
        [
            [
                "Temporal blocks",
                "Keeps time order: earlier rows train, later rows validate and test, inside each group.",
                "Data recorded in sequence. Consecutive rows are almost identical, so a random split would leak.",
            ],
            [
                "By group",
                "Whole recordings go to one part.",
                "You have several recordings and want to test on unseen ones.",
            ],
            [
                "Stratified",
                "Random, but every class keeps its proportion.",
                "Rows are independent and classes are unbalanced.",
            ],
            ["Random", "Pure random rows.", "Rows are independent and the classes are balanced."],
        ],
        [32, 74, 64],
    )
    m.note(
        "If you pick random for time series.",
        "Neighbouring rows are nearly copies of each other. A random split "
        "puts some of them in training and their twins in test, and the accuracy looks excellent for the wrong "
        "reason. Follow the suggestion unless you know why not.",
    )
    m.p(
        "Check the sizes and the class distribution of every part. A class with fewer than twenty rows in the test "
        "part gives unreliable metrics, and the app warns you."
    )

    # =========================================================================================
    m.chapter(9, "Step 6: Model and training")
    m.shot(
        "m-step6",
        "Step 6: choose the model family, the models, the compute device and press Train.",
    )
    m.h2("9.1 Choosing models")
    m.term(
        "Classical models (scikit-learn)",
        "Decision tree, random forest, extra trees, gradient boosting, logistic "
        "regression, linear models, support vector machines, k nearest neighbours, naive Bayes and a small neural "
        "network. They train in seconds, are easy to export to C, and are often enough for sensor data.",
    )
    m.term(
        "Keras MLP and 1D CNN",
        "Neural networks. They need TensorFlow or PyTorch (installed from the Environment "
        "page) and export to TensorFlow Lite. A 1D CNN reads windows of consecutive samples.",
    )
    m.term(
        "Benchmark",
        "Trains many models on identical splits and ranks them without looking at the test set.",
    )
    m.p(
        "Each model in the list shows badges. C means it can be exported to plain C. Scale means it works better "
        "after scaling. If you pick a model that cannot be exported to C, you can still train and compare it, but "
        "you cannot flash it as C code."
    )
    m.h2("9.2 Compute device")
    m.p(
        "Automatic uses a graphics card when the Python environment can see one (NVIDIA with CUDA, or AMD where "
        "supported) and the processor otherwise. CPU forces the processor. Classical scikit-learn models always use "
        "the CPU. Small networks are often as fast on the CPU as on a GPU. The page Support for GPU on this computer "
        "tells you what was found and how to enable it."
    )
    m.h2("9.3 Reading the results")
    m.shot(
        "m-step6-open",
        "Step 6 with the extra cards expanded: unseen sessions, settings that fit the board, anomalies.",
        width=140,
    )
    m.term(
        "F1 macro",
        "The average over all classes of a score that balances precision and recall. It treats every "
        "class equally, so a rare class counts as much as a common one. It is the main number for classification.",
    )
    m.term(
        "Accuracy",
        "The fraction of correct answers. It can mislead when one class is much larger than the others.",
    )
    m.term(
        "Confusion matrix",
        "A table of what the model answered against what was true. The diagonal is correct.",
    )
    m.term(
        "R2 and MAE (regression)",
        "R2 is how much of the variation the model explains, 1 is perfect. MAE is the "
        "average error in the units of the target.",
    )
    m.term(
        "Baseline",
        "The score of always answering the most common class. A model must beat it clearly.",
    )
    m.p(
        "The tabs of the quality panel are Overview, Training curves, Confusion matrix, Per class and ROC / PR. A model "
        "that scores far higher in training than in validation is overfitting, which means it memorised."
    )
    m.h2("9.4 Does it work on a new session?")
    m.p(
        "The normal test comes from the same recordings as the training, so it measures how the model behaves later "
        "in the same capture. This card leaves one whole recording out, trains on the others, and measures on the "
        "recording the model never saw. A class can only be measured when it exists in at least two recordings."
    )
    m.p(
        "If the card reports that your data cannot tell you how the model behaves on a new session, it is true: "
        "with one recording per class, a high score cannot separate the appliance from the conditions of that "
        "recording. The button Mark session columns as metadata and repeat removes columns that identify the "
        "recording, and shows the result before and after."
    )
    m.h2("9.5 Settings that fit my board")
    m.p(
        "This card tries many settings of one model, measures how much flash and RAM each would use on the board "
        "with the same calculation as the export, discards those that do not fit, and keeps the most accurate that "
        "does. The choice uses validation only, and the test set is measured once for the winner. Use these settings "
        "when training passes them to the next training."
    )
    m.h2("9.6 Detect anomalies")
    m.p(
        "The anomaly detector learns only what is normal and raises an alarm for anything that does not look like "
        "it. Mark the classes that are normal. The other classes are used only to measure how many it catches, never "
        "to train or to set the threshold. If you mark none, every row counts as normal and only false alarms are "
        "measured. A fault the detector has never seen may go unnoticed."
    )
    m.h2("9.7 The hardware panel")
    m.p(
        "While training, the panel shows processor and memory load, GPU load, GPU memory and temperatures, a chart of "
        "the last seconds, and the estimated time left. Sensors your system does not expose are shown as not available, "
        "with the reason, instead of invented values."
    )

    # =========================================================================================
    m.chapter(10, "Step 7: Board and export")
    m.shot("m-step7", "Step 7: the board, the memory budget, the export button and the fit report.")
    m.h2("10.1 Pick the board and the budget")
    m.p(
        "The board picker lists the supported boards grouped by family, with variants such as the memory size of the "
        "module. A card shows the facts of the chip and where each number comes from. Facts marked as verified were "
        "read from official documents. The budgets you can use are our suggestion and not a manufacturer value."
    )
    m.p(
        "Flash available for the model and RAM available for the model are the amounts you can spend. Use suggested "
        "budgets fills them in. If you leave them empty, the report gives the size used but cannot say whether it fits."
    )
    m.note(
        "Remember.",
        "Your firmware also needs memory for the program, Wi-Fi, sensors and the stack. Leave margin.",
    )
    m.h2("10.2 Quantize and export")
    m.p(
        "The button Quantize and export converts the model, checks it, and produces the files. For classical models "
        "it writes plain C (tm_model.c and tm_model.h). Compact tables makes trees and forests smaller with the same "
        "predictions. For neural networks it writes TensorFlow Lite in float, int8 and float16, and the arrays for "
        "TensorFlow Lite Micro."
    )
    m.p(
        "The fit report shows two verdicts, flash and RAM. Fits, tight and does not fit are shown with the numbers. "
        "If it does not fit, the app lists ways out: a smaller model, fewer inputs, or a larger module."
    )
    m.h2("10.3 Verification")
    m.p(
        "A table compares the exported code with the original model on the test vectors: how many agree, the largest "
        "difference, and whether the code compiles. This was verified on the computer. The final proof is to run it "
        "on your board."
    )
    m.h2("10.4 What was this model trained with?")
    m.p(
        "A card at the top of step 7 lists, in one place, the files and the number of rows, the task, what the "
        "model predicts, the input columns, the model, the split, the class balance, the cleaning steps and any "
        "new rows from augmentation. Compare it with the data you will measure in real life, using the same units, "
        "the same sign and the same columns."
    )
    m.h2("10.5 Try the model with your own values")
    m.p(
        "For classical models, a card lets you type the values of each input and press Predict. It answers with the "
        "class and its confidence, or with the value for a regression. It uses the same cleaning and the same model "
        "as the exported C code, which was checked equal on the test vectors. It is not a test on the board."
    )
    m.p(
        "Under each box you see the range that column had in training. If you type a value outside that range, a "
        "warning appears, because the model extrapolates there and the answer may be unreliable. An empty box means "
        "missing, and the model uses the filler it learned."
    )
    m.p(
        "The exported C code has the same protection: tm_out_of_range(num) returns how many numeric inputs are "
        "outside the training range, and 0 means all fine. Call it in your firmware before trusting a prediction."
    )
    m.h2("10.6 Saving the Arduino sketch")
    m.p(
        "Nothing is saved on your computer until you press Save the Arduino sketch and choose a folder. The app then "
        "shows the exact path of the .ino file to open in the Arduino IDE. The button Save files saves everything the "
        "export produced."
    )
    m.shot(
        "m-step7-open",
        "Step 7 with every card expanded: ONNX, which columns matter, speed and unknown answers.",
        width=140,
    )
    m.h2("10.7 Extra tools of this step")
    m.term(
        "Export ONNX",
        "Writes a portable model for computers, servers and phones. Cleaning and encoding are "
        "inside the model, so it accepts raw values. It was compared with scikit-learn on your test data.",
    )
    m.term(
        "Which columns matter",
        "Shuffles one column at a time and measures how much the result drops. Columns that "
        "barely matter could be removed to save sensors and memory. Retrain without them to confirm.",
    )
    m.term(
        "How fast would it be (estimate)",
        "Counts the operations of the model and multiplies them by assumed "
        "cycles per operation and the clock of each board. The result is a range, not a measurement. Time the "
        "prediction on your board for the real figure.",
    )
    m.term(
        "Answer unknown when unsure",
        "A classifier always picks one of its classes, even for something it never "
        "saw. This sets a confidence threshold and checks it by retraining without one class at a time. The ONNX "
        "export can apply it. The C export does not apply it yet.",
    )

    # =========================================================================================
    m.chapter(11, "Using the model on an ESP32-S3")
    m.p("This chapter explains the path from the saved files to a running board.")
    m.numbered(
        [
            "Install the Arduino IDE and, in the Boards Manager, the esp32 platform from Espressif Systems.",
            "Press Save the Arduino sketch in step 7 and open the .ino file shown. The folder must have the same name "
            "as the .ino file.",
            "Choose the board ESP32S3 Dev Module, connect the board and press Upload.",
            "Open the Serial Monitor at 115200 baud. The example sketch prints pred= and expected=. They must match.",
            "Replace the example sample by your own readings and call tm_predict.",
        ]
    )
    m.h2("11.1 The function tm_predict")
    m.p(
        "tm_predict(num, cat) receives the raw values, in the order printed in the README of the sketch and in "
        "the input table of step 7, and returns the answer. For a classifier the answer is the index of the class "
        "in the list of class names. For a regression it is the predicted number. The function applies the same "
        "cleaning and encoding as the training, so give it raw values with the same units as your CSV files."
    )
    m.p("When the model has no categorical inputs, pass NULL as the second argument.")
    m.note(
        "What if the units differ?",
        "If the board measures a quantity in different units or with a different "
        "scale than the CSV used for training, the model cannot be right. The values must be produced the same "
        "way in both places.",
    )
    m.h2("11.2 Out of range inputs")
    m.p(
        "A model only knows the range of values it saw in training. Linear models continue the line outside that "
        "range, so a value far outside gives an answer that looks confident and is unreliable. Check ranges in your "
        "firmware and decide what to do when an input is outside them."
    )

    # =========================================================================================
    m.chapter(12, "Tools: history, reports and the terminal")
    m.shot(
        "m-history", "History lists past runs. Open one to compare it or reuse its configuration."
    )
    m.h2("12.1 History")
    m.p(
        "Every training and export is stored. History lets you compare runs, open the files of a past run, and reuse "
        "a configuration on a different dataset."
    )
    m.h2("12.2 PDF report")
    m.p(
        "Creates a report with the data summary, the cleaning, the split, the metrics and the charts of the last run, "
        "to attach to your documentation."
    )
    m.h2("12.3 Command line and terminal interface")
    m.p(
        "Everything the window does is also available in the terminal. The command tinyml-studio has subcommands such "
        "as inspect, clean, train, export, export-onnx, tune, importance, latency, reject and anomaly. Run it with "
        "the option help to list them. The terminal interface (tui) is a text mode version of the workflow."
    )
    m.h2("12.4 The guided tour")
    m.shot("m-tour", "The guided tour highlights each part of the window.", width=130)
    m.p(
        "The Tutorial button starts a tour of the window, and every page has a question mark button with tips."
    )

    # =========================================================================================
    m.chapter(13, "Troubleshooting")
    m.table(
        ["Problem", "Likely cause", "What to do"],
        [
            [
                "The window asks to set up the app.",
                "No Python environment yet.",
                "Choose option 1 or point to your own environment.",
            ],
            [
                "Steps are locked.",
                "The previous step is not complete.",
                "Open a dataset first, then continue in order.",
            ],
            [
                "The numbers are suspiciously perfect.",
                "Leakage, or one recording per class.",
                "Remove columns that identify the file. Check Does it work on a new session. Use a split by group.",
            ],
            [
                "Training does not use my GPU.",
                "Classical models use the CPU. The environment does not see the GPU.",
                "Read the GPU support card. Install the GPU build of PyTorch or TensorFlow.",
            ],
            [
                "The model does not fit.",
                "It is too large for the budget.",
                "Use Settings that fit my board, a smaller model or fewer inputs.",
            ],
            [
                "Nothing appears in the Serial Monitor.",
                "Wrong port or speed, or the line ending.",
                "Use 115200 baud. Close other programs that use the port.",
            ],
            [
                "Pressing Enter in the monitor does nothing.",
                "The line ending is set to No Line Ending.",
                "Choose Newline, or wait half a second after typing in the example survey.",
            ],
            [
                "The board returns strange values.",
                "Inputs in a different order or units.",
                "Follow the input table exactly.",
            ],
        ],
        [48, 56, 66],
    )

    # =========================================================================================
    m.chapter(14, "What is verified and what is not")
    m.p("This section is deliberately plain. It lists what has been tested and what has not.")
    m.h2("14.1 Verified")
    m.numbered(
        [
            "The workflow runs end to end in the desktop application on Windows 11.",
            "Exported C code gives the same answers as the Python model on thousands of test vectors, on the computer.",
            "A regression model exported to C ran on a real ESP32-S3 and returned the same values as Python, including an "
            "interactive survey sketch over the serial port.",
            "Training with an NVIDIA GPU through PyTorch on Windows 11.",
        ]
    )
    m.h2("14.2 Not verified yet")
    m.numbered(
        [
            "Classification models running on a board, and the speed or power use measured on a board.",
            "Training on AMD GPUs, and macOS.",
            "The installer on a clean Windows account, and the Linux packages on a real Linux system.",
            "The Arduino sketches for TensorFlow Lite Micro in the real IDE.",
        ]
    )
    m.p(
        "The speed figures of the application are estimates. Board facts that are not marked as verified come from "
        "secondary sources, and every field lists where it came from."
    )

    # =========================================================================================
    m.chapter(15, "Glossary")
    for name, text in [
        (
            "Anomaly detection",
            "Finding data that does not look like the normal examples, without examples of faults.",
        ),
        ("Arena", "The block of RAM that TensorFlow Lite Micro uses for working values."),
        ("Benchmark", "Training several models on the same splits to compare them fairly."),
        ("Class", "One of the categories a classifier can answer."),
        (
            "Cross validation",
            "Repeating training and validation on different parts of the data to measure stability.",
        ),
        ("Epoch", "One pass of a neural network over the whole training set."),
        ("Feature importance", "A measure of how much each input column matters to a model."),
        (
            "Float16, int8",
            "Smaller number formats. Int8 uses one byte per weight and suits microcontrollers.",
        ),
        ("Group", "A recording or session, used to split and validate without leakage."),
        (
            "Hyperparameter",
            "A setting of a model that you choose before training, such as the depth of a tree.",
        ),
        ("ONNX", "An open file format for models that many programs can run."),
        ("Overfitting", "A model that memorised the training data and fails on new data."),
        ("Quantization", "Storing a model with smaller numbers so it uses less memory."),
        ("Recipe", "The saved configuration of a whole run, as a JSON file."),
        ("TensorFlow Lite Micro", "A small runtime to run neural networks on microcontrollers."),
        (
            "Threshold",
            "A value above or below which the application takes a decision, such as answering unknown.",
        ),
    ]:
        m.term(name, text)

    # =========================================================================================
    m.chapter(16, "License and credits")
    m.p(
        "TinyML Studio is free software under the GNU General Public License version 3. You may use, study, modify "
        "and redistribute it. Any distributed derivative work must remain open under the same license, and must keep "
        "the notice of the original author. This license allows commercial use. It does not allow closing the code."
    )
    m.p(
        "Copyright (C) 2026 BrianGTRxx. The project was designed and directed by BrianGTRxx and "
        "developed with the assistance of Claude, an AI model by Anthropic, used through Claude Code. The author "
        "decided the goals, reviewed the results and is responsible for the project."
    )
    m.p(
        "The screenshots of this manual use synthetic recordings created for the purpose, not real measurements."
    )
    m.output(str(out))


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
