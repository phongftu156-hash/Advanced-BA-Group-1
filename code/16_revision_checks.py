"""Text quality gate for the report and the deck: banned punctuation, voice, internal language, file names,
jargon outside the technical appendix, unrendered macros, and mean sentence length per section.

Scans report/main.tex and report/slides/pitch.tex (comments stripped, since they never print) and the
compiled PDFs. Writes result/logs/revision_checks.txt.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
OUT = ROOT / "result" / "logs" / "revision_checks.txt"
CHECKS = {
    "em dash or ---": r"—|(?<!-)---(?!-)",
    "en dash as punctuation ( – )": r" – ",
    "-- between letters": r"[A-Za-z]--[A-Za-z]",
    "we/our/us": r"(?i)\b(we|our|us)\b",
    "proposal": r"(?i)proposal",
    "second dataset": r"(?i)second (dataset|data source)",
    "multi-view": r"(?i)multi-view",
    "file extensions (.py .csv .parquet)": r"\.(py|csv|parquet)\b",
    "numbers.tex / run_all / code/ / result/ / data/": r"numbers\.tex|run_all|run\\_all|\bcode/|(?<!\.\./)\bresult/|\bdata/",
    "DECISIONS / random_state": r"DECISIONS|random_state|random\\_state",
    "banned adjectives": r"(?i)\b(crucial|vital|important|remarkable|groundbreaking|key|major|powerful|compelling|rich)\b",
    "significant (not statistically)": r"(?i)(?<!statistically )\bsignificant",
    "robust (not robust standard errors)": r"(?i)(?<![:{])\brobust\b(?! standard errors)",
    "quotation marks in prose": r"\"|``|''",
    "the model suggests / the data shows": r"(?i)the model suggests|the data shows",
    "[Team placeholder": r"\[Team",
}
APPENDIX_ONLY = {"out-of-fold": r"(?i)out-of-fold", "leakage": r"(?i)leakage", "Spearman": r"Spearman",
                 "isotonic": r"(?i)isotonic", "silhouette": r"(?i)silhouette", "adjusted Rand": r"(?i)adjusted Rand",
                 "Brier": r"Brier", "log-odds": r"(?i)log-odds"}


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in t.splitlines())


def body(t):
    """Document body only (the preamble holds style code and macro definitions)."""
    return t[t.index("\\begin{document}"):]


def main():
    lines, counts = [], {k: 0 for k in list(CHECKS) + list(APPENDIX_ONLY)}
    for f in (REP / "main.tex", REP / "slides" / "pitch.tex"):
        t = body(strip_comments(f.read_text(encoding="utf-8")))
        # the bibliography holds titles and outlet names, not prose
        t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t, flags=re.S)
        cut = t.find("\\section*{Technical appendix}")
        main_part = t if cut < 0 else t[:cut]
        for name, pat in CHECKS.items():
            for m in re.finditer(pat, t):
                counts[name] += 1
                lines.append(f"  {name}: {f.relative_to(ROOT)}: ...{t[max(0, m.start() - 40):m.end() + 40]!r}")
        for name, pat in APPENDIX_ONLY.items():
            for m in re.finditer(pat, main_part):
                counts[name] += 1
                lines.append(f"  {name} (outside appendix): {f.relative_to(ROOT)}: ...{main_part[max(0, m.start() - 40):m.end() + 40]!r}")
    report = ["Text checks (comments, preamble and bibliography excluded)", ""]
    total = 0
    for k, v in counts.items():
        report.append(f"{v:3d}  {k}")
        total += v
    import fitz
    for pdf in ("main.pdf", "response_to_feedback.pdf", "slides/pitch.pdf"):
        if not (REP / pdf).exists():
            continue
        text = "".join(p.get_text() for p in fitz.open(REP / pdf))
        bad = sorted(set(re.findall(r"\b[a-z]+[A-Z][A-Za-z]*\b", text)) - {"mogi"})
        bad += re.findall(r"\[Team|\?\?", text)
        report.append(f"{len(bad):3d}  camelCase tokens / ?? / [Team in {pdf}: {bad[:10]}")
        total += len(bad)
    # mean sentence length per section of the report body (floats removed)
    t = body(strip_comments((REP / "main.tex").read_text(encoding="utf-8")))
    t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    parts = re.split(r"\\section\*?\{([^}]*)\}", t)
    report += ["", "Mean words per sentence by section (prose only; macros count as one word):"]
    long_all = []
    for title, sec in zip(parts[1::2], parts[2::2]):
        s = re.sub(r"\\(subsection)\*?\{[^}]*\}|\\label\{[^}]*\}", " ", sec)
        s = re.sub(r"~?\\(ref|cite|citep|citet|citeyearpar)\{[^}]*\}", " 1", s)
        s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^]]*\])?|\\item|\\textbf|\\emph|\\textit", " ", s)
        s = re.sub(r"\\[A-Za-z]+(\{\})?", "N", s).replace("{", " ").replace("}", " ")
        s = re.sub(r"\s+", " ", s)
        sents = [x for x in re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9(])", s) if len(x.split()) >= 3]
        if not sents:
            continue
        n = [len(x.split()) for x in sents]
        long_ = [x for x, k in zip(sents, n) if k > 30]
        long_all += [(title, x) for x in long_]
        report.append(f"  {title[:34]:34s} {sum(n) / len(n):5.1f} words, {len(long_)} over 30 words")
    for title, x in long_all:
        report.append(f"    long sentence in {title}: {x[:150]}")
    report += ["", f"TOTAL text-rule hits: {total}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(report + ["", "Details:"] + lines) + "\n", encoding="utf-8")
    print("\n".join(report))
    if lines:
        print("\n".join(lines[:40]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
