import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import setup_script


class DocumentationBuildTests(unittest.TestCase):
    def test_latexmk_builds_and_copies_pdf(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "paneltime"
            (package / "qmd").mkdir(parents=True)
            doc_dir = root / "paneltime_doc"
            doc_dir.mkdir()
            (doc_dir / "main.pdf").write_bytes(b"test PDF")
            with (
                patch.object(setup_script, "CUR_DIR", package),
                patch.object(setup_script.shutil, "which", side_effect=lambda name, **kw: name),
                patch.object(setup_script, "run") as run,
            ):
                setup_script.compile_latex_doc()
            run.assert_called_once_with(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=doc_dir,
            )
            self.assertEqual((package / "qmd" / setup_script.DOC_PDF).read_bytes(), b"test PDF")

    def test_pdflatex_fallback_repeats_passes(self):
        with (
            patch.object(setup_script.shutil, "which",
                         side_effect=lambda name, **kw: None if name == "latexmk" else name),
            patch.object(setup_script, "run") as run,
            patch.object(setup_script.shutil, "copy2"),
            patch.object(Path, "exists", return_value=True),
        ):
            setup_script.compile_latex_doc()
        doc_dir = setup_script.CUR_DIR.parent / "paneltime_doc"
        latex = call(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=doc_dir,
        )
        self.assertEqual(run.call_args_list, [
            latex, call(["bibtex", "main"], cwd=doc_dir), latex, latex,
        ])

    def test_compile_error_prevents_copying(self):
        with (
            patch.object(setup_script.shutil, "which", return_value="latexmk"),
            patch.object(setup_script, "run",
                         side_effect=subprocess.CalledProcessError(1, ["latexmk"])),
            patch.object(setup_script.shutil, "copy2") as copy,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                setup_script.compile_latex_doc()
        copy.assert_not_called()

    def test_docs_only_skips_full_workflow(self):
        with (
            patch.object(setup_script.sys, "argv", ["setup_script.py", "--docs-only"]),
            patch.object(setup_script, "compile_latex_doc") as compile_doc,
            patch.object(setup_script, "clean") as clean,
            patch.object(setup_script, "build_package") as build,
            patch.object(setup_script, "run") as run,
        ):
            setup_script.main()
        compile_doc.assert_called_once_with()
        clean.assert_not_called()
        build.assert_not_called()
        run.assert_not_called()

    def test_docs_only_rejects_publishing(self):
        for flag in ("--git", "--pypi"):
            with (
                self.subTest(flag=flag),
                patch.object(setup_script.sys, "argv",
                             ["setup_script.py", "--docs-only", flag]),
                patch.object(setup_script, "compile_latex_doc") as compile_doc,
                patch("sys.stderr"),
            ):
                with self.assertRaises(SystemExit) as error:
                    setup_script.main()
                self.assertEqual(error.exception.code, 2)
                compile_doc.assert_not_called()
