# paneltime

This repository uses:

```text
setup_script.py
```

to manage the entire project workflow.

The script handles:

- package building
- installation and setup
- Quarto documentation generation
- LaTeX PDF compilation and copying into the website
- GitHub Pages generation
- copying generated pages into `paneltime.github.io`
- Git commits and GitHub push operations
- optional PyPI uploads


## Expected directory structure

```text
parent/
├── paneltime/
├── paneltime_doc/
├── paneltime.github.io/
```

## PDF documentation and links

The LaTeX source in `../paneltime_doc/main.tex` enables clickable links,
a table of contents, and section/subsection bookmarks through `hyperref`.
No special compiler flag is needed for links. The PDF requests that viewers
open their bookmarks sidebar, although viewer preferences may override this.

The normal setup workflow automatically compiles this document and copies
`main.pdf` to `qmd/paneltime_documentation.pdf` before rendering the website.
To rebuild and copy only the PDF, without cleaning, building the package,
rendering Quarto, changing versions, or publishing:

```bash
python setup_script.py --docs-only
```

Install MiKTeX (Windows), MacTeX (macOS), or TeX Live (Linux), with the
document's LaTeX packages available. The script prefers `latexmk -pdf`,
which automatically repeats compilation until the contents, links and
references are up to date. Without `latexmk`, it runs PDFLaTeX, BibTeX,
and two further PDFLaTeX passes. Compilation errors stop the workflow
before the PDF is copied.

If no suitable compiler is found, the script warns and copies an existing
`main.pdf` if available; this does not rebuild or refresh its links.

## Typical workflow

Build the package and generate the website:

```bash
python setup_script.py
```

Build and generate website, and push GitHub repositories:

```bash
python setup_script.py -g
```

Build and generate website, and upload to PyPI:

```bash
python setup_script.py -p
```

## Web generation - under the hood
The normal setup workflow generates `qmd/options.qmd` and the Effects and
OptimizerOptions tables included in `qmd/api/fit-options.qmd` from the central
schema in `paneltime/options.py`. Change option defaults, permitted-value
descriptions, and explanations in that schema, not in generated tables.
Usage examples are maintained in the API page. The `--docs-only` command
only rebuilds the PDF; it does not regenerate or render web option tables.

The web page is generated in the qmd directory in this folder, with 
```
quarto render
```
This generate html files in `paneltime.github.io`, which hosts the website.

You can manually change files under qmd, run `quarto render` there and push 
`paneltime.github.io` to publish the changes.
