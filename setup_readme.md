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
- GitHub Pages generation
- copying generated pages into `paneltime.github.io`
- Git commits and GitHub push operations
- optional PyPI uploads


## Expected directory structure

```text
parent/
├── paneltime/
├── paneltime.github.io/
```

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
The web page is generated in the qmd directory in this folder, with 
```
quarto render
```
This generate html files in `paneltime.github.io`, which hosts the website.

You can manually change files under qmd, run `quarto render` there and push 
`paneltime.github.io` to publish the changes.
