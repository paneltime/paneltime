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
└── paneltime.sitegen/
```

## Typical workflow

Build the package and generate the website:

```bash
python setup_script.py
```

Generate and push GitHub repositories:

```bash
python setup_script.py -g
```

Generate, push, and upload to PyPI:

```bash
python setup_script.py -p
```

