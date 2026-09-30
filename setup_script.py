#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import platform
import re
import shutil
import subprocess as sp
import zipfile
import os

from paneltime.options import option_schema

CUR_DIR = Path(__file__).resolve().parent


def write_option_docs(path):
    rows = [
        '---',
        'title: Fit options',
        'nav_order: 2',
        'has_toc: true',
        '---',
        '',
        '# Fit options',
        '',
        'These options are configured per model fit. The defaults, domains, and descriptions below are generated from the central option schema.',
        '',
        '| Group | Option | Default | Domain | Description | Example |',
        '|---|---|---:|---|---|---|',
    ]
    for option in option_schema():
        domain = ', '.join(map(str, option['domain'])) if isinstance(option['domain'], list) else option['domain']
        default = option['default'] if isinstance(option['default'], str) and option['default'].endswith('()') else repr(option['default'])
        example = option['example']
        rows.append(
            f"| {option['group']} | `{option['name']}` | `{default}` | {domain} | "
            f"{option['description']} | `{example}` |"
        )
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def run(cmd, cwd=CUR_DIR):
    print(f"\nRunning: {' '.join(cmd)}")
    sp.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build, render, publish and deploy paneltime.")
    parser.add_argument("-g", "--git", action="store_true", help="Push paneltime, paneltime.github.io and paneltime.sitegen to GitHub")
    parser.add_argument("-p", "--pypi", action="store_true", help="Upload package to PyPI")
    parser.add_argument("-k", "--keep-version", action="store_true", help="Do not increment patch version")
    parser.add_argument("-s", "--skip-quarto", action="store_true", help="Skip Quarto rendering ")

    args = parser.parse_args()

    clean()
    create_readme()
    zip_example()
    write_option_docs(CUR_DIR / "qmd" / "options.qmd")



    version = None
    if args.git or args.pypi:
        version = add_version(CUR_DIR, add=not args.keep_version)
        print(f"Version is now {version}")

    if not args.skip_quarto:
        run(["quarto", "render", "qmd"])
        
    build_package()

    if args.git or args.pypi:
        gitpush(version)
    else:
        print('Not pushed to GitHub. Use "-g" to push.')

    if args.pypi:
        os.system("twine upload dist/*")
    else:
        print('Not uploaded to PyPI. Use "-p" to upload.')


def clean():
    for folder in ["dist", "build", "paneltime.egg-info"]:
        shutil.rmtree(CUR_DIR / folder, ignore_errors=True)

    remove_pycache_dirs(CUR_DIR)


def build_package():
    python_cmd = "python3" if platform.system() == "Darwin" else "python"
    run([python_cmd, "-m", "build"])


def push_repo(path: Path, message: str):
    print(f"\nPushing repository: {path}")

    run(["git", "pull"], cwd=path)
    run(["git", "add", "."], cwd=path)

    result = sp.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        text=True,
        capture_output=True,
        check=True,
    )

    if not result.stdout.strip():
        print(f"No changes to commit in {path}")
    else:
        run(["git", "commit", "-m", message], cwd=path)

    run(["git", "push"], cwd=path)


def gitpush(version: str):
    reason = input("Write reason for commit: ").strip()
    message = f"Version {version} committed"
    if reason:
        message += f": {reason}"

    pages_repo = CUR_DIR.parent / "paneltime.github.io"

    push_repo(CUR_DIR, message)
    push_repo(pages_repo, message)



def add_version(wd: Path, add=True):
    srchtrm = r"(\d+\.\d+\.\d+)"

    version = re_replace(wd / "pyproject.toml", srchtrm, add=add)
    re_replace(wd / "qmd/index.qmd", srchtrm, version=version)
    re_replace(wd / "paneltime/info.py", srchtrm, version=version)

    return version


def re_replace(path: Path, searchterm: str, version=None, add=True):
    text = path.read_text(encoding="utf-8")
    match = re.search(searchterm, text, re.MULTILINE)

    if not match:
        raise RuntimeError(f"No version number found in {path}")

    if version is None:
        major, minor, patch = match.group(0).split(".")
        patch = str(int(patch) + int(add))
        version = ".".join([major, minor, patch])

    text = text[:match.start()] + version + text[match.end():]
    path.write_text(text, encoding="utf-8")

    return version


def create_readme():
    src = CUR_DIR / "qmd/index.qmd"
    dest = CUR_DIR / "README.md"

    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)

    if lines and lines[0].strip() == "---":
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        lines = lines[end + 1:]

    dest.write_text("".join(lines), encoding="utf-8")


def remove_pycache_dirs(root: Path):
    for path in root.rglob("__pycache__"):
        print(f"Removing {path}")
        shutil.rmtree(path, ignore_errors=True)


def zip_example():
    files = [
        "example.py",
        "wb.dmp",
        "loadwb.py",
        "mymodel.py",
    ]

    zip_path = CUR_DIR / "qmd/working_example.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name in files:
            file_path = CUR_DIR / "qmd" / name
            zipf.write(file_path, arcname=name)


if __name__ == "__main__":
    main()