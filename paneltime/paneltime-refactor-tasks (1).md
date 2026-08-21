# Paneltime Refactor — Task Lists

Context: `paneltime` is a Python package for panel/ARIMA/GARCH regression
(github.com/paneltime/paneltime, docs at paneltime.github.io). Its API,
output object, and options currently deviate from ecosystem conventions
(`statsmodels`, `linearmodels`, `arch`). These two task lists implement a
convention-aligned rewrite: Task List 1 covers the Python source, Task
List 2 covers the Quarto (`.qmd`) docs site. 


You are currently in the source code directory, so you should only **ONLY do source code changes**

That is, you should **ONLY DO Task list 1**


---

## Task List 1 — Source code

### 1.1 Public API surface: `execute()` → constructor + `.fit()`
- [ ] Add a `PanelARIMAGARCH` (naming TBD — could also be `Paneltime`)
      class with signature
      `__init__(self, formula, data, entity=None, time=None)`,
      accepting either a `pandas.DataFrame` with a `(entity, time)`
      `MultiIndex`, or flat columns named via `entity=`/`time=`.
- [ ] Add `.fit(order=, garch_order=, vol=, effects=, cov_type=,
      optimizer=, constraints=, likelihood=, h_function=)` per the
      grouping in §1.3.
- [ ] Keep `pt.execute(...)` as a thin deprecated wrapper that
      constructs the class and calls `.fit()` internally, emitting a
      `DeprecationWarning` pointing at the new API. Do not remove it in
      this pass.
- [ ] Rename `T`/`ID`/`HF` params (in the deprecated wrapper's
      docstring and anywhere still user-facing) to `time`/`entity`/
      `het_factors` in all new-path documentation and error messages.

### 1.2 Results object
- [ ] Create a `Results` class replacing `Summary`, exposing at the
      top level: `params`, `bse`, `tvalues`, `pvalues`, `conf_int()`,
      `nobs`, `df_resid`, `llf`, `aic`, `bic`, `converged`,
      `random_effects` (a small object with `.group`/`.time` for
      residuals and std devs), `resid`, `fittedvalues`, `summary()`.
- [ ] `params`/`bse`/`tvalues`/`pvalues` must be `pandas.Series`
      indexed by variable name — derive the index from the existing
      `names.captions`/`varnames` internally so that public consumers
      never need `results.names.*` directly.
- [ ] Move current `general.*` optimizer internals (`hessian`,
      `gradient_matrix`, `gradient_vector`, `dx_norm`, `its`, `msg`,
      `t0`, `t1`, `log_likelihood_object`, `ci`, `ci_n`) into a
      `results.optim_result` sub-object (or leading-underscore
      attributes if no separate object is preferred).
- [ ] Rename `output`/`table` to `_output`/`_table` (private) since the
      existing docs already describe them as internal-only.
- [ ] Add `predict()`/`forecast(steps=...)` if not already present in
      some internal form — check current codebase for equivalent
      functionality before adding.
- [ ] Write docstrings on every public `Results` attribute in
      NumPy/Google style so Task List 2 §2.4's page can be
      auto-generated from source.

### 1.3 Options / configuration
- [ ] Replace the flat `pt.options` global singleton with per-call
      configuration, grouped as:
  - `order: tuple[int, int, int]` (ARIMA p, d, q) — replaces
    `pqdkm[:3]`
  - `garch_order: tuple[int, int]` (GARCH k, m) — replaces `pqdkm[3:]`
  - `vol: str` (`'GARCH'` / `'EGARCH'`) — replaces boolean `EGARCH`
  - `effects: Effects` dataclass with `group`/`time`/`variance` fields
    taking `'none'|'fixed'|'random'` — replaces
    `fixed_random_group_eff`/`fixed_random_time_eff`/
    `fixed_random_variance_eff` integer triple
  - `optimizer: OptimizerOptions` dataclass bundling `tolerance`,
    `max_iterations`, `accuracy`, `use_analytical` (rename to
    `use_analytical_hessian`), `constraints_engine`,
    `initial_arima_garch_params`, `ARMA_constraint` (rename
    `arma_constraint`), `ARMA_round` (rename `arma_round`),
    `GARCH_min` (rename `garch_min`), `GARCH_assist` (rename
    `garch_assist`), `multicoll_threshold_max`,
    `multicoll_threshold_report`, `min_group_df`,
    `robustcov_lags_statistics` (rename `robust_cov_lags`),
    `variance_RE_norm` (rename `variance_re_norm`), `kurtosis_adj`
  - `constraints` — replaces `user_constraints`
  - `add_intercept: bool`, `subtract_means: bool`,
    `include_initvar: bool`, `tobit_limits: tuple`,
    `supress_output` (fix typo → `suppress_output`)
  - `likelihood` — replaces `custom_model` (see 1.4)
  - `h_function` — replaces manual `h_val`/`h_val_cpp` pairing (see 1.5)
- [ ] Normalize every renamed attribute to snake_case consistently.
- [ ] Keep `pt.options` as a deprecated module-level object whose
      attribute assignments forward into a per-call default config
      used by the deprecated `pt.execute()` wrapper, so existing user
      scripts keep working during the deprecation window. Emit
      `DeprecationWarning` on first attribute set.
- [ ] Add validation at `.fit()`-call time for every option's
      "Permissible values" constraint currently only documented in
      prose (e.g. `%s>0` patterns) — raise `ValueError` with a clear
      message instead of allowing silent bad input through to the
      optimizer.

### 1.4 Custom likelihood models
- [ ] Add an abstract base class `LikelihoodModel` (exact name TBD)
      with abstract methods `variance_bounds(self, init_var)`,
      `variance_definitions(self)`, `set_h_function(self)`,
      `loglike(self)` (was `ll`), `score(self)` (was `dll`),
      `hessian(self)` (was `ddll`), and an `__init__` with a fixed,
      documented signature.
- [ ] Add a numerical-differentiation fallback: if a subclass doesn't
      override `score`/`hessian`, compute them via finite differences
      of `loglike` (mirrors `statsmodels.GenericLikelihoodModel`), so
      new users can start with just `loglike`.
- [ ] Change the assignment/usage pattern so users pass an **instance**
      to `model.fit(likelihood=MyModel())` — the framework must accept
      and use a pre-built instance directly, not construct one
      internally from a class reference. Remove the "do not instantiate
      it yourself" constraint entirely if feasible; if internal
      re-construction is required for technical reasons (e.g. per-group
      instances), document why explicitly rather than leaving it as an
      unexplained rule.
- [ ] Add runtime validation (e.g. `abc.abstractmethod` /
      `typing.Protocol` checks) that raises a clear error at model
      construction time if a required attribute (`self.var`, `self.e`,
      `self.z`, `self.v`, `self.v_inv`, `self.var_pos`) is missing after
      `__init__`, rather than failing silently deeper in `dll`/`ddll`
      with a wrong-gradient bug.
- [ ] Provide the existing EGARCH implementation as a built-in
      reference subclass (`paneltime.models.EGARCH` or similar) rather
      than only as copy-paste documentation — this both dogfoods the
      new ABC and gives users a real importable example.

### 1.5 Heteroskedasticity function: symbolic codegen
- [ ] Add a code path accepting a single `sympy` expression in terms of
      symbols `e` and `z` (e.g. `sp.log(e**2 + 1e-8)`) as
      `h_function=` in `.fit()` (or as a `set_h_function` default
      implementation for `LikelihoodModel` subclasses).
- [ ] Auto-derive `h_e_val`, `h_2e_val`, `h_z_val`, `h_2z_val`,
      `h_ez_val` via `sympy.diff()` instead of requiring hand-derived
      NumPy expressions.
- [ ] Auto-generate the ExprTk-compatible C++ string via
      `sympy.printing.ccode` (or a custom `sympy` printer matching
      ExprTk's dialect — note ExprTk uses `^` for power, and disallows
      `and`/`or`; a custom printer may be needed rather than stock
      `ccode`). Validate the generated string round-trips through the
      existing ExprTk parser before accepting it.
- [ ] Add a consistency self-check at fit time: evaluate `h_val` (numeric
      Python path) and the ExprTk-parsed `h_val_cpp` on a small sample
      of `e`/`z` values and assert they agree within tolerance; raise a
      clear error if they diverge. This closes the current silent-drift
      risk even for users who still hand-write both forms.
- [ ] Keep manual `h_val`/`h_val_cpp` assignment supported as a fallback
      for expressions `sympy` can't convert automatically, but run the
      same consistency self-check on it.

### 1.6 General/cross-cutting source hygiene
- [ ] Full snake_case pass over the public API (constructor args,
      `.fit()` kwargs, attribute names) — no remaining capitalized
      abbreviations like `HF`, `ID`, `T` as parameter names.
- [ ] Add type hints to all new/renamed public methods and dataclasses.
- [ ] Add or update the test suite to cover: the new `.fit()` kwarg
      surface, the `Results` attribute set, the `LikelihoodModel` ABC
      (including the numerical-fallback path), and the `sympy`
      h-function codegen + consistency check.
- [ ] Update `pyproject.toml`/`setup_script.py` dependency list to add
      `sympy` (for 1.5) and, if adopted, `formulaic` or `patsy` for
      formula parsing (only if the formula parser is also being
      replaced — confirm scope before adding; this was flagged as a
      possible follow-on but is not in this task list's critical path).
- [ ] Bump major version (semver) on release, since this is a breaking
      API change even with deprecation shims in place.

---

## Task List 2 — Documentation site (`qmd/` directory)

Site is built with Quarto. Current pages: `index.qmd` (About),
`attributes.qmd` (Output), `options.qmd` (Setting options),
`custom_model.qmd` (Custom Model), `hfunc.qmd` (C++ heteroskedasticity
function syntax guide). Do this list after Task List 1 has landed, so
every code sample below can be written against the real new API.

### 2.1 Site structure
- [ ] Add a `_quarto.yml` nav restructure with these top-level sections,
      in order: **Quickstart**, **User Guide**, **API Reference**,
      **Examples**, **Migrating from options/execute (v1 → v2)**,
      **Changelog**.
- [ ] Split the current single flat nav list into a sidebar with these
      groupings (see 2.3–2.7 for per-page content).
- [ ] Add a `changelog.qmd` sourced from the PyPI/GitHub release history
      (versions 1.2.0 → 1.2.70+); one bullet per version at minimum for
      versions after the API rewrite ships.

### 2.2 New page: `quickstart.qmd`
- [ ] One end-to-end example: install → construct model → `.fit()` →
      `.summary()`. Mirror the shape of `statsmodels`/`arch` quickstarts.
- [ ] Show the *new* two-step API (`model = pt.PanelARIMAGARCH(...)`,
      `results = model.fit(...)`), not the old `pt.execute(...)`.
- [ ] Include one sentence stating what makes paneltime distinct
      (panel + ARIMA + GARCH jointly), carried over from the current
      `index.qmd` framing, but trimmed — the "unlike any other tool"
      claim should link to the migration/comparison content the
      research phase surfaced, or be softened to "no other package we
      are aware of does all three jointly."

### 2.3 Rewrite `index.qmd` → `about.qmd`
- [ ] Keep the three-bullet definition (panel / non-stationary mean /
      non-stationary variance).
- [ ] Remove implementation-detail content that moves to Quickstart
      (the full `execute()` signature) — About should sell the *what*,
      Quickstart shows the *how*.
- [ ] Update all code samples to the new API from Task List 1.

### 2.4 Rewrite `attributes.qmd` → `api/results.qmd`
- [ ] Reorganize the attribute table into two clearly separated
      sections: **"Results — everyday use"** (`params`, `bse`,
      `tvalues`, `pvalues`, `conf_int()`, `nobs`, `df_resid`, `llf`,
      `aic`, `bic`, `converged`, `random_effects`, `summary()`) and
      **"Optimizer internals (advanced)"** (`hessian`,
      `gradient_matrix`, `gradient_vector`, `dx_norm`, `its`, `msg`,
      `t0`/`t1`, `log_likelihood_object`).
- [ ] Delete or clearly mark `output` and `table` as private
      (`_output`, `_table`) — the current page already says "for
      internal use," so don't document them as public attributes.
- [ ] Regenerate this page from source docstrings now that the
      `Results` class exists (Task List 1 §1.2) rather than
      hand-maintaining the table — use `quartodoc` or a Sphinx-to-Quarto
      docstring extraction step.
- [ ] Rename the page title from "Output" to "Results object" for
      discoverability/search matching with `statsmodels` users' mental
      model.

### 2.5 Rewrite `options.qmd` → `api/fit-options.qmd`
- [ ] Restructure the single flat table into subsections matching the
      new grouped kwargs from Task List 1 §1.3: **ARIMA/GARCH order**,
      **Effects**, **Optimizer**, **Covariance**, **Constraints**,
      **Custom model / custom h-function**.
- [ ] Replace every code sample of the form `pt.options.X = Y` with the
      new `.fit(...)` kwarg form.
- [ ] Add a short **"Migrating from `pt.options`"** callout box (or a
      dedicated `migration.qmd`, linked from nav) mapping every old
      `options.*` attribute name to its new home, e.g.:
      `pqdkm` → `order=`, `garch_order=`; `EGARCH` → `vol='EGARCH'`;
      `fixed_random_group_eff`/`fixed_random_time_eff` →
      `effects=pt.Effects(group=..., time=...)`; `custom_model` →
      `likelihood=` (instance, not class).
- [ ] Fix source typos while rewriting: "defalut"→"default",
      "porperties"→"properties", "Numer"→"Number",
      "signficant"→"significant", "og"→"of", "Se example"→"See
      example," "expnential"→"exponential."
- [ ] Normalize all attribute names to snake_case in the rewritten
      table (currently `ARMA_constraint`/`GARCH_min` mix casing with
      `accuracy`/`tolerance`).

### 2.6 Rewrite `custom_model.qmd` → `guide/custom-likelihood-models.qmd`
- [ ] Replace the "define a plain class, assign the class itself, do
      not instantiate it" pattern with the new `LikelihoodModel` ABC
      pattern (subclass, instantiate, pass instance to
      `model.fit(likelihood=...)`) from Task List 1 §1.4.
- [ ] Rewrite method names in all examples: `ll`→`loglike`,
      `dll`→`score`, `ddll`→`hessian` (or whatever final names Task
      List 1 settled on — keep this page in lockstep with the source
      docstrings).
- [ ] Add a short subsection "Skipping analytical derivatives" showing
      that `score`/`hessian` are optional given the numerical
      differentiation fallback (Task List 1 §1.4) — this is a new
      capability, not just a rename, and should be called out as
      lowering the barrier to entry.
- [ ] Keep the worked EGARCH example, but update it to use the new
      `sympy`-based `h_function=` path from §2.7 instead of hand-paired
      `h_val`/`h_val_cpp`. Keep the manual dual-implementation path
      documented as a fallback for advanced users, clearly marked as
      the harder/legacy route.

### 2.7 Rewrite `hfunc.qmd` → `guide/heteroskedasticity-functions.qmd`
- [ ] Lead with the new `sympy`-expression path (one symbolic
      expression, auto-derived Python + auto-generated C++) as the
      primary/recommended workflow.
- [ ] Demote the current ExprTk hand-written C++ content (syntax
      adaptations, temporary variables, `=`/`==` rewriting rules) to a
      collapsed "Advanced: hand-writing the C++ expression directly"
      section at the bottom, for users who hit `sympy.ccode` limitations.
- [ ] Keep the ExprTk function reference table (abs/sqrt/log/exp/trig/
      floor/ceil/min/max/sign) since it's still needed for the advanced
      path.
- [ ] Add a note on what happens if the `sympy` expression can't be
      converted to valid ExprTk automatically (error message, escape
      hatch).

### 2.8 Cross-cutting
- [ ] Every code sample site-wide must be executed/tested as part of
      the Quarto render (use Quarto's executable code cells rather than
      static fenced code blocks) so API drift breaks the docs build
      instead of silently going stale.
- [ ] Add a single `api/` landing page listing all public classes
      (`PanelARIMAGARCH`, `Results`, `LikelihoodModel`, `Effects`,
      `Options`/config objects) with one-line descriptions and links.
- [ ] Run a full site-wide link check after the restructure (page URLs
      are changing, e.g. `attributes.html` → `api/results.html`) and
      add redirects in `_quarto.yml` for the old URLs to avoid breaking
      existing inbound links/citations.
