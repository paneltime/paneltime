import numpy as np
from ..processing import model_parser
from ..processing import arguments
import paneltime as pt
from . import stat_dist

def format(summaries, fmt, heading, col_headings, variable_groups, digits, size, caption, fpath=None):
	if not isinstance(summaries, (list, tuple, set)):
		summaries = [summaries]
	tbl = table(summaries, digits, variable_groups)

	if fmt.lower() == 'latex':
		s = format_latex(col_headings, heading, size, summaries, tbl, digits, caption)

	elif fmt.lower()[:4] == 'html':
		s = format_html(col_headings, heading, size, summaries, tbl, digits, 
					 fmt.lower() == 'html_page')
	else:
		raise ValueError("Unsupported format. Use 'latex' or 'html'.")
	
	if not fpath is None:
		with open(fpath, "w", encoding="utf-8") as f:
			f.write(s)

	return s



def format_latex(col_headings, heading, size, summaries, tbl, digits, caption):
	if len(summaries) == 0:
		raise RuntimeError("No summaries to format.")
	for k in list(tbl.keys()):
		k_new = k.replace('_', r'\_')
		tbl[k_new] = tbl.pop(k)

	s = LATEX_TABLE[0] % (heading, caption, SIZES[size], (len(summaries)+1)*'c')
	s += "& "
	for h in col_headings:
		s += r"\textbf{" + h.replace('_', r'\_') + r"} & "
	s += r' & \\ ' + '\n'
	s += r"\midrule" + '\n'

	for varname in tbl:
		s += varname
		for i in range(len(summaries)):
			s += tbl[varname][i][0]  # parameter value and significance
		s += r' & \\' + '\n'
		for i in range(len(summaries)):
			s += tbl[varname][i][1]  # standard error
		s += r' & \\' + '\n'

	s += dgnst_latex(summaries, digits)
	s += LATEX_TABLE[1] %(summaries[0].panel.sign_codes_tex, )
	return s

def format_html(col_headings, heading, size, summaries, tbl, digits, page):
	style = SIZES_HTML[size]
	
	rows = [f'<table class="summary-table" style="{style}">']
	rows.append(f'<caption>{heading} (robust s.e)</caption>')


	# Header row with top and bottom borders on each cell
	header_row = (
		'<tr>'
		'<th style="border-top: 1px solid black; border-bottom: 2px solid black;"></th>' +
		''.join(
			f'<th style="border-top: 1px solid black; border-bottom: 2px solid black;"><strong>{h}</strong></th>'
			for h in col_headings
		) +
		'</tr>'
	)

	# Now wrap both inside thead
	rows.append(f'<thead>{header_row}</thead>')


	# Main body (coefficients and standard errors)
	for varname in tbl:
		coef_row = [f'<tr><td>{varname}</td>']
		coef_row += [
			f'<td>{tbl[varname][i][0].replace("&", "").replace("$^{***}$", "<sup>***</sup>").replace("$^{**}$", "<sup>**</sup>").replace("$^{*}$", "<sup>*</sup>")}</td>'
			for i in range(len(summaries))
		]
		coef_row.append('</tr>')

		se_row = ['<tr><td></td>']
		se_row += [f'<td>{tbl[varname][i][1].replace("&", "")}</td>' for i in range(len(summaries))]
		se_row.append('</tr>')

		rows.extend(coef_row + se_row)

	# Diagnostics
	rows.append('<tr><td colspan="100%" style="border-top: 1px solid black;"></td></tr>')
	rows.append(dgnst_html(summaries, digits))
	rows.append('<tr><td colspan="100%" style="border-top: 1px solid black;"></td></tr>')
	
	rows.append('</tbody></table>')

	html = '\n'.join(rows)

	if page:
		html = html_wrap(html)

	return html





def get_unique_varnames(summaries, variable_groups):
	names_reg = [i.panel.args.names_d['beta'] for i in summaries]
	names_reg = set([name for sublist in names_reg for name in sublist])
	names_all = [i.panel.args.caption_v for i in summaries]
	names_all = set([name for sublist in names_all for name in sublist])
	names_internal = set(names_all) - set(names_reg)
	groups = set()
	group_members = set()
	if len(variable_groups):
		group_members = set([item for sublist in variable_groups.values() for item in sublist])
		groups = set(variable_groups.keys())
	
	names_reg = (names_reg|groups) - group_members

	names_reg = sorted(names_reg)
	names_internal = sorted(names_internal)
	names_reg = put_intercept_first(names_reg)

	return names_reg, names_internal

def put_intercept_first(names_reg):
	"""If the intercept is in the list of names, put it first."""
	intercept_name = model_parser.DEFAULT_INTERCEPT_NAME
	if intercept_name in names_reg:
		intercept = names_reg.pop(names_reg.index(intercept_name))
		names_reg = [intercept] + names_reg
	
	return names_reg


def table(summaries, digits, variable_groups):

	(unique_names_reg, unique_names_internal
  		) = get_unique_varnames(summaries, variable_groups)

	record = {}

	add_to_record(record, unique_names_reg, summaries, variable_groups, digits)
	add_to_record(record, unique_names_internal, summaries, variable_groups, digits)


	return record

def add_to_record(record, unique_names, summaries, variable_groups, digits):
	
	params = [i.results.params for i in summaries]
	tsign = [i.results.tsign for i in summaries]
	se = [i.results.se for i in summaries]
	names = [i.panel.args.caption_v for i in summaries]

	has_garch = np.any(list(sum(s.panel.pqdkm[3:]) > 0 for s in summaries))

	for i, uqname in enumerate(unique_names):
		if uqname == arguments.INITVAR_LONG:
			continue
		record[uqname] = len(names)*[['& ']*2]
		for i, namesi in enumerate(names):
			j = -1
			if uqname in namesi:
				j = namesi.index(uqname)
			elif uqname in variable_groups:
				try:
					j = namesi.index(variable_groups[uqname][i])
				except Exception as e:
					print(f"Error finding {uqname} in namesi: {namesi}."
		   				  f"variable_groups needs to be a dictionary of lists, whith each list "
						  f"item representing the associated variable name for each column/summary object."
		   				  f"Error: {e}")
			if j >= 0:
				record[uqname][i] = (
					c(params[i][j], tsign[i][j], digits, summaries), #param value and significance code
					f'& ({np.round(se[i][j],digits)})' # standard error
				)
	if not arguments.VARIANCE_CONSTANT in record:
		return
	if (not has_garch):
		record['Variance'] = record.pop(arguments.VARIANCE_CONSTANT)
	else:
		record[arguments.VARIANCE_CONSTANT] = record.pop(arguments.VARIANCE_CONSTANT)


def c(coef, sign, digits, summaries):
	codes = summaries[0].panel.sign_codes
	s=''
	for i in codes[::-1]:
		if sign < i[1]:
			if i[0]=="'":
				return f"& {np.round(coef,digits)}{i[0]}" 
			return f"& {np.round(coef,digits)}$^{{{i[0]}}}$"
	return f"& {np.round(coef,digits)}"


def dgnst(summaries, digits):
	
	diagnostics = []

	for smr in summaries:
		sts = smr.stats
		ci , ni = sts.diag.ci, sts.diag.n_ci
		ci = f"{int(ci)} ({ni})"
		diagnostics.append([
			int(sts.info.df),
			np.round(sts.diag.Rsqadj, 2),
			np.round(sts.info.aic,1), 
			np.round(sts.info.bic, 1),
			np.round(sts.info.log_lik,1),
			np.round(sts.diag.DW,2),
			ci,
			np.round(sts.diag.no_ac_prob,2), 
			])


	# additional context dependent diagnostics:

	initvars = [s.stats.info.initvar  for s in summaries]

	extra_diag = []
	#Initvar
	if not np.all(list(i is None for i in initvars)):
		append_dgnst(initvars, arguments.INITVAR_LONG, diagnostics, digits, extra_diag)


	# Hausmann test
	if len(summaries)==2: 
		result = hausmann_test(summaries)
		if result:
			append_dgnst(result, "Hausmann p-val", diagnostics, digits, extra_diag)
				  

	return diagnostics, extra_diag

def append_dgnst(values, name, diagnostics, digits, extra_diag):
	extra_diag.append(name)
	for i in range(len(diagnostics)):
		if values[i] is None:
			diagnostics[i].append('NA')
		elif type(values[i]) == str:
			diagnostics[i].append(values[i])
		else:
			diagnostics[i].append(np.round(values[i], digits) )



def hausmann_test(summaries):
	opt = [s.panel.options for s in summaries]
	if not ((opt[0].fixed_random_group_eff == 1 and
		opt[1].fixed_random_group_eff == 2) or
		(opt[0].fixed_random_time_eff == 1 and
			opt[0].fixed_random_time_eff == 2)):
		return False
	
	k=len(summaries[0].results.args['beta'])
	diff = summaries[0].results.params - summaries[1].results.params
	cov_diff = (summaries[0].results.cov_robust - summaries[1].results.cov_robust)

	diff = diff[:k]
	cov_diff = cov_diff[:k,:k]
	try:
		stat = diff.T @ np.linalg.inv(cov_diff) @ diff
		p_value = 1 - stat_dist.chisq(stat, len(diff))
		p_value = f"{p_value:.3f}"
	except np.linalg.LinAlgError:
		p_value = "NA"

	return ['', p_value]
	


	
def dgnst_latex(summaries, digits):

	s = r"\hline \\[-1.8ex]" + '\n'

	diagnostics, extra_diag = dgnst(summaries, digits)
	lables = get_dgnst_labels(summaries, extra_diag)
	for r in range(len(diagnostics[0])):
		s+= lables[r]
		for c in range(len(diagnostics)):
			s += f"& {diagnostics[c][r]}"
		s += r' & \\' + '\n'

	return s

def dgnst_html(summaries, digits):

	s = ''

	diagnostics, extra_diag = dgnst(summaries, digits)
	lables = get_dgnst_labels(summaries, extra_diag)
	for r in range(len(diagnostics[0])):
		s += '<tr><td>' + lables[r].replace("$^{2}$", "<sup>2</sup>") + '</td>'
		for c in range(len(diagnostics)):
			s += f'<td>{diagnostics[c][r]}</td>'
		s += '</tr>'

	return s

def html_wrap(html_table):
	html1 = """<!DOCTYPE html>
	<html lang="en">
	<head>
		<meta charset="UTF-8">
		<title>Regression Results</title>
		<style>
			body {
				font-family: Times;
				margin: 2em;
			}

			table {
				border-collapse: collapse;
				width: auto;
				margin-bottom: 1em;
			}

			.summary-table {
				font-family: Times;
				font-size: 14px;
				margin: 1em 0;
			}

			.summary-table caption {
				font-weight: bold;
				margin-bottom: 10px;
			}

			.summary-table th,
			.summary-table td {
				padding: 4px 8px;
				text-align: left;
				border: none; /* ✂️ No borders at all */
			}

			.summary-table th {
				font-weight: bold;
				background-color: transparent; /* ✂️ No shading */
			}
		</style>
	</head>
	"""
	html2 = f"\n<body>\n\t{html_table}\n</body>\n</html>"

	return html1 + html2

def get_dgnst_labels(summaries, extra_diag):
	
	if np.any(list(((po.fixed_random_time_eff==2) or 
				(po.fixed_random_group_eff==2)) 
				for po in [s.panel.options for s in summaries])):
		df = r"Effective d.f."
	else:
		df = r"D.f."
	
	return [df, r"Adjusted R$^{2}$", 
			r"AIC", r"BIC", r"Log-Likelihood", r"DW", r"Cond.index(k)", 
			r"Breusch-Pagan test p-value"
			] + extra_diag



LATEX_TABLE = [
	r"\begin{table}[!htbp] \centering " +'\n'
	r"\vspace{10pt}" + '\n'
	r"\caption{%s (robust clustsered s.e)} "+'\n'
	r"\label{table:%s} "+'\n'
	r"%s" +'\n'
	r"\begin{tabular}{@{\extracolsep{5pt}}l%s} "+'\n'
	r"\toprule"+'\n',
	
	r"\hline "+'\n'
	r"\hline \\[-1.8ex] "+'\n'
	r"\multicolumn{3}{r}{Sign. codes: %s} \\ "+'\n'
	r"\end{tabular} "+'\n'
	r"\end{table} "+'\n'
	]



SIZES = [r'\footnotesize', r'\small', r'\normalsize']
SIZES_HTML = ["font-size:8px;", "font-size:11px;", "font-size:14px;"]