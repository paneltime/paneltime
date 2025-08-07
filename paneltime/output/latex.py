import numpy as np
from ..processing import model_parser
from ..processing import arguments

def format(summaries, fmt, heading, col_headings, variable_groups, digits, size):
	if not isinstance(summaries, (list, tuple, set)):
		summaries = [summaries]
	tbl = table(summaries, digits, variable_groups)
	initvar = remove_initvar(summaries, tbl)

	if fmt.lower() == 'latex':
		return format_latex(col_headings, heading, size, summaries, initvar, tbl)
	elif fmt.lower() == 'html':
		return format_html(col_headings, heading, size, summaries, initvar, tbl)
	else:
		raise ValueError("Unsupported format. Use 'latex' or 'html'.")


def format_latex(col_headings, heading, size, summaries, initvar, tbl):
	for k in list(tbl.keys()):
		k_new = k.replace('_', r'\_')
		tbl[k_new] = tbl.pop(k)

	s = LATEX_TABLE[0] % (heading, SIZES[size])
	s += "& "
	for h in col_headings:
		s += r"\textbf{" + h + r"} & "
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

	s += dgnst(summaries, initvar)
	s += LATEX_TABLE[1]
	return s

def format_html(col_headings, heading, size, summaries, initvar, tbl):
	style = SIZES_HTML[size]
	
	rows = [f'<table class="summary-table" style="{style}">']
	rows.append(f'<caption>{heading}</caption>')


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
	rows.append(dgnst_html(summaries, initvar))
	rows.append('<tr><td colspan="100%" style="border-top: 1px solid black;"></td></tr>')
	
	rows.append('</tbody></table>')
	return '\n'.join(rows)





def remove_initvar(summaries, tbl):
	# removes initvar
	if not arguments.INITVAR in summaries[0].panel.args.names_v:
		return
	initvar = summaries[0].panel.args.caption_d[arguments.INITVAR][0]
	if initvar in tbl:
		return [i[2] for i in tbl.pop(initvar)]


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

	for uqname in list(unique_names):
		record[uqname] = []
		for i, namesi in enumerate(names):
			record[uqname].append(['']*3)
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
					c(params[i][j], tsign[i][j], digits), #param value and significance code
					f'& ({np.round(se[i][j],digits)})', # standard error
					np.round(params[i][j], digits)
				)

def c(coef, sign, digits):
	s=''
	if sign<0.01:
		s = '***'
	elif sign<0.05:
		s = '**'
	elif sign<0.1:
		s = '*'
	return f"& {np.round(coef,digits)}" + "$^{***}$" 

def dgnst(summaries, initvar):
	s = r"\hline \\[-1.8ex]" + '\n'
	diagnostics = []
	for smr in summaries:
		diagnostics.append([
			np.round(smr.stats.diag.Rsq,2),
			np.round(smr.stats.diag.Rsqadj, 2),
			int(smr.stats.info.N),
			np.round(smr.stats.info.aic,1), 
			np.round(smr.stats.info.bic, 1),
			np.round(smr.stats.info.log_lik,1),
			np.round(smr.stats.diag.no_ac_prob,2), 

			])
	for r in range(len(diagnostics[0])):
		s+= LABELS_DIAGS[r]
		for c in range(len(diagnostics)):
			s += f"& {diagnostics[c][r]}"
		s += r' & \\' + '\n'
	if not initvar is None:
		s += 'Initial variance'
		for i in initvar:
			s += f"& {i}"
		s += r' & \\' + '\n'
	return s

def dgnst_html(summaries, initvar):
	s = ''
	diagnostics = []
	for smr in summaries:
		diagnostics.append([
			np.round(smr.stats.diag.Rsq, 2),
			np.round(smr.stats.diag.Rsqadj, 2),
			int(smr.stats.info.N),
			np.round(smr.stats.info.aic, 1),
			np.round(smr.stats.info.bic, 1),
			np.round(smr.stats.info.log_lik, 1),
			np.round(smr.stats.diag.no_ac_prob, 2),
		])
	for r in range(len(diagnostics[0])):
		s += '<tr><td>' + LABELS_DIAGS[r] + '</td>'
		for c in range(len(diagnostics)):
			s += f'<td>{diagnostics[c][r]}</td>'
		s += '</tr>'
	if initvar is not None:
		s += '<tr><td>Initial variance</td>'
		for i in initvar:
			s += f'<td>{i}</td>'
		s += '</tr>'
	return s


LATEX_TABLE = [
	r"\begin{table}[!htbp] \centering " +'\n'
	r"\vspace{10pt}" + '\n'
	r"\caption{%s} "+'\n'
	r"\label{table:regression} "+'\n'
	r"%s" +'\n'
	r"\begin{tabular}{@{\extracolsep{5pt}}lcccccc} "+'\n'
	r"\toprule"+'\n',
	
	r"\hline "+'\n'
	r"\hline \\[-1.8ex] "+'\n'
	r"\textit{Sign. codes:}  & \multicolumn{3}{r}{$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01} \\ "+'\n'
	r"\end{tabular} "+'\n'
	r"\label{table:legitimacy-oppression}"+'\n'
	r"\end{table} "+'\n'
	]

LABELS_DIAGS = [r"R$^{2}$", r"Adjusted R$^{2}$", r"Observations", 
	r"AIC", r"BIC", r"Log-Likelihood", r"Breusch-Pagan test p-value",
	]

SIZES = [r'\footnotesize', r'\small', r'\normalsize']
SIZES_HTML = ["font-size:8px;", "font-size:11px;", "font-size:14px;"]