#!/usr/bin/env python3

import sys, os
import pandas as pd

str_input_file = sys.argv[1]
str_output_dir = './'
if len(sys.argv) > 2:
    str_output_dir = sys.argv[2]
if str_output_dir and not str_output_dir.endswith(os.sep):
    str_output_dir += os.sep
str_file_name, str_ext_name = os.path.splitext(os.path.basename(str_input_file))
if str_ext_name != '.csv':
    raise ValueError('unsupported input file format:', str_ext_name)

data = pd.read_csv(str_input_file)

cpos_cols = [c for c in data.columns if c.startswith('CPOS_')]
has_cpos  = 'CPOS_TIME' in cpos_cols
ncvx_cols = [c for c in data.columns if c.startswith('NCVX_')]
has_ncvx  = 'NCVX_TIME' in ncvx_cols
# Quantile-regression runs emit a `QUANTILE_LEVEL` column filled with τ on
# every row. When present, the summarizer relabels each per-radius plot to
# reflect the pinball-loss context; the underlying data schema is otherwise
# identical to the mean-regression case.
has_quantile = ('QUANTILE_LEVEL' in data.columns
                and data['QUANTILE_LEVEL'].notna().any()
                and float(data['QUANTILE_LEVEL'].dropna().iloc[0]) > 0)

mean_cols = ['WASS_RAD', 'TRAIN_TIME', 'TRAIN_OBJ',
             'TEST_MEAN', 'TEST_STD', 'TEST_Q10', 'TEST_MED', 'TEST_Q90']
mean_cols += cpos_cols
mean_cols += ncvx_cols
if has_quantile:
    mean_cols += ['QUANTILE_LEVEL']
std_cols = ['TRAIN_OBJ', 'TEST_MEAN', 'TEST_STD']
if has_cpos:
    std_cols = std_cols + ['CPOS_OBJ', 'CPOS_MEAN', 'CPOS_STD']
if has_ncvx:
    std_cols = std_cols + ['NCVX_OBJ', 'NCVX_MEAN', 'NCVX_STD']

grouped = data.groupby(['TRAIN_SIZE', 'WASS_IDX'])
single_replication = (grouped.size() <= 1).all()
agg = grouped[mean_cols].mean()
for c in std_cols:
    agg[c + '_SAMPLE_STD'] = grouped[c].std(ddof=1)
agg = agg.reset_index().sort_values(['TRAIN_SIZE', 'WASS_IDX'])

wass_indices = sorted(agg['WASS_IDX'].unique())
dro_indices = [w for w in wass_indices if w > 1]


def rows_for(wass_idx):
    return agg[agg['WASS_IDX'] == wass_idx].sort_values('TRAIN_SIZE')


def coords_block(rows, ycol, digits=3):
    lines = []
    for _, r in rows.iterrows():
        lines.append(
            " " * 8 + "(" + str(int(r['TRAIN_SIZE'])) + ","
            + str(round(float(r[ycol]), digits)) + ")"
        )
    return "\n".join(lines) + "\n"


eso_rows = rows_for(1)

plot_preamble = r"""\documentclass{standalone}
\usepackage{pgfplots,mathpazo}
\usetikzlibrary{pgfplots.fillbetween,patterns}
\begin{document}
\begin{tikzpicture}
"""


def axis_open(title_str, ylabel="Obj.\\ Value", legend_pos="north east"):
    return (
        "\\begin{axis}[\n"
        "    width=12cm,\n"
        "    height=8cm,\n"
        "    title={" + title_str + "},\n"
        "    xlabel={Training sample size $N$},\n"
        "    ylabel={" + ylabel + "},\n"
        "    enlarge x limits=0.02,\n"
        "    enlarge y limits=0.05,\n"
        "    legend pos=" + legend_pos + ",\n"
        "    legend style={fill=white,fill opacity=0.6,draw opacity=1,text opacity=1},\n"
        "    ymajorgrids=true,\n"
        "    grid style=dashed,\n"
        "    scaled y ticks=false,\n"
        "    yticklabel style={/pgf/number format/fixed},\n"
        "]\n"
    )


plot_footer = r"""\end{axis}
\end{tikzpicture}
\end{document}"""

for w in dro_indices:
    k = w - 1
    dro_rows = rows_for(w)
    r0 = float(dro_rows.iloc[0]['WASS_RAD'])
    r0_str = "{:.3f}".format(r0).rstrip('0').rstrip('.')
    title_str = "Initial radius $r_0 = " + r0_str + "$"
    # In a quantile-regression run every row carries the same τ; tag the
    # plot title with (pinball loss, $\tau=X$) and swap the y-axis label
    # to the pinball loss symbol. Otherwise fall back to the mean-regression
    # defaults.
    if has_quantile:
        tau_val = float(dro_rows.iloc[0]['QUANTILE_LEVEL'])
        tau_str = ("{:.3f}".format(tau_val)).rstrip('0').rstrip('.')
        title_str = title_str + r" (pinball loss, $\tau=" + tau_str + "$)"
        axis_ylabel = r"Pinball loss $\rho_\tau$"
    else:
        axis_ylabel = "Obj.\\ Value"
    body = ""

    body += "    \\addplot[name path=DRO10_" + str(k) + ",color=blue!50,densely dotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_Q10') + "    };\n"
    body += "    \\addplot[name path=DRO90_" + str(k) + ",color=blue!50,densely dotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_Q90') + "    };\n"
    body += ("    \\addplot[pattern=north east lines,pattern color=blue!10,forget plot] "
             "fill between [of=DRO10_" + str(k) + " and DRO90_" + str(k) + "];\n")

    body += "    \\addplot[name path=ESO10_" + str(k) + ",color=red!50,dashdotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_Q10') + "    };\n"
    body += "    \\addplot[name path=ESO90_" + str(k) + ",color=red!50,dashdotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_Q90') + "    };\n"
    body += ("    \\addplot[pattern=north west lines,pattern color=red!10,forget plot] "
             "fill between [of=ESO10_" + str(k) + " and ESO90_" + str(k) + "];\n")

    body += "    \\addplot[color=blue,very thick,densely dotted]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{DRO test mean and 10-90\\% range};\n"

    body += "    \\addplot[color=red,very thick,dashdotted]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{ESO test mean and 10-90\\% range};\n"

    body += "    \\addplot[color=blue,very thick,solid,mark=x]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{DRO training obj.\\ value};\n"

    body += "    \\addplot[color=red,very thick,densely dashed,mark=+]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{ESO training obj.\\ value};\n"

    plot_path = str_output_dir + str_file_name + "_" + str(k) + "_plot.tex"
    with open(plot_path, "w") as f:
        f.write(plot_preamble + axis_open(title_str, ylabel=axis_ylabel) + body + plot_footer)

if has_cpos:
    cpos_colors = ['red', 'blue', 'green!60!black',
                   'orange!80!black', 'violet', 'brown']
    title_str = "Computational time comparison"
    body = ""
    for i, w in enumerate(wass_indices):
        rows = rows_for(w)
        if rows['CPOS_TIME'].isna().all():
            continue
        r0 = float(rows.iloc[0]['WASS_RAD'])
        r0_str = "{:.3f}".format(r0).rstrip('0').rstrip('.')
        label = "$(r_0 = " + r0_str + ")$"
        c = cpos_colors[i % len(cpos_colors)]

        body += "    \\addplot[color=" + c + ",very thick,solid,mark=x]\n"
        body += "    coordinates {\n" + coords_block(rows, 'TRAIN_TIME', digits=1) + "    };\n"
        body += "    \\addlegendentry{Mo " + label + "};\n"

        body += "    \\addplot[color=" + c + ",very thick,densely dashed,mark=+]\n"
        body += "    coordinates {\n" + coords_block(rows, 'CPOS_TIME', digits=1) + "    };\n"
        body += "    \\addlegendentry{HK " + label + "};\n"

    cpos_plot_path = str_output_dir + str_file_name + "_time_comparison.tex"
    with open(cpos_plot_path, "w") as f:
        f.write(plot_preamble
                + axis_open(title_str, ylabel="Time (s)", legend_pos="north west")
                + body + plot_footer)


def fmt(v, digits):
    if pd.isna(v):
        return "--"
    return ("{:." + str(digits) + "f}").format(float(v))


def fmt_pm(mean, std, digits):
    if pd.isna(std):
        return "$" + fmt(mean, digits) + "$"
    return "$" + fmt(mean, digits) + r" \pm " + fmt(std, digits) + "$"


# Active baseline groups (display order: CPOS first, then NCVX). Each entry
# contributes a 4-column block (Time, Training Obj., Test Mean, Test Std.)
# to the right of the Moment-WDRO block. `prefix` names the CSV column
# family; `label` is the multicolumn header shown in the LaTeX table.
baseline_groups = []
if has_cpos:
    baseline_groups.append({'prefix': 'CPOS', 'label': 'Hanasusanto-Kuhn'})
if has_ncvx:
    baseline_groups.append({'prefix': 'NCVX', 'label': 'Nonconvex baseline'})

n_base_cols    = 6                                    # N, r, Time, Training Obj, Test Mean, Test Std
n_per_baseline = 4                                    # Time, Training Obj, Test Mean, Test Std
n_total_cols   = n_base_cols + n_per_baseline * len(baseline_groups)
col_spec       = 'r' * n_total_cols

if baseline_groups:
    # top row: two blank cells (for N and r), then Moment-WDRO multicolumn
    # plus one multicolumn per active baseline.
    top = ' & & ' + ' & '.join(
        [r'\multicolumn{4}{c}{Moment-WDRO}'] +
        [r'\multicolumn{4}{c}{' + g['label'] + r'}' for g in baseline_groups]
    ) + r' \\'
    # cmidrules: 3-6 for Moment-WDRO, then 7-10, 11-14, ... for each baseline.
    starts    = [3 + 4 * i for i in range(1 + len(baseline_groups))]
    cmidrules = ' '.join(
        r'\cmidrule(lr){' + str(s) + '-' + str(s + 3) + '}' for s in starts)
    # sub-header: N & r & (Time & Training Obj. & Test Mean & Test Std.)
    # broken across one line per (Moment-WDRO + baseline) group for readability
    # in the emitted .tex source.
    baseline_line = 'Time (s) & Training Obj. & Test Mean & Test Std.'
    sub_lines     = ['$N$ & $r$ & ' + baseline_line]
    for _ in baseline_groups:
        sub_lines.append('     & ' + baseline_line)
    sub = '\n'.join(sub_lines) + r' \\'
    header_block = top + '\n' + cmidrules + '\n' + sub + '\n'
else:
    header_block = r'$N$ & $r$ & Time (s) & Training Obj. & Test Mean & Test Std. \\' + '\n'

table = (
    r'\documentclass{standalone}' + '\n' +
    r'\usepackage{booktabs,mathpazo}' + '\n' +
    r'\begin{document}' + '\n' +
    r'\begin{tabular}{' + col_spec + '}' + '\n' +
    r'\toprule' + '\n' +
    header_block +
    r'\midrule' + '\n'
)


def obj_cell(row, mean_col, std_col, digits):
    if single_replication:
        return "$" + fmt(row[mean_col], digits) + "$"
    return fmt_pm(row[mean_col], row[std_col], digits)


obj_digits = 2 if (has_cpos or has_ncvx) else 3

prev_train_size = None
for _, row in agg.iterrows():
    ts = int(row['TRAIN_SIZE'])
    if prev_train_size is not None and ts != prev_train_size:
        table += "\\midrule\n"
    prev_train_size = ts
    cells = [
        str(ts),
        fmt(row['WASS_RAD'], 3),
        fmt(row['TRAIN_TIME'], 1),
        obj_cell(row, 'TRAIN_OBJ', 'TRAIN_OBJ_SAMPLE_STD', obj_digits),
        obj_cell(row, 'TEST_MEAN', 'TEST_MEAN_SAMPLE_STD', obj_digits),
        obj_cell(row, 'TEST_STD',  'TEST_STD_SAMPLE_STD',  obj_digits),
    ]
    for g in baseline_groups:
        p = g['prefix']
        cells += [
            fmt(row[p + '_TIME'], 1),
            obj_cell(row, p + '_OBJ',  p + '_OBJ_SAMPLE_STD',  obj_digits),
            obj_cell(row, p + '_MEAN', p + '_MEAN_SAMPLE_STD', obj_digits),
            obj_cell(row, p + '_STD',  p + '_STD_SAMPLE_STD',  obj_digits),
        ]
    table += " & ".join(cells) + r" \\" + "\n"

table += r"""\bottomrule
\end{tabular}
\end{document}"""

with open(str_output_dir + str_file_name + "_table.tex", "w") as f:
    f.write(table)
