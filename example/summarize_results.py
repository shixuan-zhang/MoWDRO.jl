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
has_cpos = 'CPOS_TIME' in cpos_cols
# Optional out-of-sample CVaR columns emitted by the CVaR-DRO regression
# example. Both must be present to enable the extra plot line and table
# column; the summarizer stays backward-compatible when they are absent.
has_cvar = 'TEST_CVAR' in data.columns and 'TEST_CVAR_STD' in data.columns

# Resolve low/high quantile columns: CVaR-regression CSVs use the new
# β-dependent TEST_QLOW/TEST_QHIGH; older CSVs (or non-CVaR experiments)
# still ship the hard-coded TEST_Q10/TEST_Q90 columns.
qlow_col  = 'TEST_QLOW'  if 'TEST_QLOW'  in data.columns else 'TEST_Q10'
qhigh_col = 'TEST_QHIGH' if 'TEST_QHIGH' in data.columns else 'TEST_Q90'
# β column is emitted alongside TEST_QLOW/QHIGH by the CVaR example; when
# present we can render the actual percentages in each plot's legend.
has_beta  = 'CVAR_LEVEL' in data.columns

mean_cols = ['WASS_RAD', 'TRAIN_TIME', 'TRAIN_OBJ',
             'TEST_MEAN', 'TEST_STD', qlow_col, 'TEST_MED', qhigh_col]
mean_cols += cpos_cols
if has_cvar:
    # TEST_CVAR is a per-row scalar (mean of the top-β tail) and
    # TEST_CVAR_STD is a per-row scalar (std of the same tail); we
    # want both averaged across replications when a group has >1 rep.
    mean_cols += ['TEST_CVAR', 'TEST_CVAR_STD']
if has_beta:
    mean_cols += ['CVAR_LEVEL']
std_cols = ['TRAIN_OBJ', 'TEST_MEAN', 'TEST_STD']
if has_cpos:
    std_cols = std_cols + ['CPOS_OBJ', 'CPOS_MEAN', 'CPOS_STD']

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
    # β-dependent quantile band label. When the CSV carries CVAR_LEVEL we
    # derive the exact percentages from β (β/2 and 1-β/2); otherwise fall
    # back to the legacy 10-90% label.
    if has_beta:
        beta      = float(dro_rows.iloc[0]['CVAR_LEVEL'])
        low_pct   = int(round(beta / 2 * 100))
        high_pct  = int(round((1 - beta / 2) * 100))
    else:
        low_pct, high_pct = 10, 90
    band_label = str(low_pct) + "-" + str(high_pct) + "\\%"
    body = ""

    body += "    \\addplot[name path=DRO10_" + str(k) + ",color=blue!50,densely dotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, qlow_col) + "    };\n"
    body += "    \\addplot[name path=DRO90_" + str(k) + ",color=blue!50,densely dotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, qhigh_col) + "    };\n"
    body += ("    \\addplot[pattern=north east lines,pattern color=blue!10,forget plot] "
             "fill between [of=DRO10_" + str(k) + " and DRO90_" + str(k) + "];\n")

    body += "    \\addplot[name path=ESO10_" + str(k) + ",color=red!50,dashdotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, qlow_col) + "    };\n"
    body += "    \\addplot[name path=ESO90_" + str(k) + ",color=red!50,dashdotted,thick,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, qhigh_col) + "    };\n"
    body += ("    \\addplot[pattern=north west lines,pattern color=red!10,forget plot] "
             "fill between [of=ESO10_" + str(k) + " and ESO90_" + str(k) + "];\n")

    body += "    \\addplot[color=blue,very thick,densely dotted]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{DRO test mean and " + band_label + " range};\n"

    body += "    \\addplot[color=red,very thick,dashdotted]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{ESO test mean and " + band_label + " range};\n"

    # test CVaR curves — same colour/base pattern as the matching test-mean
    # line so DRO/ESO groupings stay visually consistent; marker distinguishes
    # CVaR from mean.
    if has_cvar:
        body += "    \\addplot[color=blue,very thick,densely dotted,mark=triangle*,mark size=2pt]\n"
        body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_CVAR') + "    };\n"
        body += "    \\addlegendentry{DRO test CVaR};\n"

        body += "    \\addplot[color=red,very thick,dashdotted,mark=square*,mark size=2pt]\n"
        body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_CVAR') + "    };\n"
        body += "    \\addlegendentry{ESO test CVaR};\n"

    body += "    \\addplot[color=blue,very thick,solid,mark=x]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{DRO training obj.\\ value};\n"

    body += "    \\addplot[color=red,very thick,densely dashed,mark=+]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{ESO training obj.\\ value};\n"

    plot_path = str_output_dir + str_file_name + "_" + str(k) + "_plot.tex"
    with open(plot_path, "w") as f:
        f.write(plot_preamble + axis_open(title_str) + body + plot_footer)

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


if has_cpos:
    # extra column when TEST_CVAR is available
    _cvar_header = " & Test CVaR" if has_cvar else ""
    _col_spec    = "rrrrrrrrrrr" if has_cvar else "rrrrrrrrrr"
    _mo_span     = 5             if has_cvar else 4
    _mo_range    = "3-7"         if has_cvar else "3-6"
    _hk_range    = "8-11"        if has_cvar else "7-10"
    table = (
        r"\documentclass{standalone}" "\n"
        r"\usepackage{booktabs,mathpazo}" "\n"
        r"\begin{document}" "\n"
        r"\begin{tabular}{" + _col_spec + "}\n"
        r"\toprule" "\n"
        " & & \\multicolumn{" + str(_mo_span) + r"}{c}{Moment-WDRO} "
        "& \\multicolumn{4}{c}{Hanasusanto-Kuhn} \\\\\n"
        r"\cmidrule(lr){" + _mo_range + "} "
        r"\cmidrule(lr){" + _hk_range + "}\n"
        "$N$ & $r$ & Time (s) & Training Obj. & Test Mean & Test Std."
        + _cvar_header +
        "\n     & Time (s) & Training Obj. & Test Mean & Test Std. \\\\\n"
        r"\midrule" "\n"
    )
else:
    _cvar_header = " & Test CVaR" if has_cvar else ""
    _col_spec    = "rrrrrrr" if has_cvar else "rrrrrr"
    table = (
        r"\documentclass{standalone}" "\n"
        r"\usepackage{booktabs,mathpazo}" "\n"
        r"\begin{document}" "\n"
        r"\begin{tabular}{" + _col_spec + "}\n"
        r"\toprule" "\n"
        "$N$ & $r$ & Time (s) & Training Obj. & Test Mean & Test Std."
        + _cvar_header + " \\\\\n"
        r"\midrule" "\n"
    )


def obj_cell(row, mean_col, std_col, digits):
    if single_replication:
        return "$" + fmt(row[mean_col], digits) + "$"
    return fmt_pm(row[mean_col], row[std_col], digits)


obj_digits = 2 if has_cpos else 3

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
        obj_cell(row, 'TEST_STD', 'TEST_STD_SAMPLE_STD', obj_digits),
    ]
    # Test-CVaR ± tail-std (both averaged across replications). Uses
    # `fmt_pm` for the plus-minus rendering; both quantities are always
    # present when `has_cvar` is True.
    if has_cvar:
        cells.append(fmt_pm(row['TEST_CVAR'], row['TEST_CVAR_STD'], obj_digits))
    if has_cpos:
        cells += [
            fmt(row['CPOS_TIME'], 1),
            obj_cell(row, 'CPOS_OBJ', 'CPOS_OBJ_SAMPLE_STD', obj_digits),
            obj_cell(row, 'CPOS_MEAN', 'CPOS_MEAN_SAMPLE_STD', obj_digits),
            obj_cell(row, 'CPOS_STD', 'CPOS_STD_SAMPLE_STD', obj_digits),
        ]
    table += " & ".join(cells) + r" \\" + "\n"

table += r"""\bottomrule
\end{tabular}
\end{document}"""

with open(str_output_dir + str_file_name + "_table.tex", "w") as f:
    f.write(table)
