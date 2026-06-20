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

mean_cols = ['WASS_RAD', 'TRAIN_TIME', 'TRAIN_OBJ',
             'TEST_MEAN', 'TEST_STD', 'TEST_Q10', 'TEST_MED', 'TEST_Q90']
std_cols = ['TRAIN_OBJ', 'TEST_MEAN', 'TEST_STD']

grouped = data.groupby(['TRAIN_SIZE', 'WASS_IDX'])
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


def axis_open(title_str):
    return (
        "\\begin{axis}[\n"
        "    width=12cm,\n"
        "    height=8cm,\n"
        "    title={" + title_str + "},\n"
        "    xlabel={Training sample size $N$},\n"
        "    ylabel={Obj.\\ Value},\n"
        "    enlarge x limits=0.02,\n"
        "    enlarge y limits=0.05,\n"
        "    legend pos=north east,\n"
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
        f.write(plot_preamble + axis_open(title_str) + body + plot_footer)


def fmt(v, digits):
    if pd.isna(v):
        return "--"
    return ("{:." + str(digits) + "f}").format(float(v))


def fmt_pm(mean, std, digits):
    if pd.isna(std):
        return "$" + fmt(mean, digits) + "$"
    return "$" + fmt(mean, digits) + r" \pm " + fmt(std, digits) + "$"


table = r"""\documentclass{standalone}
\usepackage{booktabs,mathpazo}
\begin{document}
\begin{tabular}{rrrrrr}
\toprule
$N$ & $r$ & Time (s) & Training Obj. & Test Mean & Test Std. \\
\midrule
"""

prev_train_size = None
for _, row in agg.iterrows():
    ts = int(row['TRAIN_SIZE'])
    if prev_train_size is not None and ts != prev_train_size:
        table += "\\midrule\n"
    prev_train_size = ts
    table += (
        str(ts) + " & "
        + fmt(row['WASS_RAD'], 3) + " & "
        + fmt(row['TRAIN_TIME'], 1) + " & "
        + fmt_pm(row['TRAIN_OBJ'], row['TRAIN_OBJ_SAMPLE_STD'], 3) + " & "
        + fmt_pm(row['TEST_MEAN'], row['TEST_MEAN_SAMPLE_STD'], 3) + " & "
        + fmt_pm(row['TEST_STD'], row['TEST_STD_SAMPLE_STD'], 3) + r" \\" + "\n"
    )

table += r"""\bottomrule
\end{tabular}
\end{document}"""

with open(str_output_dir + str_file_name + "_table.tex", "w") as f:
    f.write(table)
