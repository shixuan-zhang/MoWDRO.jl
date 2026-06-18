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

plot_header = r"""\documentclass{standalone}
\usepackage{pgfplots,mathpazo}
\usetikzlibrary{pgfplots.fillbetween}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=16cm,
    height=8cm,
    xlabel={Training sample size $n$},
    ylabel={Obj.\ Value},
    enlargelimits=0.05,
    legend pos=south east,
    ymajorgrids=true,
    grid style=dashed,
]
"""

plot_footer = r"""\end{axis}
\end{tikzpicture}
\end{document}"""

for w in dro_indices:
    k = w - 1
    dro_rows = rows_for(w)
    body = ""

    body += "    \\addplot[name path=DRO10_" + str(k) + ",color=blue!20,densely dotted,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_Q10') + "    };\n"
    body += "    \\addplot[name path=DRO90_" + str(k) + ",color=blue!20,densely dotted,forget plot]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_Q90') + "    };\n"
    body += ("    \\addplot[blue!10,forget plot] fill between [of=DRO10_"
             + str(k) + " and DRO90_" + str(k) + "];\n")

    body += "    \\addplot[name path=ESO10_" + str(k) + ",color=red!20,dashdotted,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_Q10') + "    };\n"
    body += "    \\addplot[name path=ESO90_" + str(k) + ",color=red!20,dashdotted,forget plot]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_Q90') + "    };\n"
    body += ("    \\addplot[red!10,forget plot] fill between [of=ESO10_"
             + str(k) + " and ESO90_" + str(k) + "];\n")

    body += "    \\addplot[color=blue,very thick,densely dotted]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{DRO test mean ($k=" + str(k) + "$)};\n"

    body += "    \\addplot[color=red,very thick,dashdotted]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TEST_MEAN') + "    };\n"
    body += "    \\addlegendentry{ESO test mean};\n"

    body += "    \\addplot[color=blue,solid,mark=x]\n"
    body += "    coordinates {\n" + coords_block(dro_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{DRO in-sample obj.\\ ($k=" + str(k) + "$)};\n"

    body += "    \\addplot[color=red,densely dashed,mark=+]\n"
    body += "    coordinates {\n" + coords_block(eso_rows, 'TRAIN_OBJ') + "    };\n"
    body += "    \\addlegendentry{ESO in-sample obj.};\n"

    plot_path = str_output_dir + str_file_name + "_" + str(k) + "_plot.tex"
    with open(plot_path, "w") as f:
        f.write(plot_header + body + plot_footer)


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
\begin{tabular}{rrrrrrr}
\toprule
$n$ & $w$ & $r$ & Train Time (s) & Train Obj. & Test Mean & Test Std. \\
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
        + str(int(row['WASS_IDX'])) + " & "
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
