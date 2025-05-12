#!/usr/bin/env python3

# use system module to obtain the input file and output directory
import sys, os
# import modules for parsing and processing
import pandas as pd
import numpy as np

# get the command line arguments
str_input_file = sys.argv[1]
str_output_dir = './'
if len(sys.argv) > 2:
    str_output_dir = sys.argv[2]
str_file_name, str_ext_name = os.path.splitext(os.path.basename(str_input_file))
if str_ext_name != '.csv':
    raise ValueError('unsupported input file format:', str_ext_name)

# parse the data from the csv file
data = pd.DataFrame(pd.read_csv(str_input_file))

# plot the in-sample training objective values 
min_rad_increment = data['WASS_RAD'][1] - data['WASS_RAD'][0]
scale_axis_factor = -round(np.log10(min_rad_increment))
data_train = data.loc[data['WASS_RAD'] <= [min_rad_increment*i for i in range(len(data['WASS_RAD']))]]
output_train = r"""\documentclass{standalone}
\usepackage{pgfplots,mathpazo}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=16cm,
    height=8cm,
    xlabel={Wasserstein radius $r$},
    ylabel={Mean Obj.\ Value},
    enlargelimits=0.05,
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,""" + "\n"
if min_rad_increment < 1:
    output_train += "    scaled x ticks=base 10:" + str(scale_axis_factor) + ",\n"
output_train += r"""]
\addplot[color=blue,mark=x]
    coordinates {""" + "\n"
for i in range(len(data_train['WASS_RAD'])):
    output_train += " "*8 + "(" + str(data_train['WASS_RAD'][i]) + ","
    output_train += str(round(data_train['TRAIN_OBJ'][i],2)) + ")\n"
output_train += " "*4 + "};\n"
output_train += r"""    \addlegendentry{in-sample};
    \addplot[color=red,mark=+]
    coordinates {""" + "\n"
for i in range(len(data_train['WASS_RAD'])):
    output_train += " "*8 + "(" + str(data_train['WASS_RAD'][i]) + ","
    output_train += str(round(data_train['TEST_MEAN'][i],2)) + ")\n"
output_train += " "*4 + "};\n"
output_train += r"""    \addlegendentry{out-of-sample};
\end{axis}
\end{tikzpicture}
\end{document}"""
with open(str_output_dir+str_file_name+"_train.tex", "w") as file_output:
    file_output.write(output_train)



# plot the out-of-sample performance (mean and quantiles)
min_rad = round(data['WASS_RAD'].values[1],2)
max_rad = round(data['WASS_RAD'].values[-1],2)
output_test = r"""\documentclass{standalone}
\usepackage{pgfplots,mathpazo}
\usetikzlibrary{pgfplots.fillbetween}
\begin{document}
\begin{tikzpicture}
\begin{semilogxaxis}[
    width=16cm,
    height=8cm,
    xlabel={Wasserstein radius $r$ (log-scale)},
    ylabel={Obj.\ Value},
    legend pos=north west,
    ymajorgrids=true,
    grid style=loosely dotted,
]
    \addplot[color=blue,thick]
    coordinates {""" + "\n"
for i in range(len(data['WASS_RAD'])):
    output_test += " "*8 + "(" + str(data['WASS_RAD'][i]) + ","
    output_test += str(round(data['TEST_MEAN'][i],2)) + ")\n"
output_test += " "*4 + "};\n"
output_test += r"""    \addlegendentry{DRO mean};
    \addplot[name path=DRO10,color=blue!20,dashed]
    coordinates {""" + "\n"
for i in range(len(data['WASS_RAD'])):
    output_test += " "*8 + "(" + str(data['WASS_RAD'][i]) + ","
    output_test += str(round(data['TEST_Q90'][i],2)) + ")\n"
output_test += " "*4 + "};\n"
output_test += r"""    \addlegendentry{DRO 10-90\%};
    \addplot[name path=DRO90,color=blue!20,dashed,forget plot]
    coordinates {""" + "\n"
for i in range(len(data['WASS_RAD'])):
    output_test += " "*8 + "(" + str(data['WASS_RAD'][i]) + ","
    output_test += str(round(data['TEST_Q10'][i],2)) + ")\n"
output_test += " "*4 + "};\n"
output_test += r"""   \addplot[color=blue!20,dashed,forget plot]
    coordinates {""" + "\n"
for i in range(len(data['WASS_RAD'])):
    output_test += " "*8 + "(" + str(data['WASS_RAD'][i]) + ","
    output_test += str(round(data['TEST_MED'][i],2)) + ")\n"
output_test += " "*4 + "};\n"
output_test += r"""    \addplot[blue!10,forget plot] fill between [of=DRO10 and DRO90];
    \addplot[color=red,thick]
    coordinates {
        (""" + str(min_rad) + "," + str(round(data['TEST_MEAN'][0],2)) + ")\n"
output_test += " "*8 + "(" + str(max_rad) + "," + str(round(data['TEST_MEAN'][0],2)) + ")\n"
output_test += r"""    };
    \addlegendentry{ESO mean};
    \addplot[name path=ESO10,color=red!20,dashed]
    coordinates {
        (""" + str(min_rad) + "," + str(round(data['TEST_Q90'][0],2)) + ")\n"
output_test += " "*8 + "(" + str(max_rad) + "," + str(round(data['TEST_Q90'][0],2)) + ")\n"
output_test += r"""    };
    \addlegendentry{ESO 10-90\%};
    \addplot[name path=ESO90,color=red!20,dashed,forget plot]
    coordinates {
        (""" + str(min_rad) + "," + str(round(data['TEST_Q10'][0],2)) + ")\n"
output_test += " "*8 + "(" + str(max_rad) + "," + str(round(data['TEST_Q10'][0],2)) + ")\n"
output_test += r"""    };
    \addplot[color=red!20,dashed,forget plot]
    coordinates {
        (""" + str(min_rad) + "," + str(round(data['TEST_MED'][0],2)) + ")\n"
output_test += " "*8 + "(" + str(max_rad) + "," + str(round(data['TEST_MED'][0],2)) + ")\n"
output_test += r"""    };
\end{semilogxaxis}
\end{tikzpicture}
\end{document}"""
with open(str_output_dir+str_file_name+"_test.tex", "w") as file_output:
    file_output.write(output_test)
