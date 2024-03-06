from matplotlib import pyplot as plt
import numpy as np

cm_to_inch = 1 / 2.54
cm_to_point = 28.3465
plt.rcParams["figure.dpi"] = 600
plt.rcParams["lines.linewidth"] = 0.005 * cm_to_point
plt.rcParams["figure.subplot.left"] = 0.125
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.125
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.03 * cm_to_point
# エラーバーの設定
error_config = {
    "lw": 0.02 * cm_to_point,
    "capsize": 0.05 * cm_to_point,
    "capthick": 0.02 * cm_to_point,
}  # 太さとキャップの太さを設定
lw = 0.02 * cm_to_point

green = tuple(np.array([0, 176, 80]) / 255)
magenta = tuple(np.array([208, 0, 149]) / 255)
orange = tuple(np.array([237, 125, 49]) / 255)
violet = tuple(np.array([112, 48, 160]) / 255)
yellow = tuple(np.array([255, 192, 0]) / 255)
dark_blue = tuple(np.array([47, 85, 151]) / 255)
medium_blue = tuple(np.array([143, 170, 220]) / 255)
# light_blue = tuple(np.array([218, 227, 243]) / 255)
light_blue = tuple(np.array([193, 205, 237]) / 255)
