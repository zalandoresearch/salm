from matplotlib import pyplot as plt
import numpy as np
import sys


prefix = sys.argv[1]


def load_results(path):
    with open(path) as f:
        ll = [float(x) for x in f.read().split("\n")]
    return np.exp(-np.mean(ll))


def get_curves():
    baseline = []
    salm = []

    for x in ["_5000", "_10000", "_15000", "_20000", "_25000", "_30000", "_35000", "_40000", ""]:
        baseline.append(load_results("./models/{}{}.ll.txt".format(prefix, x)))
        salm.append(load_results("./models/{}_tag{}.ll.txt".format(prefix, x)))

    return np.array(baseline), np.array(salm)


baseline, salm = get_curves()


x_axis = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 42068]


titles = {
    'coco_char': 'CoCo - character',
    'penn_char': 'Penn - character',
    'coco': 'CoCo - word',
    'penn': 'Penn - word',
}


plt.figure()
plt.plot(x_axis, baseline)
plt.plot(x_axis, salm)
plt.legend(["baseline", "salm"])
plt.xlabel("N sentences")
plt.ylabel("per-word perplexity")
plt.title(titles[prefix])
plt.show()
