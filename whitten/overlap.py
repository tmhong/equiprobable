import numpy as np
import uproot
import matplotlib.pyplot as plt


def make_plot(var, top_data, hh_data, quark):
    plt.hist(top_data[quark + "_" + var], bins=50, density=True, alpha=0.5, label=f"Top Quark {var}")
    plt.hist(hh_data[quark + "_" + var], bins=50, density=True, alpha=0.5, label=f"HH {var}")
    plt.xlabel(var)
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"plots/compare/{quark}_{var}_histogram.png")
    plt.close()


def main():
    with uproot.open("rootFiles/reco_tt.root") as f:
        tree = f["ttbar"]
        top_data = tree.arrays(library="np")
    with uproot.open("rootFiles/reco_hh.root") as f:
        tree = f["ttbar"]
        hh_data = tree.arrays(library="np")


    for branch in top_data.keys():
        top_data[branch] = top_data[branch][:len(hh_data[branch])]

    make_plot("phi", top_data, hh_data, "t")
    make_plot("theta", top_data, hh_data, "t")
    make_plot("phistar", top_data, hh_data, "t")
    make_plot("thetaprime", top_data, hh_data, "t")
    make_plot("phi", top_data, hh_data, "t~")
    make_plot("theta", top_data, hh_data, "t~")
    make_plot("phistar", top_data, hh_data, "t~")
    make_plot("thetastar", top_data, hh_data, "t~")

    plt.hist(top_data["rapidity"], bins=50, density=True, alpha=0.5, label=f"TTbar rapidity")
    plt.hist(hh_data["rapidity"], bins=50, density=True, alpha=0.5, label=f"HH rapidity")
    plt.xlabel("rapidity")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"plots/compare/rapidity_histogram.png")
    plt.close()

    plt.hist(top_data["cosprodangle"], bins=50, density=True, alpha=0.5, label=f"TTbar prodangle")
    plt.hist(hh_data["cosprodangle"], bins=50, density=True, alpha=0.5, label=f"HH prodangle")
    plt.xlabel("cos(prodangle)")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"plots/compare/prodangle_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()