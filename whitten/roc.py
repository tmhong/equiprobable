import numpy as np
import matplotlib.pyplot as plt

def main():
    osde_roc = np.load("osde_saved/roc.npz")
    nn_roc = np.load("osde_saved/nn_roc.npz")

    plt.plot(osde_roc['eps_hh'], osde_roc['eps_tt'], label="OSDE", color="blue")
    plt.plot(nn_roc['eps_hh'], nn_roc['eps_tt'], label="NN", color="orange")
    plt.title("ROC Curve")
    plt.xlabel(r"$\epsilon_{HH}$")
    plt.ylabel(r"$\epsilon_{tt}$")
    plt.legend()
    plt.savefig("plots/ROC/roc.png")


if __name__ == "__main__":
    main()