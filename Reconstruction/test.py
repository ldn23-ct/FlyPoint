import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    MC_degree0_nodefect = np.load("./data/0_degree_no_defect_single.npy")
    MC_degree0_defect = np.load("./data/0_degree_defect_single.npy")
    SIM_degree0_nodefect = np.load("./data/detresponse0_simulated.npy")
    SIM_degree0_defect = np.load("./data/detresponse1_simulated.npy")
    
    plt.imshow(MC_degree0_nodefect, cmap='gray', aspect='auto')
    # MC_value = np.sum(MC_degree0_defect, axis=0)
    MC_value = np.sum(MC_degree0_nodefect, axis=0)
    SIM_value = np.sum(SIM_degree0_defect, axis=0)
    # SIM_value = np.sum(SIM_degree0_nodefect, axis=0)
    x = np.arange(SIM_value.shape[0])
    # plt.scatter(x, SIM_value)
    # plt.scatter(x, MC_value/np.max(MC_value))
    plt.show()
    
    # mu = 0.134*2.7*1e-3
    # x1 = np.linspace(0, 70, 701)
    # x2 = 1.414 * x1
    # x = x1 + x2
    # y = np.exp(-1 * mu * x)
    # plt.scatter(x, y)
    # plt.show()