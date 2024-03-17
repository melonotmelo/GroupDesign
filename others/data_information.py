type_channel_index = {
    1: [4],
    2: [4],
    3: [4],
    4: [0],
    5: [0, 3, 4],
    6: [0, 1, 3, 4],
    7: [0, 1, 3, 4],
    8: [0, 1, 3, 4],
    9: [3, 5],
    10: [6, 7],
    11: [8],
    12: [0, 1, 3, 9, 10],
    13: [0, 1, 2, 3, 4],
    14: [0, 1, 2, 3, 4],
    15: [0, 1, 2, 3, 4],
    16: [0, 1, 2, 3, 4],
}


def get_type_from_filename(filename):
    if "1D_Advection" in filename:
        return 1
    elif "1D_diff-sorp" in filename:
        return 2
    elif "ReacDiff" in filename:
        return 3
    elif "1D_Burgers" in filename:
        return 4
    elif "1D_CFD" in filename:
        return 5
    elif "2D_CFD" in filename:
        return 6
    elif "KH_M" in filename:
        return 7
    elif "2D_shock" in filename:
        return 8
    elif "2D_DarcyFlow" in filename:
        return 9
    elif "2D_diff-react" in filename:
        return 10
    elif "2D_rdb" in filename:
        return 11
    elif "ns_incom_inhom" in filename:
        return 12
    elif "3D_CFD_Turb" in filename:
        return 13
    elif "3D_CFD_Rand" in filename:
        return 14
    elif "Turb_M" in filename:
        return 15
    elif "BlastWave" in filename:
        return 16
    else:
        raise NotImplementedError


tp_name = {
    1: "1D_Advection",
    2: "1D_Diff_Sorp",
    3: "1D_Reac_Diff",
    4: "1D_burgers\'",
    5: "1D_CFD",
    6: "2D_CFD",
    7: "2D_KH_M",
    8: "2D_Shock",
    # 9: "2D_DarcyFlow",
    10: "2D_Reac_Diff",
    11: "2D_Rdb",
    12: "2D_NS_Incom_Inhom",
    13: "3D_CFD_Turb",
    14: "3D_CFD_Rand",
    15: "3D_Turb_M",
    16: "3D_BlastWave"
}


datafile_paths = [
    # "1D_Advection_Sols_beta0.1.hdf5",
    # "1D_Advection_Sols_beta0.2.hdf5",
    # "1D_Advection_Sols_beta0.4.hdf5",
    # "1D_Advection_Sols_beta1.0.hdf5",
    # "1D_Advection_Sols_beta2.0.hdf5",
    # "1D_Advection_Sols_beta4.0.hdf5",
    # "1D_Burgers_Sols_Nu0.001.hdf5",
    # "1D_Burgers_Sols_Nu0.002.hdf5",
    # "1D_Burgers_Sols_Nu0.004.hdf5",
    # "1D_Burgers_Sols_Nu0.01.hdf5",
    # "1D_Burgers_Sols_Nu0.02.hdf5",
    # "1D_Burgers_Sols_Nu0.04.hdf5",
    # "1D_Burgers_Sols_Nu0.1.hdf5",
    # "1D_Burgers_Sols_Nu0.2.hdf5",
    # "1D_Burgers_Sols_Nu0.4.hdf5",
    # "1D_Burgers_Sols_Nu1.0.hdf5",
    # "1D_Burgers_Sols_Nu2.0.hdf5",
    # "1D_Burgers_Sols_Nu4.0.hdf5",
    # "1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5",
    # "1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5",
    # "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
    # "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5",
    # "1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5",
    # "1D_diff-sorp_NA_NA.h5",
    # "ReacDiff_Nu0.5_Rho1.0.hdf5",
    # "ReacDiff_Nu0.5_Rho10.0.hdf5",
    # "ReacDiff_Nu0.5_Rho2.0.hdf5",
    # "ReacDiff_Nu0.5_Rho5.0.hdf5",
    # "ReacDiff_Nu1.0_Rho1.0.hdf5",
    # "ReacDiff_Nu1.0_Rho10.0.hdf5",
    # "ReacDiff_Nu1.0_Rho2.0.hdf5",
    # "ReacDiff_Nu1.0_Rho5.0.hdf5",
    # "ReacDiff_Nu2.0_Rho1.0.hdf5",
    # "ReacDiff_Nu2.0_Rho10.0.hdf5",
    # "ReacDiff_Nu2.0_Rho2.0.hdf5",
    # "ReacDiff_Nu2.0_Rho5.0.hdf5",
    # "ReacDiff_Nu5.0_Rho1.0.hdf5",
    # "ReacDiff_Nu5.0_Rho10.0.hdf5",
    # "ReacDiff_Nu5.0_Rho2.0.hdf5",
    # "ReacDiff_Nu5.0_Rho5.0.hdf5",

    "E:/data/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
    # "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5",
    # "2D_diff-react_NA_NA.h5",
    # "2D_rdb_NA_NA.h5",
]
