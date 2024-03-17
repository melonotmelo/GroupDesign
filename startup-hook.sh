#bash
free -m
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install apex -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U model_hub==0.26.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple

export PATH=$PATH:$HOME/minio-binaries/

$HOME/minio-binaries/mc alias set shahe http://10.212.253.24:9000 wangzm wzmwzmbuaa

#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta0.2.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta0.4.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5 .
#
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.002.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.004.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.02.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.04.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.1.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.2.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu0.4.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Burgers/Train/1D_Burgers_Sols_Nu4.0.hdf5 .
#
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/CFD/Train/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/CFD/Train/1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/CFD/Train/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/CFD/Train/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/CFD/Train/1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5 .
#
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/Diff_Sorp/Train/1D_diff-sorp_NA_NA.h5 .
#
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu0.5_Rho1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu0.5_Rho10.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu0.5_Rho2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu0.5_Rho5.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho10.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho5.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu2.0_Rho1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu2.0_Rho10.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu2.0_Rho2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu2.0_Rho5.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu5.0_Rho1.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu5.0_Rho10.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu5.0_Rho2.0.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/1D/ReactionDiffusion/Train/ReacDiff_Nu5.0_Rho5.0.hdf5 .


#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/2D/2DCFD/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/2D/2DCFD/2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 .
#$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/2D/SWE/2D_rdb_NA_NA.h5 .
$HOME/minio-binaries/mc cp shahe/datasets/PDEbench/2D/2D_ReacDiff/2D_diff-react_NA_NA.h5 .