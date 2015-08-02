# circe environment setup.

module load compilers/intel/14.0.1 apps/cuda/6.5.14
export PATH=$PATH:/shares/rogers/psi4/bin
export PSIDATADIR=/shares/rogers/psi4/interfaces/share/psi
export OMP_NUM_THREADS=12

# Calling code:
# /shares/rogers/psi4public/src/bin/fnocc/df_ccsd.cc
# Reference residual code:
# /shares/rogers/psi4public/src/bin/fnocc/df_cc_residual.cc
# Run cmd:
# psi4 test/input.dat -n 2

