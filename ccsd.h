#ifndef PLUGIN_CCSD_H
#define PLUGIN_CCSD_H

#include<pthread.h>
#include <sys/types.h>
#include<psi4-dec.h>
#include<libiwl/iwl.h>
#include<libpsio/psio.hpp>
#include<libpsio/psio.h>
#include<libmints/wavefunction.h>
#include<psifiles.h>
#include<../bin/fnocc/ccsd.h>
#include<sys/times.h>
#include"gpuhelper.h"

// cuda libraries
#include<cuda.h>
#include<cublas.h>
#include<cuda_runtime.h>

typedef struct {
        int id;
} parm;

namespace psi{ namespace fnocc{

// GPU DFCC class
class GPUDFCoupledCluster : public DFCoupledCluster{

  public:
    GPUDFCoupledCluster(boost::shared_ptr<psi::Wavefunction>wfn,Options&options);
    ~GPUDFCoupledCluster();

    virtual bool same_a_b_orbs() const { return true; }
    virtual bool same_a_b_dens() const { return true; }
    double compute_energy();
    void common_init();
    void pthreadCCResidual(int id);
  protected:

    /// t1-transformed 3-index integrals
    void T1Integrals();
    /// t1-transformed 3-index integrals (from SCF) for Fock build
    void T1Fock();

    /// cc diagrams:
    virtual void CCResidual();
    //virtual void saveCCResidual();

    /// extra storage for gpu (ac|bd) function: 1/2o(o+1)v(v+1)
    double * tempr, ** tempr2;
    void useVabcd1();
    void workinguseVabcd1();

    /// triples
    PsiReturnType triples();

    /// (ac|bd) diagram - this will be the first target to gpuify
    void brokenVabcd1();
    void tiledVabcd1();
    void slowVabcd1();
    void saveVabcd1();
    void workingVabcd1();
    void cpuVabcd1();
    void nearlythereVabcd1();
    virtual void Vabcd1();
    void FinishVabcd1();

    void UpdateT2();

    /// initialize cuda
    void CudaInit();

    /// initialize cuda
    void CudaFinalize();

    /// define tiling based on available gpu memory
    void DefineTiling();

    /// allocate memory for gpu
    void AllocateGPUMemory();

    /// allocate memory for cpu
    virtual void AllocateMemory();

    pthread_t base_thread;

    /// gpu buffers
    double ** gpubuffer;
    //double * gput2;
    //double * gpuv;
    //double * gpur2;
    //double * gputemp;
    //double * gput1;
    //double * gpur1;


    /// GPU-specific variables ... some are obsolete
    int num_gpus;
    double left, wasted;
    long int ovtilesize, novtiles, lastovtile, lastov2tile, ov2tilesize, nov2tiles;
    long int tilesize, ntiles, lasttile;
    long int ncputhreads,ngputhreads, nblocks, num;
    bool gpudone;
    bool cpudone;
    long int last_a;


    boost::shared_ptr<GPUHelper> helper_;
    //__global__ void GPUKernel_Iqdb(int a,int v,int nQ,double * in,double * out);

};

    /// GPU kernels
    //__global__ void GPUKernel_Iqdb(int a,int v,int nQ,double * in,double * out);
    //__global__ void GPUKernel_Vm(int a,int v,double * in,double * out);
    //__device__ int  GPUKernel_Position(int i,int j);

}}


#endif
