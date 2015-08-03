/*
 *@BEGIN LICENSE
 *
 * GPU-accelerated density-fitted coupled-cluster, a plugin to:
 *
 * PSI4: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#include"ccsd.h"
#include<psiconfig.h>
#include<../bin/fnocc/blas.h>
#include<libmints/matrix.h>
#include<libmints/molecule.h>
#include<libmints/mints.h>
#include<libciomr/libciomr.h>
#include<libqt/qt.h>

extern "C" {
  #include <parser.h>
}
#include <stdio.h>
#include <string.h>

#define PSIF_CIM 273 // TODO: move to psifiles.h

#ifdef _OPENMP
    #include<omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_set_num_threads(a)
    #define omp_get_num_threads()
    #define omp_set_dynamic(a)
    #define omp_set_nested(a)
#endif
#ifdef HAVE_MKL
    #include<mkl.h>
#else
    #define mkl_set_dynamic(a)
    #define mkl_set_num_threads(a)
    #define mkl_domain_set_num_threads(a,b)
#endif

#define NUMTHREADS 32
#define MAXBLOCKS 65535

using namespace psi;

namespace psi{namespace fnocc{

GPUDFCoupledCluster::GPUDFCoupledCluster(boost::shared_ptr<Wavefunction> reference_wavefunction, Options &options):
        DFCoupledCluster(reference_wavefunction,options)
{
    reference_wavefunction_ = reference_wavefunction;
    common_init();
}

void GPUDFCoupledCluster::common_init() {
}


GPUDFCoupledCluster::~GPUDFCoupledCluster()
{
}

void set_rand(double *x, int len) {
    int two = 2;
    int seed[4] = {4, 13, 28, 47};

    dlarnv_(&two, seed, &len, x);
}

void GPUDFCoupledCluster::CCResidual(){
    bool timer = options_.get_bool("CC_TIMINGS");
    int o = ndoccact;
    int v = nvirt;
    int vvoo[] = {v,v,o,o};
    int vo[] = {v,o};
    int oo[] = {o,o};
    int ov[] = {o,v};
    int vv[] = {v,v};
    int doo[] = {nQ,o,o};
    int dov[] = {nQ,o,v};
    int dvv[] = {nQ,v,v};
    uint32_t chks[8];
    long int nthreads = omp_get_max_threads();
    struct Environ e = {
        .debuglevel = 0,
    };
    char name_w1[] = "w1";
    char name_R[] = "R";
    void *names[] = {&name_w1, &name_R};
    FILE *eqns = fopen("df-ccsd.tex", "r");
    SMap *defs = slack_parse_inp(&e, eqns);
    Tensor *R1, *R2;
    fclose(eqns);

    set_rand(t1, o*v);
    printf("sizes = %d %d %d\n", nQ, o, v);
    boost::shared_ptr<PSIO> psio (new PSIO());

    if(defs == NULL) {
        printf("bad parse.\n");
    }// else if(e.debuglevel) {
    //    n = smap_iter(defs, show_assign, NULL);
    //    printf("Total assignments = %d\n", n);
    //}

    // Add inputs
    if (t2_on_disk){
        psio->open(PSIF_DCC_T2,PSIO_OPEN_OLD);
        psio->read_entry(PSIF_DCC_T2,"t2",(char*)&tempv[0],o*o*v*v*sizeof(double));
        psio->close(PSIF_DCC_T2,1);
        tb = tempv;
    }
    const char *arrs[] = {"t", "Fij", "Fia", "Fai", "Fab", "Uoo", "Uov", "Uvv"};
    Ast *tens[] = { mkLit(4, vvoo, tb),
		    mkLit(2,  oo,  Fij),
		    mkLit(2,  ov,  Fia),
		    mkLit(2,  vo,  Fai),
		    mkLit(2,  vv,  Fab),
		    mkLit(3, doo,  Qoo),
		    mkLit(3, dov,  Qov),
		    mkLit(3, dvv,  Qvv)
		  };
    /*smap_put(defs,   "t", mkLit(4, vvoo, tb));  // v, v, o, o
    smap_put(defs, "Fij", mkLit(2,  oo,  Fij)); // o, o
    smap_put(defs, "Fia", mkLit(2,  ov,  Fia)); // o, v
    smap_put(defs, "Fai", mkLit(2,  vo,  Fai)); // v, o
    smap_put(defs, "Fab", mkLit(2,  vv,  Fab)); // v, v
    smap_put(defs, "Uoo", mkLit(3, doo,  Qoo)); // d, o, o
    smap_put(defs, "Uov", mkLit(3, dov,  Qov)); // d, o, v
    smap_put(defs, "Uvv", mkLit(3, dvv,  Qvv)); // d, v, v*/
    for(int i=0; i<8; i++) {
	Tensor *t = tens[i]->base->t;
	smap_put(defs, arrs[i], tens[i]);
	chks[i] = crc32_bitwise(t->x, t->len*sizeof(double));
    }
    //memset(w1, 0, v*o*sizeof(double)); // already done

    MemSpace *mem = memspace_ctor(32, 1<<30); // 1 Gb
    if(mem == NULL) {
        printf("Error constructing memspace.\n");
        return;
    }
    if(run_quark_n(2, names, nthreads, mem, defs)) {
        printf("Error executing dag.\n");
        return;
    }
    R1 = (Tensor *)names[0]; R2 = (Tensor *)names[1];
    /*Ast *a = (Ast *)smap_get(defs, "R");
    if(a == NULL) {
	printf("R not found.\n");
	return;
    }
    R2 = exec_ast(a, nthreads, mem, defs);*/
    if(R1 == NULL || R2 == NULL) {
	printf("Error executing dag.\n");
	return;
    }

    printf("Used mem = %lu\n", mem->used);

    // singles residual
    C_DAXPY(v*o,1.0,R1->x,1,w1,1);
    //memcpy(w1, R1->x, o*v*sizeof(double));
    // doubles residual 
    psio->open(PSIF_DCC_R2,PSIO_OPEN_NEW);
    psio->write_entry(PSIF_DCC_R2,"residual",(char *)R2->x,o*o*v*v*sizeof(double));
    psio->close(PSIF_DCC_R2,1);

    tensor_dtor(&R1, mem);
    tensor_dtor(&R2, mem);

    //AB1(); // check other impl.
    for(int i=0; i<8; i++) {
	Tensor *t = tens[i]->base->t;
	if(chks[i] != crc32_bitwise(t->x, t->len*sizeof(double))) {
	    printf("Tensor %s changed!\n", arrs[i]);
	}
    }

    memspace_dtor(&mem);
    smap_dtor(&defs);

    return;
}

void GPUDFCoupledCluster::AB1(){
    bool timer = options_.get_bool("CC_TIMINGS");
    long int o = ndoccact;
    long int v = nvirt;
    long int nthreads = omp_get_max_threads();

    // A1 (G):  U(c,d,k,l) (ad|kc)
    #pragma omp parallel for schedule (dynamic) num_threads(nthreads)
    for (int d = 0; d < v; d++) {
	for (int i = 0; i < o; i++) {
	    for (int k = 0; k < o; k++) {
		for (int c = 0; c < v; c++) {
		    tempt[d*o*o*v+i*o*v+k*v+c] = (2.0*tb[c*o*o*v+d*o*o+k*o+i] - tb[c*o*o*v+d*o*o+i*o+k]);
		}
	    }
	}
    }
    F_DGEMM('t','n',o*v,nQ,o*v,1.0,tempt,o*v,Qov,o*v,0.0,tempv,o*v);
    #pragma omp parallel for schedule (dynamic) num_threads(nthreads)
    for (int q = 0; q < nQ; q++) {
	for (int a = 0; a < v; a++) {
	    for (int b = 0; b < v; b++) {
		integrals[q*v*v+b*v+a] = Qvv[q*v*v+a*v+b];
	    }
	}
    }
    F_DGEMM('n','t',o,v,v*nQ,1.0,tempv,o,integrals,v,1.0,w1,o);

    /*if (timer) {
	outfile->Printf("        A1 =      U(c,d,k,l) (ad|kc)                                    %6.2lf\n",omp_get_wtime()-start);
	start = omp_get_wtime();
    }*/

    // B1 (H): -U(a,c,k,l) (ki|lc)
    F_DGEMM('n','t',o*v,o*o,nQ,1.0,Qov,o*v,Qoo,o*o,0.0,integrals,o*v);
    #pragma omp parallel for schedule (dynamic) num_threads(nthreads)
    for (int i = 0; i < o; i++) {
	for (int c = 0; c < v; c++) {
	    for (int k = 0; k < o; k++) {
		for (int l = 0; l < o; l++) {
		    tempv[i*o*o*v+c*o*o+k*o+l] = integrals[k*o*o*v+i*o*v+l*v+c];
		}
	    }
	}
    }
    C_DCOPY(o*o*v*v,tb,1,tempt,1);
    #pragma omp parallel for schedule (dynamic) num_threads(nthreads)
    for (int a = 0; a < v; a++) {
	for (int c = 0; c < v; c++) {
	    for (int k = 0; k < o; k++) {
		C_DAXPY(o,-0.5,tb+a*o*o*v+c*o*o+k,o,tempt+a*o*o*v+c*o*o+k*o,1);
	    }
	}
    }
    F_DGEMM('t','n',o,v,o*o*v,-2.0,tempv,o*o*v,tempt,o*o*v,1.0,w1,o);

    /*if (timer) {
	outfile->Printf("        B1 =    - U(a,c,k,l) (ki|lc)                                    %6.2lf\n",omp_get_wtime()-start);
	start = omp_get_wtime();
    }*/
}

// t1-transformed 3-index fock matrix (using 3-index integrals from SCF)
void GPUDFCoupledCluster::T1Fock(){
    long int o = ndoccact;
    long int v = nvirt;
    long int full = o+v+nfzc+nfzv;

    // Ca_L = C(1-t1^T)
    // Ca_R = C(1+t1)
    double * Catemp = (double*)malloc(nso*full*sizeof(double));
    if ( reference_wavefunction_->isCIM() ) {
        boost::shared_ptr<PSIO> psio (new PSIO());
        psio->open(PSIF_CIM,PSIO_OPEN_OLD);
        psio->read_entry(PSIF_CIM,"C matrix",(char*)&Catemp[0],nso*full*sizeof(double));
        psio->close(PSIF_CIM,1);
        C_DCOPY(nso*full,&Catemp[0],1,Ca_L,1);
        C_DCOPY(nso*full,&Catemp[0],1,Ca_R,1);
    }else {
        C_DCOPY(nso*full,&Ca[0][0],1,Ca_L,1);
        C_DCOPY(nso*full,&Ca[0][0],1,Ca_R,1);
        C_DCOPY(nso*full,&Ca[0][0],1,Catemp,1);
    }

    #pragma omp parallel for schedule (static)
    for (int mu = 0; mu < nso; mu++) {
        for (int a = 0; a < v; a++) {
            double dum = 0.0;
            for (int i = 0; i < o; i++) {
                dum += Catemp[mu*full+i+nfzc] * t1[a*o+i];
            }
            Ca_L[mu*full + a + ndocc] -= dum;
        }
    }
    #pragma omp parallel for schedule (static)
    for (int mu = 0; mu < nso; mu++) {
        for (int i = 0; i < o; i++) {
            double dum = 0.0;
            for (int a = 0; a < v; a++) {
                dum += Catemp[mu*full+a+ndocc] * t1[a*o+i];
            }
            Ca_R[mu*full + i + nfzc] += dum;
        }
    }
    free(Catemp);

    // (Q|rs)
    boost::shared_ptr<PSIO> psio(new PSIO());
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);
    psio_address addr1  = PSIO_ZERO;
    psio_address addr2  = PSIO_ZERO;
    psio_address addroo = PSIO_ZERO;
    psio_address addrov = PSIO_ZERO;
    psio_address addrvo = PSIO_ZERO;
    psio_address addrvv = PSIO_ZERO;

    long int nrows = 1;
    long int rowsize = nQ_scf;
    while ( rowsize*nso*nso > o*o*v*v ) {
        nrows++;
        rowsize = nQ_scf / nrows;
        if (nrows * rowsize < nQ_scf) rowsize++;
        if (rowsize == 1) break;
    }
    long int lastrowsize = nQ_scf - (nrows - 1L) * rowsize;
    long int * rowdims = new long int [nrows];
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;
    for (int row = 0; row < nrows; row++) {
        psio->read(PSIF_DCC_QSO,"Qso SCF",(char*)&integrals[0],rowdims[row]*nso*nso*sizeof(double),addr1,&addr1);
        F_DGEMM('n','n',full,nso*rowdims[row],nso,1.0,Ca_L,full,integrals,nso,0.0,tempv,full);
        for (int q = 0; q < rowdims[row]; q++) {
            for (int mu = 0; mu < nso; mu++) {
                C_DCOPY(full,tempv+q*nso*full+mu*full,1,integrals+q*nso*full+mu,nso);
            }
        }
        F_DGEMM('n','n',full,full*rowdims[row],nso,1.0,Ca_R,full,integrals,nso,0.0,tempv,full);
        // full Qmo
        psio->write(PSIF_DCC_QSO,"Qmo SCF",(char*)&tempv[0],rowdims[row]*full*full*sizeof(double),addr2,&addr2);
    }
    delete rowdims;

    // build Fock matrix

    memset((void*)Fij,'\0',o*o*sizeof(double));
    memset((void*)Fia,'\0',o*v*sizeof(double));
    memset((void*)Fai,'\0',o*v*sizeof(double));
    memset((void*)Fab,'\0',v*v*sizeof(double));

    // transform H
    double ** hp = H->pointer();
    double * h = (double*)malloc(nmo*nmo*sizeof(double));
    for (int mu = 0; mu < nso; mu++) {
        for (int p = 0; p < nmo; p++) {
            double dum = 0.0;
            for (int nu = 0; nu < nso; nu++) {
                dum += Ca_L[nu*full + p + nfzc] * hp[nu][mu];
            }
            integrals[p*nso+mu] = dum;
        }
    }
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            double dum = 0.0;
            for (int nu = 0; nu < nso; nu++) {
                dum += Ca_R[nu*full+q+nfzc] * integrals[p*nso+nu];
            }
            h[p*nmo+q] = dum;
        }
    }

    double * temp3 = (double*)malloc(full*full*sizeof(double));

    memset((void*)temp3,'\0',full*full*sizeof(double));
    psio_address addr = PSIO_ZERO;

    nrows = 1;
    rowsize = nQ_scf;
    while ( rowsize*full*full > o*o*v*v ) {
        nrows++;
        rowsize = nQ_scf / nrows;
        if (nrows * rowsize < nQ_scf) rowsize++;
        if (rowsize == 1) break;
    }
    lastrowsize = nQ_scf - (nrows - 1L) * rowsize;
    rowdims = new long int [nrows];
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;
    for (int row = 0; row < nrows; row++) {
        psio->read(PSIF_DCC_QSO,"Qmo SCF",(char*)&integrals[0],rowdims[row]*full*full*sizeof(double),addr,&addr);
        for (int q = 0; q < rowdims[row]; q++) {
            // sum k (q|rk) (q|ks)
            F_DGEMM('n','n',full,full,ndocc,-1.0,integrals+q*full*full,full,integrals+q*full*full,full,1.0,temp3,full);

            // sum k (q|kk) (q|rs)
            double dum = 0.0;
            for (int k = 0; k < ndocc; k++) {
                dum += integrals[q*full*full+k*full + k];
            }
            F_DAXPY(full*full,2.0 * dum,integrals+q*full*full,1,temp3,1);
        }
    }
    delete rowdims;
    psio->close(PSIF_DCC_QSO,1);

    // Fij
    for (int i = 0; i < o; i++) {
        for (int j = 0; j < o; j++) {
            Fij[i*o+j] = h[i*nmo+j] + temp3[(i+nfzc)*full+(j+nfzc)];
        }
    }

    // Fia
    for (int i = 0; i < o; i++) {
        for (int a = 0; a < v; a++) {
            Fia[i*v+a] = h[i*nmo+a+o] + temp3[(i+nfzc)*full+(a+ndocc)];
        }
    }

    // Fai
    for (int a = 0; a < v; a++) {
        for (int i = 0; i < o; i++) {
            Fai[a*o+i] = h[(a+o)*nmo+i] + temp3[(a+ndocc)*full+(i+nfzc)];
        }
    }

    // Fab
    for (int a = 0; a < v; a++) {
        for (int b = 0; b < v; b++) {
            Fab[a*v+b] = h[(a+o)*nmo+b+o] + temp3[(a+ndocc)*full+(b+ndocc)];
        }
    }

    // replace eps
    for (int i = 0; i < o; i++) {
        eps[i] = Fij[i*o+i];
    }
    for (int a = 0; a < v; a++) {
        eps[a+o] = Fab[a*v+a];
    }

    free(h);
    free(temp3);
}

// t1-transformed 3-index integrals
void GPUDFCoupledCluster::T1Integrals(){
    long int o = ndoccact;
    long int v = nvirt;
    long int full = o+v+nfzc+nfzv;

    // Ca_L = C(1-t1^T)
    // Ca_R = C(1+t1)
    double * Catemp = (double*)malloc(nso*full*sizeof(double));
    if ( reference_wavefunction_->isCIM() ) {
        boost::shared_ptr<PSIO> psio (new PSIO());
        psio->open(PSIF_CIM,PSIO_OPEN_OLD);
        psio->read_entry(PSIF_CIM,"C matrix",(char*)&Catemp[0],nso*full*sizeof(double));
        psio->close(PSIF_CIM,1);
        C_DCOPY(nso*full,&Catemp[0],1,Ca_L,1);
        C_DCOPY(nso*full,&Catemp[0],1,Ca_R,1);
    }else {
        C_DCOPY(nso*full,&Ca[0][0],1,Ca_L,1);
        C_DCOPY(nso*full,&Ca[0][0],1,Ca_R,1);
        C_DCOPY(nso*full,&Ca[0][0],1,Catemp,1);
    }

    #pragma omp parallel for schedule (static)
    for (int mu = 0; mu < nso; mu++) {
        for (int a = 0; a < v; a++) {
            double dum = 0.0;
            for (int i = 0; i < o; i++) {
                dum += Catemp[mu*full+i+nfzc] * t1[a*o+i];
            }
            Ca_L[mu*full + a + ndocc] -= dum;
        }
    }
    #pragma omp parallel for schedule (static)
    for (int mu = 0; mu < nso; mu++) {
        for (int i = 0; i < o; i++) {
            double dum = 0.0;
            for (int a = 0; a < v; a++) {
                dum += Catemp[mu*full+a+ndocc] * t1[a*o+i];
            }
            Ca_R[mu*full + i + nfzc] += dum;
        }
    }
    free(Catemp);

    // (Q|rs)
    boost::shared_ptr<PSIO> psio(new PSIO());
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);
    psio_address addr1  = PSIO_ZERO;
    psio_address addrvo = PSIO_ZERO;
    long int nrows = 1;
    long int rowsize = nQ;
    while ( rowsize*nso*nso > o*o*v*v ) {
        nrows++;
        rowsize = nQ / nrows;
        if (nrows * rowsize < nQ) rowsize++;
        if ( rowsize == 1 ) break;
    }
    long int lastrowsize = nQ - (nrows - 1L) * rowsize;
    long int * rowdims = new long int [nrows];
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;
    for (int row = 0; row < nrows; row++) {
        psio->read(PSIF_DCC_QSO,"Qso CC",(char*)&integrals[0],rowdims[row]*nso*nso*sizeof(double),addr1,&addr1);
        F_DGEMM('n','n',full,nso*rowdims[row],nso,1.0,Ca_L,full,integrals,nso,0.0,tempv,full);
        for (int q = 0; q < rowdims[row]; q++) {
            for (int mu = 0; mu < nso; mu++) {
                C_DCOPY(full,tempv+q*nso*full+mu*full,1,integrals+q*nso*full+mu,full);
            }
        }
        F_DGEMM('n','n',full,full*rowdims[row],nso,1.0,Ca_R,full,integrals,nso,0.0,tempv,full);

        // Qoo
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int i = 0; i < o; i++) {
                for (int j = 0; j < o; j++) {
                    Qoo[(q+rowdims[0]*row)*o*o+i*o+j] = tempv[q*full*full+(i+nfzc)*full+(j+nfzc)];
                }
            }
        }
        // Qov
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int i = 0; i < o; i++) {
                for (int a = 0; a < v; a++) {
                    Qov[(q+rowdims[0]*row)*o*v+i*v+a] = tempv[q*full*full+(i+nfzc)*full+(a+ndocc)];
                }
            }
        }
        // Qvo
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int a = 0; a < v; a++) {
                for (int i = 0; i < o; i++) {
                    integrals[q*o*v+a*o+i] = tempv[q*full*full+(a+ndocc)*full+(i+nfzc)];
                }
            }
        }
        psio->write(PSIF_DCC_QSO,"qvo",(char*)&integrals[0],rowdims[row]*o*v*sizeof(double),addrvo,&addrvo);
        // Qvv
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int a = 0; a < v; a++) {
                for (int b = 0; b < v; b++) {
                    Qvv[(q+rowdims[0]*row)*v*v+a*v+b] = tempv[q*full*full+(a+ndocc)*full+(b+ndocc)];
                }
            }
        }
    }
    delete rowdims;
    psio->close(PSIF_DCC_QSO,1);
}

double GPUDFCoupledCluster::compute_energy() {
  PsiReturnType status = Success;

  //WriteBanner();
  AllocateMemory();
  status = CCSDIterations();

  // free some memory!
  free(Fij);
  free(Fab);
  free(Abij);
  free(Sbij);
  free(integrals);
  free(w1);
  free(I1);
  free(I1p);
  free(diisvec);
  free(tempv);

  // tstart in fnocc
  tstop();

  // mp2 energy
  Process::environment.globals["MP2 CORRELATION ENERGY"] = emp2;
  Process::environment.globals["MP2 TOTAL ENERGY"] = emp2 + escf;
  Process::environment.globals["MP2 OPPOSITE-SPIN CORRELATION ENERGY"] = emp2_os;
  Process::environment.globals["MP2 SAME-SPIN CORRELATION ENERGY"] = emp2_ss;

  // ccsd energy
  Process::environment.globals["CCSD CORRELATION ENERGY"] = eccsd;
  Process::environment.globals["CCSD OPPOSITE-SPIN CORRELATION ENERGY"] = eccsd_os;
  Process::environment.globals["CCSD SAME-SPIN CORRELATION ENERGY"] = eccsd_ss;
  Process::environment.globals["CCSD TOTAL ENERGY"] = eccsd + escf;
  Process::environment.globals["CURRENT ENERGY"] = eccsd + escf;

  //if (options_.get_bool("COMPUTE_TRIPLES")){
  //} else {
      free(Qoo);
      free(Qov);
      free(Qvv);
  //}

  // free remaining memory
  free(Fia);
  free(Fai);
  free(t1);
  free(tb);

  return Process::environment.globals["CURRENT ENERGY"];
}

void GPUDFCoupledCluster::UpdateT2(){
    long int v = nvirt;
    long int o = ndoccact;
    long int rs = nmo;

    boost::shared_ptr<PSIO> psio(new PSIO());

    // df (ai|bj)
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);
    psio->read_entry(PSIF_DCC_QSO,"qvo",(char*)&tempv[0],nQ*o*v*sizeof(double));
    psio->close(PSIF_DCC_QSO,1);
    F_DGEMM('n','t',o*v,o*v,nQ,1.0,tempv,o*v,tempv,o*v,0.0,integrals,o*v);

    // residual
    psio->open(PSIF_DCC_R2,PSIO_OPEN_OLD);
    psio->read_entry(PSIF_DCC_R2,"residual",(char*)&tempv[0],o*o*v*v*sizeof(double));
    psio->close(PSIF_DCC_R2,1);

    #pragma omp parallel for schedule (static)
    for (long int a=o; a<rs; a++){
        double da = eps[a];
        for (long int b=o; b<rs; b++){
            double dab = da + eps[b];
            for (long int i=0; i<o; i++){
                double dabi = dab - eps[i];
                for (long int j=0; j<o; j++){
                    long int iajb = (a-o)*v*o*o+i*v*o+(b-o)*o+j;
                    long int ijab = (a-o)*v*o*o+(b-o)*o*o+i*o+j;

                    double dijab = dabi-eps[j];
                    double tnew  = - (integrals[iajb] + tempv[ijab])/dijab;
                    tempv[ijab]  = tnew;
                }
            }
        }
    }

    if (t2_on_disk){
        psio->open(PSIF_DCC_T2,PSIO_OPEN_OLD);
        psio->read_entry(PSIF_DCC_T2,"t2",(char*)&integrals[0],o*o*v*v*sizeof(double));
        C_DAXPY(o*o*v*v,1.0,tempv,1,integrals,1);
        psio->write_entry(PSIF_DCC_T2,"t2",(char*)&integrals[0],o*o*v*v*sizeof(double));
        psio->close(PSIF_DCC_T2,1);
    }else {
        C_DAXPY(o*o*v*v,1.0,tempv,1,tb,1);
    }
}

}}

const uint32_t Polynomial = 0xEDB88320;
uint32_t crc32_bitwise(const void* data, size_t length, uint32_t previousCrc32){
    uint32_t crc = ~previousCrc32;
    uint8_t* current = (uint8_t*) data;
    while (length--) {
	crc ^= *current++;
	for (uint8_t j = 0; j < 8; j++) { // instead of
	    uint8_t lowestBit = crc & 1;
	    crc >>= 1;
	    if(lowestBit)
		crc ^= Polynomial;
	}
    }
    return ~crc;
} 
