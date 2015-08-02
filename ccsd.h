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
  protected:

    /// cc diagrams:
    virtual void CCResidual();
    void T1Fock();
    void T1Integrals();
    void UpdateT2();
    double compute_energy();
};
}}


#endif
