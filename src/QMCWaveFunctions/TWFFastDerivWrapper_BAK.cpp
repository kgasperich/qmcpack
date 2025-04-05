//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by:   Raymond Clay III, rclay@sandia.gov, Sandia National Laboratories
//
// File created by:   Raymond Clay III, rclay@sandia.gov, Sandia National Laboratories
//////////////////////////////////////////////////////////////////////////////////////

#include "QMCWaveFunctions/TWFFastDerivWrapper.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "Numerics/DeterminantOperators.h"
#include "OMPTarget/ompBLAS.hpp"
#include "type_traits/ConvertToReal.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include <iostream>
namespace qmcplusplus
{

using ValueMatrix = Matrix<QMCTraits::ValueType>;
using GradMatrix  = Matrix<QMCTraits::GradType>;
using RealMatrix  = Matrix<QMCTraits::RealType>;

TWFFastDerivWrapper::IndexType TWFFastDerivWrapper::getTWFGroupIndex(const IndexType gid) const
{
  IndexType return_group_index(-1);
  for (IndexType i = 0; i < groups_.size(); i++)
    if (gid == groups_[i])
      return_group_index = i;

  assert(return_group_index != -1);

  return return_group_index;
}

void TWFFastDerivWrapper::addGroup(const ParticleSet& P, const IndexType gid, SPOSet* spo)
{
  if (std::find(groups_.begin(), groups_.end(), gid) == groups_.end())
  {
    groups_.push_back(gid);
    spos_.push_back(spo);
  }
}

void TWFFastDerivWrapper::addMultiSlaterDet(const ParticleSet& P, const WaveFunctionComponent* msd)
{
  if (multislaterdet_)
  {
    // only one MSD is allowed to be registered
    throw std::runtime_error("Error: This TWFFastDerivWrapper already has a MultiSlaterDet");
  }
  // register msd and add pointer to wrapper
  multislaterdet_ = msd;
  /// NOTE: we could call `addGroup` for the msd->Dets here. The singledet version does that by
  /// registering the diracdets when registerTWFFastDerivWrapper is called, so this is consistent with that behavior
}

void TWFFastDerivWrapper::getM(const ParticleSet& P, std::vector<ValueMatrix>& mvec) const
{
  IndexType ngroups = spos_.size();
  for (IndexType i = 0; i < ngroups; i++)
  {
    const IndexType gid    = groups_[i];
    const IndexType first  = P.first(i);
    const IndexType last   = P.last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs  = spos_[i]->getOrbitalSetSize();
    GradMatrix tmpgmat;
    ValueMatrix tmplmat;
    tmpgmat.resize(nptcls, norbs);
    tmplmat.resize(nptcls, norbs);
    spos_[i]->evaluate_notranspose(P, first, last, mvec[i], tmpgmat, tmplmat);
  }
}

TWFFastDerivWrapper::RealType TWFFastDerivWrapper::evaluateJastrowVGL(const ParticleSet& P,
                                                                      ParticleSet::ParticleGradient& G,
                                                                      ParticleSet::ParticleLaplacian& L) const
{
  WaveFunctionComponent::LogValue logpsi = 0.0;
  G                                      = 0.0;
  L                                      = 0.0;
  for (int i = 0; i < jastrow_list_.size(); ++i)
  {
    logpsi += jastrow_list_[i]->evaluateLog(P, G, L);
  }
  RealType rval = std::real(logpsi);
  return rval;
}

TWFFastDerivWrapper::RealType TWFFastDerivWrapper::evaluateJastrowRatio(ParticleSet& P, const int iel) const
{
  //legacy calls are hit and miss with const.  Remove const for index.
  int iel_(iel);
  WaveFunctionComponent::PsiValue r(1.0);
  for (int i = 0; i < jastrow_list_.size(); ++i)
  {
    r *= jastrow_list_[i]->ratio(P, iel_);
  }

  RealType ratio_return(1.0);
  convertToReal(r, ratio_return);
  return ratio_return;
}

TWFFastDerivWrapper::RealType TWFFastDerivWrapper::calcJastrowRatioGrad(ParticleSet& P,
                                                                        const int iel,
                                                                        GradType& grad) const
{
  int iel_(iel);
  WaveFunctionComponent::PsiValue r(1.0);
  for (int i = 0; i < jastrow_list_.size(); ++i)
  {
    r *= jastrow_list_[i]->ratioGrad(P, iel_, grad);
  }
  RealType ratio_return(1.0);
  convertToReal(r, ratio_return);
  return ratio_return;
}

TWFFastDerivWrapper::GradType TWFFastDerivWrapper::evaluateJastrowGradSource(ParticleSet& P,
                                                                             ParticleSet& source,
                                                                             const int iat) const
{
  GradType grad_iat = GradType();
  for (int i = 0; i < jastrow_list_.size(); ++i)
    grad_iat += jastrow_list_[i]->evalGradSource(P, source, iat);
  return grad_iat;
}

TWFFastDerivWrapper::GradType TWFFastDerivWrapper::evaluateJastrowGradSource(
    ParticleSet& P,
    ParticleSet& source,
    const int iat,
    TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>& grad_grad,
    TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>& lapl_grad) const
{
  GradType grad_iat = GradType();
  for (int dim = 0; dim < OHMMS_DIM; dim++)
    for (int i = 0; i < grad_grad[0].size(); i++)
    {
      grad_grad[dim][i] = GradType();
      lapl_grad[dim][i] = 0.0;
    }
  for (int i = 0; i < jastrow_list_.size(); ++i)
    grad_iat += jastrow_list_[i]->evalGradSource(P, source, iat, grad_grad, lapl_grad);
  return grad_iat;
}





void TWFFastDerivWrapper::getEGradELaplM(const ParticleSet& P,
                                         std::vector<ValueMatrix>& mvec,
                                         std::vector<GradMatrix>& gmat,
                                         std::vector<ValueMatrix>& lmat) const
{
  IndexType ngroups = mvec.size();
  for (IndexType i = 0; i < ngroups; i++)
  {
    const IndexType gid    = groups_[i];
    const IndexType first  = P.first(i);
    const IndexType last   = P.last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs  = spos_[i]->getOrbitalSetSize();
    spos_[i]->evaluate_notranspose(P, first, last, mvec[i], gmat[i], lmat[i]);
  }
}

void TWFFastDerivWrapper::getIonGradM(const ParticleSet& P,
                                      const ParticleSet& source,
                                      const int iat,
                                      std::vector<std::vector<ValueMatrix>>& dmvec) const
{
  IndexType ngroups = dmvec[0].size();
  for (IndexType i = 0; i < ngroups; i++)
  {
    const IndexType gid    = groups_[i];
    const IndexType first  = P.first(i);
    const IndexType last   = P.last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs  = spos_[i]->getOrbitalSetSize();

    GradMatrix grad_phi;

    grad_phi.resize(nptcls, norbs);
    spos_[i]->evaluateGradSource(P, first, last, source, iat, grad_phi);

    for (IndexType idim = 0; idim < OHMMS_DIM; idim++)
      for (IndexType iptcl = 0; iptcl < nptcls; iptcl++)
        for (IndexType iorb = 0; iorb < norbs; iorb++)
        {
          dmvec[idim][i][iptcl][iorb] += grad_phi[iptcl][iorb][idim];
        }
  }
}

void TWFFastDerivWrapper::getIonGradIonGradELaplM(const ParticleSet& P,
                                                  const ParticleSet& source,
                                                  int iat,
                                                  std::vector<std::vector<ValueMatrix>>& dmvec,
                                                  std::vector<std::vector<GradMatrix>>& dgmat,
                                                  std::vector<std::vector<ValueMatrix>>& dlmat) const
{
  IndexType ngroups = dmvec[0].size();
  for (IndexType i = 0; i < ngroups; i++)
  {
    const IndexType gid    = groups_[i];
    const IndexType first  = P.first(i);
    const IndexType last   = P.last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs  = spos_[i]->getOrbitalSetSize();

    GradMatrix grad_phi;
    HessMatrix grad_grad_phi;
    GradMatrix grad_lapl_phi;

    grad_phi.resize(nptcls, norbs);
    grad_grad_phi.resize(nptcls, norbs);
    grad_lapl_phi.resize(nptcls, norbs);

    spos_[i]->evaluateGradSource(P, first, last, source, iat, grad_phi, grad_grad_phi, grad_lapl_phi);

    for (IndexType idim = 0; idim < OHMMS_DIM; idim++)
      for (IndexType iptcl = 0; iptcl < nptcls; iptcl++)
        for (IndexType iorb = 0; iorb < norbs; iorb++)
        {
          dmvec[idim][i][iptcl][iorb] += grad_phi[iptcl][iorb][idim];
          dlmat[idim][i][iptcl][iorb] += grad_lapl_phi[iptcl][iorb][idim];
          for (IndexType ielec = 0; ielec < OHMMS_DIM; ielec++)
            dgmat[idim][i][iptcl][iorb][ielec] += grad_grad_phi[iptcl][iorb](idim, ielec);
        }
  }
}


void TWFFastDerivWrapper::mw_getGSMatricesForDerivatives(
    const RefVectorWithLeader<TWFFastDerivWrapper>& psi_wrapper_list,
    const std::vector<std::vector<std::vector<ValueMatrix>>>& A_list,
    std::vector<std::vector<std::vector<ValueMatrix>>>& Aslice_list,
    int dim_index)

{
	/*
  const int nw = psi_wrapper_list.size();
  if (nw == 0)
    return;

  // Find the maximum number of species across all walkers
  IndexType max_nspecies = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    if (dim_index < A_list[iw].size())
      max_nspecies = std::max(max_nspecies, static_cast<IndexType>(A_list[iw][dim_index].size()));
  }

  // Process all walkers by species
  for (IndexType id = 0; id < max_nspecies; ++id)
  {
    // Process each walker for this species
    for (int iw = 0; iw < nw; ++iw)
    {
      // Skip if this walker doesn't have this dimension or species
      if (dim_index >= A_list[iw].size() || id >= A_list[iw][dim_index].size())
        continue;

      // Get the input matrix for this walker, dimension, and species
      const ValueMatrix& A_matrix = A_list[iw][dim_index][id];
      IndexType ptclnum           = A_matrix.rows();

      // Resize the output matrix
      Aslice_list[iw][dim_index][id].resize(ptclnum, ptclnum);

      // Copy the matrix elements
      for (IndexType i = 0; i < ptclnum; i++)
      {
        for (IndexType j = 0; j < ptclnum; j++)
        {
          Aslice_list[iw][dim_index][id][i][j] = A_matrix[i][j];
        }
      }
    }
  }*/
	 const int nw = psi_wrapper_list.size();
  if (nw == 0) return;

  // Process each walker independently
  #pragma omp parallel for
  for (int iw = 0; iw < nw; ++iw)
  {
    // Check dimension index validity
    if (dim_index >= A_list[iw].size()) continue;

    // Get reference to this walker's wrapper
    auto& wf = psi_wrapper_list[iw];

    // Prepare input/output for serial function
    const auto& A = A_list[iw][dim_index];
    auto& Aslice = Aslice_list[iw][dim_index];

    // Call serial implementation
    wf.getGSMatrices(A, Aslice);
  }
}


void TWFFastDerivWrapper::mw_getM(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                  const RefVectorWithLeader<ParticleSet>& p_list,
                                  std::vector<std::vector<ValueMatrix>>& m_list)
{
  const int nw = wf_list.size();
  if (nw == 0)
    return;
  
  // Update all particle sets first
  for (int iw = 0; iw < nw; ++iw){
    p_list[iw].update();
    wf_list[iw].getM(p_list[iw], m_list[iw]);
}
  /*  
  // Get the number of groups from the first walker
  const IndexType ngroups = wf_list[0].spos_.size();
  
  // Process each group
  for (IndexType i = 0; i < ngroups; i++)
  {
    // Get values for first walker to dimension arrays
    const IndexType gid    = wf_list[0].groups_[i];
    const IndexType first  = p_list[0].first(i);
    const IndexType last   = p_list[0].last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs  = wf_list[0].spos_[i]->getOrbitalSetSize();
    
    // Create vectors of references for batched call
    RefVectorWithLeader<SPOSetT<ValueType>> spo_refs(*wf_list[0].spos_[i]);
    RefVector<ValueMatrix> logdet_refs;
    RefVector<GradMatrix> dlogdet_refs;
    RefVector<ValueMatrix> d2logdet_refs;
    
    // Prepare temp matrices for each walker
    std::vector<GradMatrix> tmpgmats(nw);
    std::vector<ValueMatrix> tmplmats(nw);
    
    for (int iw = 0; iw < nw; ++iw)
    {
      spo_refs.push_back(*wf_list[iw].spos_[i]);
      logdet_refs.push_back(m_list[iw][i]);
      
      // Resize temp matrices
      tmpgmats[iw].resize(nptcls, norbs);
      tmplmats[iw].resize(nptcls, norbs);
      
      dlogdet_refs.push_back(tmpgmats[iw]);
      d2logdet_refs.push_back(tmplmats[iw]);
    }
    
    // Make batched call to evaluate_notranspose
    spo_refs[0].mw_evaluate_notranspose(spo_refs, p_list, first, last, 
                                      logdet_refs, dlogdet_refs, d2logdet_refs);
  }*/
}
void TWFFastDerivWrapper::mw_invertMatrices(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                            const std::vector<std::vector<ValueMatrix>>& M_list,
                                            std::vector<std::vector<ValueMatrix>>& Minv_list)
{
	/*
  const int nw = wf_list.size();
  if (nw == 0)
    return;

  // Find the maximum number of species across all walkers
  IndexType max_nspecies = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    max_nspecies = std::max(max_nspecies, static_cast<IndexType>(M_list[iw].size()));
  }

  // Ensure Minv_list has the correct size for all walkers
  for (int iw = 0; iw < nw; ++iw)
  {
    Minv_list[iw].resize(M_list[iw].size());
  }

  // Process all walkers by species
  for (IndexType id = 0; id < max_nspecies; ++id)
  {
    for (int iw = 0; iw < nw; ++iw)
    {
      // Skip if this walker doesn't have this species
      if (id >= M_list[iw].size())
        continue;

      // Copy matrix and then invert it
      const ValueMatrix& M_matrix = M_list[iw][id];

      // Verify that the matrix is square
      assert(M_matrix.cols() == M_matrix.rows());

      // Copy the input matrix to the output matrix
      Minv_list[iw][id] = M_matrix;

      // Invert the matrix
      invert_matrix(Minv_list[iw][id]);
    }
  }*/

const int nw = wf_list.size();
  if (nw == 0) return;

  // Process each walker independently
  for (int iw = 0; iw < nw; ++iw)
  {
    // Get references for this walker
    auto& wf = wf_list[iw];
    const auto& M = M_list[iw];
    auto& Minv = Minv_list[iw];
    
    // Call serial implementation
    wf.invertMatrices(M, Minv);
  }


}

void TWFFastDerivWrapper::computeMDDerivatives_Obs(const std::vector<ValueMatrix>& Minv_Mv,
                                                   const std::vector<ValueMatrix>& Minv_B,
                                                   const std::vector<IndexType>& mdd_spo_ids,
                                                   const std::vector<const WaveFunctionComponent*>& mdds,
                                                   std::vector<ValueVector>& dvals_O) const
{
  // mdd_id is multidiracdet id as ordered in multislaterdet
  for (size_t mdd_id = 0; mdd_id < mdd_spo_ids.size(); mdd_id++)
  {
    // sid is sposet id as ordered in TWFFastDerivWrapper (index into first dim of M, X, B, etc.)
    IndexType sid = mdd_spo_ids[mdd_id];

    const auto& multidiracdet_i = static_cast<const MultiDiracDeterminant&>(*mdds[mdd_id]);

    IndexType ndet     = multidiracdet_i.getNumDets();     // number of DiracDets in this MultiDiracDet
    size_t nelec       = multidiracdet_i.getNumPtcls();    // total occ orbs in refdet (should be same as n_elec)
    size_t nocc        = nelec;                            // total occ orbs in refdet (should be same as n_elec)
    size_t norb        = multidiracdet_i.getNumOrbitals(); // total occ + virt orbs
    size_t nvirt       = norb - nocc;                      // number of orbs past those occupied in refdet
    size_t virt_offset = nocc;                             // first idx of virt orbs within norb
    assert(norb == numOrbitals(sid));

    /// NOTE: in any place where we handle virtuals, we will only ever need the ones which appear as particles in the excited dets
    ///       in *SOME* places where we handle occupied orbs, we only need the ones which appear as holes in the excited dets
    ///       these are denoted as h (for hole) below

    // indices:
    // h: holes (occupied in refdet, unoccupied in some exc det)
    // o: occupied in refdet
    // e: electron (could be other particle, but typically electron, and this avoids confusion with the other meaning of "particle")
    // v: virtual (anything unoccupied in refdet; for better performance, can be restricted to only consist of particles in excited dets))
    // n: full orb list (o+v)

    // currently, "h" is handled the same as "o", but denoted separately below to clarify where we can reduce computation


    // quantities to compute are:
    // OD/D = tr( {a}^-1 . {S})
    // where {X} is subset of X corresponding to holes/particles (rows/cols) for a particular excited det D
    // a = Minv[h,e].M[e,v]
    // S = (Minv[h,e].B[e,v] - Minv[h,e].B[e,o].Minv[o,e].M[e,v]) ("M" from paper)


    // input mats:
    // Minv_Mv[o,v] = Minv[o,e].M[e,v]
    // Minv_B[h,n]  = Minv[h,e].B[e,n]

    /// NOTE: we only need Minv[o,e] for the third term in Minv.B.Minv.Mv; otherwise only need [h] for first dimension
    ///       can use Minv[h/o,e].B[e,o].Minv[o,e] if already available from elsewhere


    /// NOTE: gemm below is reversed from what is written in comments
    // c = a.b
    // c.T = (b.T) . (a.T)
    // a,b are row-major, but gemm assumes col-major layout, so reorder args
    // if we want output to be col-major, we can reverse inputs and use 't','t'

    // s[h,v] = -Minv_B[h,o].Minv_Mv[o,v]
    ValueMatrix s_hv(nocc, nvirt);
    BLAS::gemm('n', 'n', nvirt, nocc, nocc, -1.0, Minv_Mv[sid].data(), Minv_Mv[sid].cols(), Minv_B[sid].data(),
               Minv_B[sid].cols(), 0.0, s_hv.data(), s_hv.cols());

    // s[h,v] = Minv_B[h,v] - Minv_B[h,o].Minv_Mv[o,v]
    for (size_t i = 0; i < s_hv.rows(); i++)
      for (size_t j = 0; j < s_hv.cols(); j++)
        s_hv(i, j) += Minv_B[sid](i, j + virt_offset);


    // compute difference from ref here and add that term back later
    dvals_O[mdd_id][0] = 0.0;


    // dvals_O[idet] = tr(inv({Minv_Mv}).{s_hv})
    // build full mats for all required pairs, then select submatrices for each excited det
    // use exc index data to map into full arrays and create [k,k] tables

    // skip ref/ground state determinant
    size_t det_offset  = 1;
    size_t data_offset = 1;
    int max_exc_level  = multidiracdet_i.getMaxExcLevel();

    /// NOTE: this is only an OffloadVector because that's what is present in MultiDiracDeterminant
    const OffloadVector<int>& excdata = multidiracdet_i.getDetData();

    // update indices into list of determinants and excitation data
    auto update_offsets = [&](size_t exc_level) {
      det_offset += multidiracdet_i.getNdetPerExcLevel(exc_level);
      data_offset += multidiracdet_i.getNdetPerExcLevel(exc_level) * (3 * exc_level + 1);
    };

    // dets ordered as {ref, all singles, all doubles, all triples, ...}
    //      shift by ndet_per_exc_level after each exc_level
    // data is ordered in same way, but each det has several contiguous elements
    //      each det of exc_level k has (3*k+1) elements: [k, {pos}, {uno}, {ocp}]
    //      k: exc level
    //      pos : positions of holes in refdet
    //      uno : MO idx of particles
    //      ocp : MO idx of holes

    for (size_t k = 1; k <= max_exc_level; k++)
    {
      size_t ndet_exc = multidiracdet_i.getNdetPerExcLevel(k);

      // hole/particle idx for an excited det
      std::vector<IndexType> hlist(k, 0);
      std::vector<IndexType> plist(k, 0);

      // submatrix for rows/cols corresponding to holes/particles
      // a is also ainv (inverse is computed in place)
      ValueMatrix a(k, k); // {Minv_Mv} and inv({Minv_Mv})

      // loop over all dets of this excitation level
      for (size_t idet = 0; idet < ndet_exc; idet++)
      {
        size_t excdata_offset = data_offset + idet * (3 * k + 1);
        assert(excdata[excdata_offset] == k); // first element of det data is exc level
        for (size_t i = 0; i < k; i++)
        {
          plist[i] = excdata[excdata_offset + k + 1 + i];
          hlist[i] = excdata[excdata_offset + 2 * k + 1 + i];
        }

        // construct submatrix a (no need to explicitly construct s, just get elements later)
        for (size_t i = 0; i < k; i++)
          for (size_t j = 0; j < k; j++)
            a(i, j) = Minv_Mv[sid](hlist[i], plist[j] - virt_offset);

        // invert a in place
        invert_matrix(a, false);

        // dval_O_excdet - dval_O_refdet = tr(ainv.s)
        ValueType dval_O = 0.0;

        // tr(ainv.s)
        for (size_t i = 0; i < k; i++)
          for (size_t j = 0; j < k; j++)
            dval_O += a(i, j) * s_hv(hlist[j], plist[i] - virt_offset);

        dvals_O[mdd_id][det_offset + idet] = dval_O;
      }

      // update offsets
      update_offsets(k);
    }
  }

  return;
}

void TWFFastDerivWrapper::transform_Av_AoBv(const ValueMatrix& A, const ValueMatrix& B, ValueMatrix& X) const
{
  // A [h,o+v]
  // B [o,v]

  // return A[h,v] - A[h,o].B[o,v]

  const int nholes = A.rows();
  const int nocc   = B.rows();
  const int nvirt  = B.cols();

  assert(A.cols() == nocc + nvirt);

  for (size_t i = 0; i < nholes; i++)
    for (size_t j = 0; j < nvirt; j++)
      X(i, j) = A(i, j + nocc);

  // X = X - A[h,o].B[o,v]
  BLAS::gemm('n', 'n', nvirt, nholes, nocc, -1.0, B.data(), B.cols(), A.data(), A.cols(), 1.0, X.data(), X.cols());
  return;
}

void TWFFastDerivWrapper::computeMDDerivatives_dmu(const std::vector<ValueMatrix>& Minv_Mv,
                                                   const std::vector<ValueMatrix>& Minv_B,
                                                   const std::vector<ValueMatrix>& Minv_dM,
                                                   const std::vector<ValueMatrix>& Minv_dB,
                                                   const std::vector<IndexType>& mdd_spo_ids,
                                                   const std::vector<const WaveFunctionComponent*>& mdds,
                                                   std::vector<ValueVector>& dvals_dmu_O,
                                                   std::vector<ValueVector>& dvals_dmu) const
{
  // mdd_id is multidiracdet id as ordered in multislaterdet
  for (size_t mdd_id = 0; mdd_id < mdd_spo_ids.size(); mdd_id++)
  {
    // sid is sposet id as ordered in TWFFastDerivWrapper (index into first dim of M, X, B, etc.)
    IndexType sid = mdd_spo_ids[mdd_id];

    const auto& multidiracdet_i = static_cast<const MultiDiracDeterminant&>(*mdds[mdd_id]);

    IndexType ndet     = multidiracdet_i.getNumDets();     // number of DiracDets in this MultiDiracDet
    size_t nelec       = multidiracdet_i.getNumPtcls();    // total occ orbs in refdet (should be same as n_elec)
    size_t nocc        = nelec;                            // total occ orbs in refdet (should be same as n_elec)
    size_t norb        = multidiracdet_i.getNumOrbitals(); // total Occ + virt orbs
    size_t nvirt       = norb - nocc;                      // number of orbs past those occupied in refdet
    size_t virt_offset = nocc;                             // first idx of virt orbs within norb
    assert(norb == numOrbitals(sid));


    // input mats:
    // a  = Minv_Mv  [occ, virt]
    // X2 = Minv_dM  [occ, occ+virt]
    // X3 = Minv_B   [occ, occ+virt]
    // X4 = Minv_dB  [occ, occ+virt]


    /// NOTE: in any place where we handle virtuals, we will only ever need the ones which appear as particles in the excited dets
    ///       in *SOME* places where we handle occupied orbs, we only need the ones which appear as holes in the excited dets
    ///       these are denoted as h (for hole) below

    // indices:
    // h: holes (occupied in refdet, unoccupied in some exc det)
    // o: occupied in refdet
    // e: electron (could be other particle, but typically electron, and this avoids confusion with the other meaning of "particle")
    // v: virtual (anything unoccupied in refdet; for better performance, can be restricted to only consist of particles in excited dets))
    // n: full orb list (o+v)

    // currently, "h" is handled the same as "o", but denoted separately below to clarify where we can reduce computation


    // dvals_dmu_O[idet] = tr(
    //     - inv({a}) . {X2[h,v] - X2[h,o].a[o,v]} . inv({a}) . {X3[h,v] - X3[h,o].a[o,v]} ...
    //     + inv({a}) . {(X4[h,v] - X3[h,o].X2[o,v]) - (X4[h,o] - X3[h,o].X2[o,o]).a[o,v] - X2[h,o].(X3[o,v] - X3[o,o].a[o,v])}
    //   )

    // we have several terms like A'[h/o,v] = (A[h/o,v] - A[h/o,o].a[o,v])
    //
    // if X432[h,o/v] = (X4[h,o/v] - X3[h,o].X2[o,o/v])
    // dvals_dmu_O[idet] = tr(-inv({a}) . {X2'[h,v]} . inv({a}) . {X3'[h,v]}  + inv({a}) . {X432'[h,v] - X2[h,o].X3'[o,v]})

    // X32[h,n] = X3[h,o].X2[o,n]
    // X32[h,n] = Minv_B[h,o].Minv_dM[o,n]
    ValueMatrix X32_hn(nocc, norb);
    BLAS::gemm('n', 'n', norb, nocc, nocc, 1.0, Minv_dM[sid].data(), Minv_dM[sid].cols(), Minv_B[sid].data(),
               Minv_B[sid].cols(), 0.0, X32_hn.data(), X32_hn.cols());


    // Minv_dB - X32
    // Minv_dB[h,n] - Minv_B[h,o].Minv_dM[o,n]
    ValueMatrix X432_hn(nocc, norb);
    X432_hn = Minv_dB[sid] - X32_hn;

    // (X432[h,v] - X432[h,o].a[o,v])
    ValueMatrix X432b_hv(nocc, nvirt);
    transform_Av_AoBv(X432_hn, Minv_Mv[sid], X432b_hv);

    // (X2[h,v] - X2[h,o].a[o,v])
    ValueMatrix X2b_hv(nocc, nvirt);
    transform_Av_AoBv(Minv_dM[sid], Minv_Mv[sid], X2b_hv);

    // (X3[o,v] - X3[o,o].a[o,v])
    ValueMatrix X3b_ov(nocc, nvirt);
    transform_Av_AoBv(Minv_B[sid], Minv_Mv[sid], X3b_ov);

    // X432b[h,v] -= Minv_dM[h,o].X3b[o,v]
    BLAS::gemm('n', 'n', nvirt, nocc, nocc, -1.0, X3b_ov.data(), X3b_ov.cols(), Minv_dM[sid].data(),
               Minv_dM[sid].cols(), 1.0, X432b_hv.data(), X432b_hv.cols());


    // compute difference from ref here and add that term back later
    dvals_dmu_O[mdd_id][0] = 0.0;
    dvals_dmu[mdd_id][0]   = 0.0;


    // a' = inv({a})
    // dvals_dmu[idet]   = tr(a'.{X2b})
    // dvals_dmu_O[idet] = tr(-a'.{X2b}.a'.{X3b} + a'.{X432b})
    // (X432b here is after subtracting Minv_dM[h,o].X3b[o,v])


    // skip ref/ground state determinant
    size_t det_offset  = 1;
    size_t data_offset = 1;
    int max_exc_level  = multidiracdet_i.getMaxExcLevel();

    /// NOTE: this is only an OffloadVector because that's what is present in MultiDiracDeterminant
    const OffloadVector<int>& excdata = multidiracdet_i.getDetData();

    // update indices into list of determinants and excitation data
    auto update_offsets = [&](size_t exc_level) {
      det_offset += multidiracdet_i.getNdetPerExcLevel(exc_level);
      data_offset += multidiracdet_i.getNdetPerExcLevel(exc_level) * (3 * exc_level + 1);
    };

    // dets ordered as {ref, all singles, all doubles, all triples, ...}
    //      shift by ndet_per_exc_level after each exc_level
    // data is ordered in same way, but each det has several contiguous elements
    //      each det of exc_level k has (3*k+1) elements: [k, {pos}, {uno}, {ocp}]
    //      k: exc level
    //      pos : positions of holes in refdet
    //      uno : MO idx of particles
    //      ocp : MO idx of holes

    for (size_t k = 1; k <= max_exc_level; k++)
    {
      size_t ndet_exc = multidiracdet_i.getNdetPerExcLevel(k);

      // hole/particle idx for an excited det
      std::vector<IndexType> hlist(k, 0);
      std::vector<IndexType> plist(k, 0);

      // a' = inv({a})
      // dvals_dmu_O[idet] = tr(-a'.{X2b}.a'.{X3b} + a'.{X432b})
      // dvals_dmu[idet]   = tr(a'.{X2b})


      // submatrix for rows/cols corresponding to holes/particles
      // a is also ainv (inverse is computed in place)
      ValueMatrix a(k, k); // {Minv_Mv} and inv({Minv_Mv})
      ValueMatrix x2(k, k);
      ValueMatrix x3(k, k);
      ValueMatrix x4(k, k);

      ValueMatrix ainv_x2(k, k); // tr -> d(logD)
      ValueMatrix ainv_x3(k, k);

      // loop over all dets of this excitation level
      for (size_t idet = 0; idet < ndet_exc; idet++)
      {
        size_t excdata_offset = data_offset + idet * (3 * k + 1);
        assert(excdata[excdata_offset] == k); // first element of det data is exc level
        for (size_t i = 0; i < k; i++)
        {
          plist[i] = excdata[excdata_offset + k + 1 + i];
          hlist[i] = excdata[excdata_offset + 2 * k + 1 + i];
        }

        // construct submatrices (don't need x4, but makes code more readable)
        for (size_t i = 0; i < k; i++)
          for (size_t j = 0; j < k; j++)
          {
            a(i, j)  = Minv_Mv[sid](hlist[i], plist[j] - virt_offset);
            x2(i, j) = X2b_hv(hlist[i], plist[j] - virt_offset);
            x3(i, j) = X3b_ov(hlist[i], plist[j] - virt_offset);
            x4(i, j) = X432b_hv(hlist[i], plist[j] - virt_offset);
          }

        // invert a in place
        invert_matrix(a, false);

        // ainv_x3 = ainv.x3
        BLAS::gemm('n', 'n', k, k, k, 1.0, x3.data(), x3.cols(), a.data(), a.cols(), 0.0, ainv_x3.data(),
                   ainv_x3.cols());

        // ainv_x2 = ainv.x2
        BLAS::gemm('n', 'n', k, k, k, 1.0, x2.data(), x2.cols(), a.data(), a.cols(), 0.0, ainv_x2.data(),
                   ainv_x2.cols());

        // dval_dmu_O = tr(ainv.x4 - ainv_x2.ainv_x3)
        // dval_dmu = tr(ainv_x2)

        ValueType dval_dmu_O = 0.0;
        ValueType dval_dmu   = 0.0;


        for (size_t i = 0; i < k; i++)
        {
          dval_dmu += ainv_x2(i, i);
          for (size_t j = 0; j < k; j++)
          {
            dval_dmu_O += a(i, j) * x4(j, i) - ainv_x2(i, j) * ainv_x3(j, i);
          }
        }

        dvals_dmu_O[mdd_id][det_offset + idet] = dval_dmu_O;
        dvals_dmu[mdd_id][det_offset + idet]   = dval_dmu;
      }

      // update offsets
      update_offsets(k);
    }
  }

  return;
}

std::tuple<TWFFastDerivWrapper::ValueType, TWFFastDerivWrapper::ValueType, TWFFastDerivWrapper::ValueType>
    TWFFastDerivWrapper::computeMDDerivatives_total(const std::vector<const WaveFunctionComponent*>& mdds,
                                                    const std::vector<ValueVector>& dvals_dmu_O,
                                                    const std::vector<ValueVector>& dvals_O,
                                                    const std::vector<ValueVector>& dvals_dmu) const
{
  /**
   * \f[
   *   \partial_\mu\frac{\hat{O} \Psi}{\Psi} =
   *       \frac{\sum_i c_i D_i \left(\partial_\mu\frac{\hat{O}D_i}{D_i}\right)}{\Psi} 
   *     + \frac{\sum_i c_i D_i \left(\frac{\hat{O}D_i}{D_i}\right)\partial_\mu\log(D_i)}{\Psi} 
   *     - \left(\frac{\sum_i c_i D_i \left(\frac{\hat{O}D_i}{D_i}\right)}{\Psi}\right)
   *     * \left(\frac{\sum_i c_i D_i \partial_\mu \log(D_i)}{\Psi}\right)
   * \f]
   * 
   * with:
   *  w_i = c_i D_i
   *  x_i = d_mu(O D_i/D_i)
   *  y_i = O D_i/D_i
   *  z_i = d_mu(log(D_i))
   * 
   * \f[
   *   \partial_\mu\frac{\hat{O} \Psi}{\Psi} =
   *       \frac{\sum_i w_i * x_i}{\Psi} 
   *     + \frac{\sum_i w_i * y_i * z_i}{\Psi} 
   *     - \left(\frac{\sum_i w_i * y_i}{\Psi}\right)
   *     * \left(\frac{\sum_i w_i * z_i}{\Psi}\right)
   * \f]
   *
   * D is slaterdet composed of spindets
   * D_i = A_{a_i}*B_{b_i} 
   * A and B are alpha/beta spindets
   * a and b map from slaterdet list into unique alpha/beta lists
   * 
   * with D = AB
   *   OD/D = O(AB)/(AB) = ((OA)B + A(OB))/(AB) = OA/A + OB/B
   *   d(OD/D) = d(OA/A + OB/B) = d(OA/A) + d(OB/B)
   *   d(log(D)) = d(AB)/(AB) = ((dA)B + A(dB))/(AB) = dA/A + dB/B = d(log(A)) + d(log(B))
   * 
   */

  // multislaterdet
  const auto& msd = static_cast<const MultiSlaterDetTableMethod&>(getMultiSlaterDet());

  // total slaterdets
  const int num_slaterdets = msd.getNumSlaterDets();

  // Coefs for slaterdets
  const std::vector<ValueType>& C = msd.get_C();

  // mapping from [group_idx][slater_idx] to diracdet_idx
  const std::vector<std::vector<size_t>>& C2node = msd.get_C2node();

  // number of unique diracdets of each group
  std::vector<IndexType> num_diracdets;

  std::vector<const OffloadVector<ValueType>*> diracdet_ratios_to_ref;

  for (size_t i = 0; i < msd.getDetSize(); i++)
  {
    num_diracdets.push_back(msd.getDet(i).getNumDets());
    diracdet_ratios_to_ref.push_back(&msd.getDet(i).getRatiosToRefDet());
  }

  const int num_groups = num_diracdets.size();

  ValueType total_psi   = C[0]; // sum_i c_i D_i
  ValueType total_dmu_O = 0.0;  // d_mu(OD/D)
  ValueType total_O     = 0.0;  // OD/D
  ValueType total_dmu   = 0.0;  // d_mu(log(D))
  ValueType total_Odmu  = 0.0;  // (OD/D) * d_mu(log(D))

  /// NOTE: sum doesn't include refdet
  /// total terms are added later; psi C0 term is added above
  for (size_t i_sd = 1; i_sd < num_slaterdets; i_sd++)
  {
    ValueType tmp_psi   = C[i_sd]; // C[i_sd] * prod_i ratio[i][i_sd]
    ValueType tmp_dmu_O = 0.0;     // d_mu(OD/D)
    ValueType tmp_O     = 0.0;     // OD/D
    ValueType tmp_dmu   = 0.0;     // d_mu(log(D))
    ValueType tmp_Odmu  = 0.0;     // (OD/D) * d_mu(log(D))

    for (size_t i_group = 0; i_group < num_groups; i_group++)
    {
      size_t i_dd = C2node[i_group][i_sd];

      tmp_dmu_O += dvals_dmu_O[i_group][i_dd];
      tmp_O += dvals_O[i_group][i_dd];
      tmp_dmu += dvals_dmu[i_group][i_dd];
      tmp_Odmu += dvals_dmu[i_group][i_dd] * dvals_O[i_group][i_dd];
      tmp_psi *= (*diracdet_ratios_to_ref[i_group])[i_dd];
    }

    total_psi += tmp_psi;
    total_dmu_O += tmp_dmu_O * tmp_psi;
    total_O += tmp_O * tmp_psi;
    total_dmu += tmp_dmu * tmp_psi;
    // total_Odmu += tmp_Odmu * tmp_psi;
    total_Odmu += tmp_O * tmp_dmu * tmp_psi;
  }

  ValueType norm_dmu_O = total_dmu_O / total_psi;
  ValueType norm_O     = total_O / total_psi;
  ValueType norm_dmu   = total_dmu / total_psi;
  ValueType norm_Odmu  = total_Odmu / total_psi;

  // d_mu(log(Psi))
  ValueType dmu_psi = norm_dmu;

  // (OPsi/Psi) (this is the same for all spatial derivs, but we get it for free here)
  ValueType Opsi = norm_O;

  // d_mu(OPsi/Psi)
  ValueType dmu_O_psi = norm_Odmu + norm_dmu_O - norm_O * norm_dmu;

  return {dmu_O_psi, dmu_psi, Opsi};
}

void TWFFastDerivWrapper::invertMatrices(const std::vector<ValueMatrix>& M, std::vector<ValueMatrix>& Minv)
{
  IndexType nspecies = M.size();
  for (IndexType id = 0; id < nspecies; id++)
  {
    assert(M[id].cols() == M[id].rows());
    Minv[id] = M[id];
    invert_matrix(Minv[id]);
  }
}

void TWFFastDerivWrapper::buildX(const std::vector<ValueMatrix>& Minv,
                                 const std::vector<ValueMatrix>& B,
                                 std::vector<ValueMatrix>& X)
{
  IndexType nspecies = Minv.size();

  for (IndexType id = 0; id < nspecies; id++)
  {
    int ptclnum = Minv[id].rows();
    assert(Minv[id].rows() == Minv[id].cols());
    ValueMatrix tmpmat;
    X[id].resize(ptclnum, ptclnum);
    tmpmat.resize(ptclnum, ptclnum);
    //(B*A^-1)
    for (int i = 0; i < ptclnum; i++)
      for (int j = 0; j < ptclnum; j++)
        for (int k = 0; k < ptclnum; k++)
        {
          tmpmat[i][j] += B[id][i][k] * Minv[id][k][j];
        }
    //A^{-1}*B*A^{-1}
    for (int i = 0; i < ptclnum; i++)
      for (int j = 0; j < ptclnum; j++)
        for (int k = 0; k < ptclnum; k++)
        {
          X[id][i][j] += Minv[id][i][k] * tmpmat[k][j];
        }
  }
}
<<<<<<< HEAD
void TWFFastDerivWrapper::mw_buildX(const RefVectorWithLeader<TWFFastDerivWrapper>& psi_wrapper_list,
                                   const std::vector<std::vector<ValueMatrix>>& Minv,
                                   const std::vector<std::vector<ValueMatrix>>& B_gs,
                                   std::vector<std::vector<ValueMatrix>>& X)
=======
void TWFFastDerivWrapper::buildIntermediates(const std::vector<ValueMatrix>& Minv,
                                             const std::vector<ValueMatrix>& B,
                                             const std::vector<ValueMatrix>& M,
                                             std::vector<ValueMatrix>& X,
                                             std::vector<ValueMatrix>& Minv_B,
                                             std::vector<ValueMatrix>& Minv_Mv)
{
  IndexType nspecies = Minv.size();

  for (IndexType id = 0; id < nspecies; id++)
  {
    int ptclnum = Minv[id].rows();
    assert(Minv[id].rows() == Minv[id].cols());
    assert(X[id].rows() == ptclnum);
    assert(X[id].cols() == ptclnum);
    int norb  = M[id].cols();
    int nvirt = norb - ptclnum;

    // o: occupied orbs in ground state det
    // e: electrons (or other species)
    // v: virtual orbs (unoccupied in ground state det)
    // n: all orbs (o+v)

    // Minv_Mv = Minv[o,e].M[e,v]
    BLAS::gemm('n', 'n', nvirt, ptclnum, ptclnum, 1.0, M[id].data() + ptclnum, M[id].cols(), Minv[id].data(),
               Minv[id].cols(), 0.0, Minv_Mv[id].data(), Minv_Mv[id].cols());

    // Minv_B = Minv[o,e].B[e,n]
    BLAS::gemm('n', 'n', norb, ptclnum, ptclnum, 1.0, B[id].data(), B[id].cols(), Minv[id].data(), Minv[id].cols(), 0.0,
               Minv_B[id].data(), Minv_B[id].cols());


    // X = Minv_B[o,o].Minv[o,e]
    BLAS::gemm('n', 'n', ptclnum, ptclnum, ptclnum, 1.0, Minv[id].data(), Minv[id].cols(), Minv_B[id].data(),
               Minv_B[id].cols(), 0.0, X[id].data(), X[id].cols());
  }
}


void TWFFastDerivWrapper::buildIntermediates_dmu(const std::vector<ValueMatrix>& Minv,
                                                 const std::vector<std::vector<ValueMatrix>>& dB,
                                                 const std::vector<std::vector<ValueMatrix>>& dM,
                                                 std::vector<std::vector<ValueMatrix>>& Minv_dB,
                                                 std::vector<std::vector<ValueMatrix>>& Minv_dM)
{
  IndexType nspecies = Minv.size();
  IndexType ndim     = dB.size();

  // o: occupied orbs in ground state det
  // e: electrons (or other species)
  // n: all orbs (o+v)
  for (IndexType id = 0; id < nspecies; id++)
  {
    int ptclnum = Minv[id].rows();
    assert(Minv[id].rows() == Minv[id].cols());

    for (IndexType idim = 0; idim < ndim; idim++)
    {
      int norb = dM[idim][id].cols();
      // Minv_dM = Minv[o,e].dM[e,n]
      BLAS::gemm('n', 'n', norb, ptclnum, ptclnum, 1.0, dM[idim][id].data(), dM[idim][id].cols(), Minv[id].data(),
                 Minv[id].cols(), 0.0, Minv_dM[idim][id].data(), Minv_dM[idim][id].cols());
      // Minv_dB = Minv[o,e].dB[e,n]
      BLAS::gemm('n', 'n', norb, ptclnum, ptclnum, 1.0, dB[idim][id].data(), dB[idim][id].cols(), Minv[id].data(),
                 Minv[id].cols(), 0.0, Minv_dB[idim][id].data(), Minv_dB[idim][id].cols());
    }
  }
}

>>>>>>> e51f733f1a9a2cb33fd0d48abcc2933b2387081e

{
	 const int nw = psi_wrapper_list.size();
  if (nw == 0) return;

  // Process each walker independently
  #pragma omp parallel for
  for (int iw = 0; iw < nw; ++iw)
  {
    // Get references for this walker
    auto& wf = psi_wrapper_list[iw];
    const auto& walker_Minv = Minv[iw];
    const auto& walker_B_gs = B_gs[iw];
    auto& walker_X = X[iw];

    // Ensure X is properly sized
    walker_X.resize(walker_Minv.size());

    // Call serial implementation
    wf.buildX(walker_Minv, walker_B_gs, walker_X);
  }
	/*
  const int nw = psi_wrapper_list.size();
  if (nw == 0)
    return;

  // Find the maximum number of species across all walkers
  IndexType max_nspecies = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    max_nspecies = std::max(max_nspecies, static_cast<IndexType>(Minv[iw].size()));
  }

  // Process all walkers by species
  for (IndexType id = 0; id < max_nspecies; ++id)
  {
    // Process each walker for this species
    for (int iw = 0; iw < nw; ++iw)
    {
      // Skip if this walker doesn't have this species
      if (id >= Minv[iw].size() || id >= B_gs[iw].size())
        continue;

      // Get the input matrices for this walker and species
      const ValueMatrix& Minv_matrix = Minv[iw][id];
      const ValueMatrix& B_matrix    = B_gs[iw][id];
      IndexType ptclnum              = Minv_matrix.rows();

      // Skip if matrices don't match
      if (Minv_matrix.rows() != Minv_matrix.cols() || B_matrix.rows() != ptclnum || B_matrix.cols() != ptclnum)
        continue;

      // Temp matrix for intermediate result (B*A^-1)
      ValueMatrix tmpmat;
      tmpmat.resize(ptclnum, ptclnum);

      // BLAS constants
      constexpr char transa = 'n';
      constexpr char transb = 'n';
      constexpr ValueType zone(1);
      constexpr ValueType zero(0);

      // Compute (B*A^-1) using BLAS gemm
      BLAS::gemm(transa, transb,
                ptclnum, ptclnum, ptclnum,
                zone,
                B_matrix.data(), ptclnum,      // B matrix and its leading dimension
                Minv_matrix.data(), ptclnum,   // Minv matrix and its leading dimension
                zero,
                tmpmat.data(), ptclnum);       // tmpmat and its leading dimension

      // Compute A^{-1}*B*A^{-1} using BLAS gemm
      BLAS::gemm(transa, transb,
                ptclnum, ptclnum, ptclnum,
                zone,
                Minv_matrix.data(), ptclnum,  // Minv matrix and its leading dimension
                tmpmat.data(), ptclnum,       // tmpmat and its leading dimension
                zero,
                X[iw][id].data(), ptclnum);   // X matrix and its leading dimension
    }
  }*/
}
void TWFFastDerivWrapper::wipeMatrices(std::vector<ValueMatrix>& A)
{
  for (IndexType id = 0; id < A.size(); id++)
  {
    A[id] = 0.0;
  }
}

<<<<<<< HEAD


void TWFFastDerivWrapper::mw_trAB(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                const std::vector<std::vector<ValueMatrix>>& A_list,
                                const std::vector<std::vector<std::vector<ValueMatrix>>>& B_list,
                                std::vector<ValueType>& results,
                                int dim)
{
  const int nw = wf_list.size();
  if (nw == 0)
    return;
  results.resize(nw, 0.0);
  
  // Process each walker in parallel
  for (int w = 0; w < nw; ++w)
  {
    // Get the walker's data
    auto& wf = wf_list[w];
    
    // Call the serial trAB function for this walker
    results[w] = wf.trAB(A_list[w], B_list[w][dim]);
=======
void TWFFastDerivWrapper::wipeVectors(std::vector<ValueVector>& A)
{
  for (IndexType id = 0; id < A.size(); id++)
  {
    A[id] = 0.0;
>>>>>>> e51f733f1a9a2cb33fd0d48abcc2933b2387081e
  }
}

TWFFastDerivWrapper::ValueType TWFFastDerivWrapper::trAB(const std::vector<ValueMatrix>& A,
                                                         const std::vector<ValueMatrix>& B)
{
  IndexType nspecies = A.size();
  assert(A.size() == B.size());
  ValueType val = 0.0;
  //Now to compute the kinetic energy
  for (IndexType id = 0; id < nspecies; id++)
  {
    int ptclnum      = A[id].rows();
    ValueType val_id = 0.0;
    assert(A[id].cols() == B[id].rows() && A[id].rows() == B[id].cols());
    for (int i = 0; i < A[id].rows(); i++)
      for (int j = 0; j < A[id].cols(); j++)
      {
        val_id += A[id][i][j] * B[id][j][i];
      }
    val += val_id;
  }

  return val;
}
/*
void TWFFastDerivWrapper::mw_trAB(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                const std::vector<std::vector<ValueMatrix>>& A_list,
                                const std::vector<std::vector<std::vector<ValueMatrix>>>& B_list,
                                std::vector<ValueType>& results,
                                int dim)
{
  const int nw = wf_list.size();
  if (nw == 0)
    return;
  results.resize(nw, 0.0);

  // Find max number of species across walkers
  int max_species = 0;
  for (int w = 0; w < nw; ++w)
    max_species = std::max(max_species, static_cast<int>(A_list[w].size()));

#pragma omp parallel
  {
#pragma omp for
    for (int s = 0; s < max_species; ++s)
    {
      // For each walker that has this species
      for (int w = 0; w < nw; ++w)
      {
        // Skip if walker doesn't have this species or dimension
        if (s >= A_list[w].size() || dim >= B_list[w].size() || s >= B_list[w][dim].size())
          continue;

        const ValueMatrix& A = A_list[w][s];
        const ValueMatrix& B = B_list[w][dim][s];
        int rows = A.rows();
        int cols = A.cols();

        // Ensure matrices have compatible dimensions
        if (cols != B.rows() || rows != B.cols())
          continue;

        ValueType val_s = 0.0;

        // Preallocate vectors for transposed columns of B
        std::vector<ValueType> B_col(cols);

        // For each row of A, compute dot product with corresponding column of B^T
        for (int i = 0; i < rows; ++i)
        {
          // Extract column i of B^T (which is row i of B)
          for (int j = 0; j < cols; ++j)
          {
            B_col[j] = B[j][i];
          }

          // Compute dot product: A[i,:] · B[:,i]
          val_s += BLAS::dot(cols, A[i], 1, B_col.data(), 1);
        }

#pragma omp atomic
        results[w] += val_s;
      }
    }
  }
}
*/
void TWFFastDerivWrapper::getGSMatrices(const std::vector<ValueMatrix>& A, std::vector<ValueMatrix>& Aslice) const
{
  IndexType nspecies = A.size();
  Aslice.resize(nspecies);
  for (IndexType id = 0; id < nspecies; id++)
  {
    IndexType ptclnum = A[id].rows();
    Aslice[id].resize(ptclnum, ptclnum);
    for (IndexType i = 0; i < ptclnum; i++)
      for (IndexType j = 0; j < ptclnum; j++)
        Aslice[id][i][j] = A[id][i][j];
  }
}
void TWFFastDerivWrapper::mw_getGSMatrices(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                           const std::vector<std::vector<ValueMatrix>>& A_list,
                                           std::vector<std::vector<ValueMatrix>>& Aslice_list)

{
  const int nw = wf_list.size();
  if (nw == 0)
    return;
Aslice_list.resize(nw);

  // Process each walker in parallel
  for (int w = 0; w < nw; ++w)
  {
    // Get reference to this walker's data
    auto& wf = wf_list[w];
    
    // Call serial implementation for this walker
    wf.getGSMatrices(A_list[w],       // Input matrices for this walker
                     Aslice_list[w]  // Output slice for this walker
                    );
  }/*
  // Find the maximum number of species across all walkers
  IndexType max_nspecies = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    max_nspecies = std::max(max_nspecies, static_cast<IndexType>(A_list[iw].size()));
  }

  // Resize Aslice_list for all walkers
  for (int iw = 0; iw < nw; ++iw)
  {
    Aslice_list[iw].resize(A_list[iw].size());
  }

  // Process all walkers by species
  for (IndexType id = 0; id < max_nspecies; ++id)
  {
    // Process each walker for this species
    for (int iw = 0; iw < nw; ++iw)
    {
      // Skip if this walker doesn't have this species
      if (id >= A_list[iw].size())
        continue;

      // Get the input matrix for this walker and species
      const ValueMatrix& A_matrix = A_list[iw][id];
      IndexType ptclnum           = A_matrix.rows();

      // Resize the output matrix
      Aslice_list[iw][id].resize(ptclnum, ptclnum);

      // Copy the matrix elements
      for (IndexType i = 0; i < ptclnum; i++)
      {
        for (IndexType j = 0; j < ptclnum; j++)
        {
          Aslice_list[iw][id][i][j] = A_matrix[i][j];
        }
      }
    }
  }*/
}

TWFFastDerivWrapper::IndexType TWFFastDerivWrapper::getRowM(const ParticleSet& P,
                                                            const IndexType iel,
                                                            ValueVector& val) const
{
  IndexType gid = P.getGroupID(iel);
  IndexType sid = getTWFGroupIndex(gid);

  GradVector tempg;
  ValueVector templ;

  IndexType norbs = spos_[sid]->getOrbitalSetSize();

  tempg.resize(norbs);
  templ.resize(norbs);

  spos_[sid]->evaluateVGL(P, iel, val, tempg, templ);

  return sid;
}
/*
void TWFFastDerivWrapper::mw_buildX(const RefVectorWithLeader<TWFFastDerivWrapper>& psi_wrapper_list,
                                    const std::vector<std::vector<ValueMatrix>>& Minv,
                                    const std::vector<std::vector<ValueMatrix>>& B_gs,
                                    std::vector<std::vector<ValueMatrix>>& X)
{
  const int nw = psi_wrapper_list.size();
  if (nw == 0)
    return;

  // Find the maximum number of species across all walkers
  IndexType max_nspecies = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    max_nspecies = std::max(max_nspecies, static_cast<IndexType>(Minv[iw].size()));
  }

  // Process all walkers by species
  for (IndexType id = 0; id < max_nspecies; ++id)
  {
    // Process each walker for this species
    for (int iw = 0; iw < nw; ++iw)
    {
      // Skip if this walker doesn't have this species
      if (id >= Minv[iw].size() || id >= B_gs[iw].size())
        continue;

      // Get the input matrices for this walker and species
      const ValueMatrix& Minv_matrix = Minv[iw][id];
      const ValueMatrix& B_matrix    = B_gs[iw][id];
      IndexType ptclnum              = Minv_matrix.rows();

      // Skip if matrices don't match
      if (Minv_matrix.rows() != Minv_matrix.cols() || B_matrix.rows() != ptclnum || B_matrix.cols() != ptclnum)
        continue;

      // Temp matrix for intermediate result
      ValueMatrix tmpmat;
      tmpmat.resize(ptclnum, ptclnum);

      // Compute (B*A^-1)
      for (int i = 0; i < ptclnum; i++)
        for (int j = 0; j < ptclnum; j++)
          for (int k = 0; k < ptclnum; k++)
          {
            tmpmat[i][j] += B_matrix[i][k] * Minv_matrix[k][j];
          }

      // Compute A^{-1}*B*A^{-1}
      for (int i = 0; i < ptclnum; i++)
        for (int j = 0; j < ptclnum; j++)
          for (int k = 0; k < ptclnum; k++)
          {
            X[iw][id][i][j] += Minv_matrix[i][k] * tmpmat[k][j];
          }
    }
  }
}
*/



TWFFastDerivWrapper::ValueType TWFFastDerivWrapper::computeGSDerivative(const std::vector<ValueMatrix>& Minv,
                                                                        const std::vector<ValueMatrix>& X,
                                                                        const std::vector<ValueMatrix>& dM,
                                                                        const std::vector<ValueMatrix>& dB) const
{
  IndexType nspecies = Minv.size();
  ValueType dval     = 0.0;
  for (int id = 0; id < nspecies; id++)
  {
    int ptclnum       = Minv[id].rows();
    ValueType dval_id = 0.0;
    for (int i = 0; i < ptclnum; i++)
      for (int j = 0; j < ptclnum; j++)
      {
        //Tr[M^{-1} dB - X * dM ]
        dval_id += Minv[id][i][j] * dB[id][j][i] - X[id][i][j] * dM[id][j][i];
      }
    dval += dval_id;
  }
  return dval;
}


void TWFFastDerivWrapper::mw_computeGSDerivative(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                               const std::vector<std::vector<ValueMatrix>>& Minv_list,
                                               const std::vector<std::vector<ValueMatrix>>& X_list,
                                               const std::vector<std::vector<std::vector<ValueMatrix>>>& dM_list,
                                               const std::vector<std::vector<std::vector<ValueMatrix>>>& dB_list,
                                               std::vector<ValueType>& results,
                                               int dim)
{
	/*
  const int nw = wf_list.size();
  if (nw == 0)
    return;
  results.resize(nw, 0.0);

  // Find max number of species across walkers
  int max_species = 0;
  for (int w = 0; w < nw; ++w)
    max_species = std::max(max_species, static_cast<int>(Minv_list[w].size()));

// Process by species first, then walkers - to maximize batching potential
#pragma omp parallel
  {
#pragma omp for
    for (int s = 0; s < max_species; ++s)
    {
      // For each walker that has this species
      for (int w = 0; w < nw; ++w)
      {
        // Skip if walker doesn't have this species or dimension
        if (s >= Minv_list[w].size() || s >= X_list[w].size() || dim >= dM_list[w].size() ||
            s >= dM_list[w][dim].size() || dim >= dB_list[w].size() || s >= dB_list[w][dim].size())
          continue;

        const ValueMatrix& Minv = Minv_list[w][s];
        const ValueMatrix& X    = X_list[w][s];
        const ValueMatrix& dM   = dM_list[w][dim][s];
        const ValueMatrix& dB   = dB_list[w][dim][s];
        int ptclnum = Minv.rows();

        ValueType trace_MinvdB = 0.0;
        ValueType trace_XdM = 0.0;

        // For each row i of Minv and column i of dB^T
        for (int i = 0; i < ptclnum; ++i)
        {
          const ValueType* Minv_row = Minv[i];
          const ValueType* X_row = X[i];

          // Create arrays of transposed column elements
          std::vector<ValueType> dB_col(ptclnum);
          std::vector<ValueType> dM_col(ptclnum);
          for (int j = 0; j < ptclnum; ++j)
          {
            dB_col[j] = dB[j][i];
            dM_col[j] = dM[j][i];
          }

          // Use BLAS dot product for row-column multiplication
          trace_MinvdB += BLAS::dot(ptclnum, Minv_row, 1, dB_col.data(), 1);
          trace_XdM += BLAS::dot(ptclnum, X_row, 1, dM_col.data(), 1);
        }

        // Final result: Tr[M^{-1} dB^T] - Tr[X * dM^T]
        ValueType dval_s = trace_MinvdB - trace_XdM;

        results[w] += dval_s;
      }
    }
  }
  */
	const int nw = wf_list.size();
  if (nw == 0)
    return;

  // Ensure 'results' can hold one ValueType per walker
  results.resize(nw);

  // Per-walker fallback: calls the single-walker version
  for (int iw = 0; iw < nw; ++iw)
  {
    // Single walker references
    auto& wf = wf_list[iw];

    // dM_list[iw][dim] is the dimension-slice of dM for walker iw
    const auto& dM_dim = dM_list[iw][dim];
    const auto& dB_dim = dB_list[iw][dim];

    // Revert to single walker call
    results[iw] = wf.computeGSDerivative(
        Minv_list[iw],
        X_list[iw],
        dM_dim,
        dB_dim);
  }
}



void TWFFastDerivWrapper::mw_wipeMatrices(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                          std::vector<std::vector<ValueMatrix>>& matrices_list)
{
  const int nw = wf_list.size();
  for (int iw = 0; iw < nw; ++iw)
    wf_list[iw].wipeMatrices(matrices_list[iw]);
}
void TWFFastDerivWrapper::mw_wipeDerivMatrices(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                               std::vector<std::vector<std::vector<ValueMatrix>>>& matrices)
{
  const int nw   = wf_list.size();
  const int ndim = matrices[0].size();

  for (int iw = 0; iw < nw; ++iw)
    for (int dim = 0; dim < ndim; ++dim)
      wf_list[iw].wipeMatrices(matrices[iw][dim]);
}


void TWFFastDerivWrapper::mw_evaluateJastrowGradSource(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                                       const RefVectorWithLeader<ParticleSet>& p_list,
                                                       const RefVectorWithLeader<ParticleSet>& ion_list,
                                                       int iat,
                                                       std::vector<ParticleSet::ParticleGradient>& wfgradraw)
{
  const int nw = wf_list.size();
  if (nw == 0)
    return;

//#pragma omp parallel for
  for (int w = 0; w < nw; w++)
  {
    GradType grad_iat{};
    const auto& jastrows = wf_list[w].jastrow_list_;
    for (auto* jastrow_ptr : jastrows)
      grad_iat += jastrow_ptr->evalGradSource(p_list[w], ion_list[w], iat);
    wfgradraw[w][iat] = grad_iat;
  }
}
/*
void TWFFastDerivWrapper::mw_evaluateJastrowGradSource(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                                     const RefVectorWithLeader<ParticleSet>& p_list,
                                                     const RefVectorWithLeader<ParticleSet>& ion_list,
                                                     int iat,
                                                     std::vector<ParticleSet::ParticleGradient>& wfgradraw)
{
  const int nw = wf_list.size();
  if (nw == 0)
    return;

  // Get the leader to access types and structures
  const auto& wf_leader = wf_list.getLeader();
  
  // For each Jastrow type in the leader's list
  for (size_t j = 0; j < wf_leader.jastrow_list_.size(); j++) {
    // Create a reference vector for this specific Jastrow type
    auto* jastrow_leader_ptr = wf_leader.jastrow_list_[j];
    
    // Create a RefVector containing the same Jastrow type from all walkers
    RefVectorWithLeader<WaveFunctionComponent> jastrow_ref_list(*jastrow_leader_ptr);
    
    // Add Jastrow components from other walkers
    for (int w = 0; w < nw; w++) {
      jastrow_ref_list.push_back(*wf_list[w].jastrow_list_[j]);
    }
    
    // Call the batched evalGradSource for this Jastrow type
    jastrow_leader_ptr->mw_evalGradSource(jastrow_ref_list, p_list, ion_list, iat, wfgradraw);
  }
}
*/

void TWFFastDerivWrapper::mw_getIonGradM(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                         const RefVectorWithLeader<ParticleSet>& P_list,
                                         const RefVectorWithLeader<ParticleSet>& source_list,
                                         int iat,
                                         std::vector<std::vector<std::vector<ValueMatrix>>>& dmvec_list)
{
  const int nw = wf_list.size();
  for (int iw = 0; iw < nw; ++iw)
    wf_list[iw].getIonGradM(P_list[iw], source_list[iw], iat, dmvec_list[iw]);
}



void TWFFastDerivWrapper::mw_getIonGradM_batch(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                        const RefVectorWithLeader<ParticleSet>& P_list,
                                        const RefVectorWithLeader<ParticleSet>& source_list,
                                        const std::vector<int>& iat_list,
                                        std::vector<std::vector<std::vector<ValueMatrix>>>& dmvec_list)
{   
/*  auto& leader = wf_list.getLeader();
  const int nw = wf_list.size();
  
  // Ensure we have an ion index for each walker
  assert(iat_list.size() == nw);
  
  const IndexType ngroups = leader.spos_.size();
  
  // Initialize dmvec_list for all walkers
  dmvec_list.resize(nw);
  for (int iw = 0; iw < nw; ++iw) {
    dmvec_list[iw].resize(OHMMS_DIM); // x,y,z dimensions
    for (int idim = 0; idim < OHMMS_DIM; ++idim) {
      dmvec_list[iw][idim].resize(ngroups);
    }
  }
  
  // Process each group
  for (IndexType i = 0; i < ngroups; ++i) {
    const IndexType gid = leader.groups_[i];
   
    // Get dimensions from first walker (assuming consistent across walkers)
    const IndexType first = P_list[0].first(i);
    const IndexType last = P_list[0].last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs = leader.spos_[i]->getOrbitalSetSize();
    
    // Prepare batched inputs
    RefVectorWithLeader<SPOSetT<ValueType>> spo_list(*leader.spos_[i]);
    RefVector<GradMatrix> gradphi_list;
    
    // Create temporary storage for gradients
    std::vector<GradMatrix> tmp_grad(nw, GradMatrix(nptcls, norbs));
    
    for (int iw = 0; iw < nw; ++iw) {
      // Cast to concrete SPOSet type
      auto& wf = static_cast<const TWFFastDerivWrapper&>(wf_list[iw]);
      spo_list.push_back(*wf.spos_[i]);
      
      // Initialize temporary gradient matrix
      tmp_grad[iw].resize(nptcls, norbs);
      gradphi_list.push_back(tmp_grad[iw]);
    } 
   
    // Batched gradient evaluation with per-walker ion indices
    //leader.spos_[i]->mw_evaluateGradSource_batch(spo_list, P_list, first, last,
    //                                      source_list, iat_list, gradphi_list);
    
    // Accumulate results into dmvec_list
    for (int iw = 0; iw < nw; ++iw) {
      auto& grad_phi = gradphi_list[iw].get();
      for (IndexType idim = 0; idim < OHMMS_DIM; ++idim) {
        dmvec_list[iw][idim][i].resize(nptcls, norbs);
        for (IndexType iptcl = 0; iptcl < nptcls; ++iptcl) {
          for (IndexType iorb = 0; iorb < norbs; ++iorb) {
            dmvec_list[iw][idim][i][iptcl][iorb] += grad_phi[iptcl][iorb][idim];
          }
        }
      }
    }
  }
  */
	std::cout<<"mw_getIonGradM_batch"<<std::endl;
	exit(0);
}

/*
void TWFFastDerivWrapper::mw_getIonGradM(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                        const RefVectorWithLeader<ParticleSet>& P_list,
                                        const RefVectorWithLeader<ParticleSet>& source_list,
                                        int iat,
                                        std::vector<std::vector<std::vector<ValueMatrix>>>& dmvec_list)
{
  auto& leader = wf_list.getLeader();
  const int nw = wf_list.size();
  const IndexType ngroups = leader.spos_.size();

  // Initialize dmvec_list for all walkers
  dmvec_list.resize(nw);
  for (int iw = 0; iw < nw; ++iw) {
    dmvec_list[iw].resize(OHMMS_DIM); // x,y,z dimensions
    for (int idim = 0; idim < OHMMS_DIM; ++idim) {
      dmvec_list[iw][idim].resize(ngroups);
    }
  }

  // Process each group
  for (IndexType i = 0; i < ngroups; ++i) {
    const IndexType gid = leader.groups_[i];
    
    // Get dimensions from first walker (assuming consistent across walkers)
    const IndexType first = P_list[0].first(i);
    const IndexType last = P_list[0].last(i);
    const IndexType nptcls = last - first;
    const IndexType norbs = leader.spos_[i]->getOrbitalSetSize();

    // Prepare batched inputs
    RefVectorWithLeader<SPOSetT<ValueType>> spo_list(*leader.spos_[i]);
    RefVector<GradMatrix> gradphi_list;

    // Create temporary storage for gradients
    std::vector<GradMatrix> tmp_grad(nw, GradMatrix(nptcls, norbs));

    for (int iw = 0; iw < nw; ++iw) {
      // Cast to concrete SPOSet type
      auto& wf = static_cast<const TWFFastDerivWrapper&>(wf_list[iw]);
      spo_list.push_back(*wf.spos_[i]);
      
      // Initialize temporary gradient matrix
      tmp_grad[iw].resize(nptcls, norbs);
      gradphi_list.push_back(tmp_grad[iw]);
    }

    // Batched gradient evaluation
    leader.spos_[i]->mw_evaluateGradSource(spo_list, P_list, first, last,
                                          source_list, iat, gradphi_list);

    // Accumulate results into dmvec_list
    for (int iw = 0; iw < nw; ++iw) {
      auto& grad_phi = gradphi_list[iw].get();
      for (IndexType idim = 0; idim < OHMMS_DIM; ++idim) {
        dmvec_list[iw][idim][i].resize(nptcls, norbs);
        for (IndexType iptcl = 0; iptcl < nptcls; ++iptcl) {
          for (IndexType iorb = 0; iorb < norbs; ++iorb) {
            dmvec_list[iw][idim][i][iptcl][iorb] += grad_phi[iptcl][iorb][idim];
          }
        }
      }
    }
  }
}
*/
void TWFFastDerivWrapper::mw_getIonGradIonGradELaplM(const RefVectorWithLeader<TWFFastDerivWrapper>& wf_list,
                                                     const RefVectorWithLeader<ParticleSet>& P_list,
                                                     const RefVectorWithLeader<ParticleSet>& source_list,
                                                     int iat,
                                                     std::vector<std::vector<std::vector<ValueMatrix>>>& dmvec_list,
                                                     std::vector<std::vector<std::vector<GradMatrix>>>& dgmat_list,
                                                     std::vector<std::vector<std::vector<ValueMatrix>>>& dlmat_list)
{
  const int nw = wf_list.size();
  for (int iw = 0; iw < nw; ++iw)
  {
    wf_list[iw].getIonGradIonGradELaplM(P_list[iw], source_list[iw], iat, dmvec_list[iw], dgmat_list[iw],
                                        dlmat_list[iw]);
  }
}

} // namespace qmcplusplus
