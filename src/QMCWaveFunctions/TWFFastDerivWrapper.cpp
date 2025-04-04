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
void TWFFastDerivWrapper::mw_buildX(const RefVectorWithLeader<TWFFastDerivWrapper>& psi_wrapper_list,
                                   const std::vector<std::vector<ValueMatrix>>& Minv,
                                   const std::vector<std::vector<ValueMatrix>>& B_gs,
                                   std::vector<std::vector<ValueMatrix>>& X)

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
