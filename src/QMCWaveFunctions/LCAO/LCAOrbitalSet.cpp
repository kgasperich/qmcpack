//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////


#include "LCAOrbitalSet.h"
#include "Numerics/MatrixOperators.h"
#include "CPU/BLAS.hpp"
#include "OMPTarget/ompBLAS.hpp"
#include <ResourceCollection.h>
#include <numeric>

namespace qmcplusplus
{

struct LCAOrbitalSet::LCAOMultiWalkerMem : public Resource
{
  LCAOMultiWalkerMem() : Resource("LCAOrbitalSet") {}
  LCAOMultiWalkerMem(const LCAOMultiWalkerMem&) : LCAOMultiWalkerMem() {}

  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<LCAOMultiWalkerMem>(*this); }

  OffloadMWVGLArray phi_vgl_v;                           // [5][NW][NumMO]
  OffloadMWVGLArray basis_vgl_mw;                        // [5][NW][NumAO]
  OffloadMWVArray phi_v;                                 // [NW][NumMO]
  OffloadMWVArray basis_v_mw;                            // [NW][NumAO]
  OffloadMWVArray vp_phi_v;                              // [NVPs][NumMO]
  OffloadMWVArray vp_basis_v_mw;                         // [NVPs][NumAO]
  OffloadVector<const ValueType*> invRow_deviceptr_list; // [NVPs]
  OffloadVector<ValueType> ratios_buffer;                // [NVPs]
  OffloadVector<size_t> nVP_index_list;                  // [NVPs]
};

LCAOrbitalSet::LCAOrbitalSet(const std::string& my_name, std::unique_ptr<basis_type>&& bs, size_t norbs, bool identity)
    : SPOSet(my_name),
      BasisSetSize(bs ? bs->getBasisSetSize() : 0),
      Identity(identity),
      basis_timer_(createGlobalTimer("LCAOrbitalSet::Basis", timer_level_fine)),
      mo_timer_(createGlobalTimer("LCAOrbitalSet::MO", timer_level_fine))
{
  if (!bs)
    throw std::runtime_error("LCAOrbitalSet cannot take nullptr as its  basis set!");
  myBasisSet     = std::move(bs);
  OrbitalSetSize = norbs;
  Temp.resize(BasisSetSize);
  Temph.resize(BasisSetSize);
  Tempgh.resize(BasisSetSize);
  OrbitalSetSize = norbs;
  if (!Identity)
  {
    Tempv.resize(OrbitalSetSize);
    Temphv.resize(OrbitalSetSize);
    Tempghv.resize(OrbitalSetSize);
    C = std::make_shared<OffloadValueMatrix>(OrbitalSetSize, BasisSetSize);
  }
  LCAOrbitalSet::checkObject();
}

LCAOrbitalSet::LCAOrbitalSet(const LCAOrbitalSet& in)
    : SPOSet(in),
      myBasisSet(in.myBasisSet->makeClone()),
      C(in.C),
      BasisSetSize(in.BasisSetSize),
      C_copy(in.C_copy),
      Identity(in.Identity),
      basis_timer_(in.basis_timer_),
      mo_timer_(in.mo_timer_)
{
  Temp.resize(BasisSetSize);
  Temph.resize(BasisSetSize);
  Tempgh.resize(BasisSetSize);
  if (!in.Identity)
  {
    Tempv.resize(OrbitalSetSize);
    Temphv.resize(OrbitalSetSize);
    Tempghv.resize(OrbitalSetSize);
  }
  LCAOrbitalSet::checkObject();
}

void LCAOrbitalSet::setOrbitalSetSize(int norbs)
{
  throw std::runtime_error("LCAOrbitalSet::setOrbitalSetSize should not be called");
}

void LCAOrbitalSet::checkObject() const
{
  if (Identity)
  {
    if (OrbitalSetSize != BasisSetSize)
      throw std::runtime_error(
          "LCAOrbitalSet::checkObject OrbitalSetSize and BasisSetSize must be equal if Identity = true!");
    if (C)
      throw std::runtime_error("LCAOrbitalSet::checkObject C should be nullptr if Identity = true!");
  }
  else
  {
    if (!C)
      throw std::runtime_error("LCAOrbitalSet::checkObject C should not be nullptr if Identity = false!");
    if (OrbitalSetSize != C->rows())
      throw std::runtime_error("LCAOrbitalSet::checkObject C rows doesn't match OrbitalSetSize.");
    if (BasisSetSize != C->cols())
      throw std::runtime_error("LCAOrbitalSet::checkObject C columns doesn't match BasisSetSize.");
  }
}

void LCAOrbitalSet::createResource(ResourceCollection& collection) const
{
  myBasisSet->createResource(collection);

  auto resource_index = collection.addResource(std::make_unique<LCAOMultiWalkerMem>());
}

void LCAOrbitalSet::acquireResource(ResourceCollection& collection, const RefVectorWithLeader<SPOSet>& spo_list) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();

  spo_leader.myBasisSet->acquireResource(collection, extractBasisRefList(spo_list));

  spo_leader.mw_mem_handle_ = collection.lendResource<LCAOMultiWalkerMem>();
}

void LCAOrbitalSet::releaseResource(ResourceCollection& collection, const RefVectorWithLeader<SPOSet>& spo_list) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();

  spo_leader.myBasisSet->releaseResource(collection, extractBasisRefList(spo_list));

  collection.takebackResource(spo_leader.mw_mem_handle_);
}

RefVectorWithLeader<typename LCAOrbitalSet::basis_type> LCAOrbitalSet::extractBasisRefList(
    const RefVectorWithLeader<SPOSet>& spo_list) const
{
  RefVectorWithLeader<basis_type> basis_list(*spo_list.getCastedLeader<LCAOrbitalSet>().myBasisSet);
  basis_list.reserve(spo_list.size());
  for (size_t iw = 0; iw < spo_list.size(); iw++)
    basis_list.push_back(*spo_list.getCastedElement<LCAOrbitalSet>(iw).myBasisSet);
  return basis_list;
}
std::unique_ptr<SPOSet> LCAOrbitalSet::makeClone() const { return std::make_unique<LCAOrbitalSet>(*this); }

void LCAOrbitalSet::evaluateValue(const ParticleSet& P, int iat, ValueVector& psi)
{
  if (Identity)
  { //PAY ATTENTION TO COMPLEX
    myBasisSet->evaluateV(P, iat, psi.data());
  }
  else
  {
    Vector<ValueType> vTemp(Temp.data(0), BasisSetSize);
    myBasisSet->evaluateV(P, iat, vTemp.data());
    assert(psi.size() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), psi.size(), BasisSetSize);
    MatrixOperators::product(C_partial_view, vTemp, psi);
  }
}

/** Find a better place for other user classes, Matrix should be padded as well */
template<typename T, unsigned D, typename Alloc>
inline void Product_ABt(const VectorSoaContainer<T, D>& A, const Matrix<T, Alloc>& B, VectorSoaContainer<T, D>& C)
{
  constexpr char transa = 't';
  constexpr char transb = 'n';
  constexpr T zone(1);
  constexpr T zero(0);
  BLAS::gemm(transa, transb, B.rows(), D, B.cols(), zone, B.data(), B.cols(), A.data(), A.capacity(), zero, C.data(),
             C.capacity());
}

inline void LCAOrbitalSet::evaluate_vgl_impl(const vgl_type& temp,
                                             ValueVector& psi,
                                             GradVector& dpsi,
                                             ValueVector& d2psi) const
{
  const size_t output_size = psi.size();
  std::copy_n(temp.data(0), output_size, psi.data());
  const ValueType* restrict gx = temp.data(1);
  const ValueType* restrict gy = temp.data(2);
  const ValueType* restrict gz = temp.data(3);
  for (size_t j = 0; j < output_size; j++)
  {
    dpsi[j][0] = gx[j];
    dpsi[j][1] = gy[j];
    dpsi[j][2] = gz[j];
  }
  std::copy_n(temp.data(4), output_size, d2psi.data());
}

inline void LCAOrbitalSet::evaluate_vgh_impl(const vgh_type& temp,
                                             ValueVector& psi,
                                             GradVector& dpsi,
                                             HessVector& d2psi) const
{
  const size_t output_size = psi.size();
  std::copy_n(temp.data(0), output_size, psi.data());
  const ValueType* restrict gx  = temp.data(1);
  const ValueType* restrict gy  = temp.data(2);
  const ValueType* restrict gz  = temp.data(3);
  const ValueType* restrict hxx = temp.data(4);
  const ValueType* restrict hxy = temp.data(5);
  const ValueType* restrict hxz = temp.data(6);
  const ValueType* restrict hyy = temp.data(7);
  const ValueType* restrict hyz = temp.data(8);
  const ValueType* restrict hzz = temp.data(9);

  for (size_t j = 0; j < output_size; j++)
  {
    dpsi[j][0] = gx[j];
    dpsi[j][1] = gy[j];
    dpsi[j][2] = gz[j];

    d2psi[j](0, 0) = hxx[j];
    d2psi[j](0, 1) = d2psi[j](1, 0) = hxy[j];
    d2psi[j](0, 2) = d2psi[j](2, 0) = hxz[j];
    d2psi[j](1, 1)                  = hyy[j];
    d2psi[j](2, 1) = d2psi[j](1, 2) = hyz[j];
    d2psi[j](2, 2)                  = hzz[j];
  }
}

inline void LCAOrbitalSet::evaluate_vghgh_impl(const vghgh_type& temp,
                                               int i,
                                               ValueMatrix& psi,
                                               GradMatrix& dpsi,
                                               HessMatrix& d2psi,
                                               GGGMatrix& dghpsi) const
{
  const size_t output_size = psi.cols();
  std::copy_n(temp.data(0), output_size, psi[i]);
  const ValueType* restrict gx     = temp.data(1);
  const ValueType* restrict gy     = temp.data(2);
  const ValueType* restrict gz     = temp.data(3);
  const ValueType* restrict hxx    = temp.data(4);
  const ValueType* restrict hxy    = temp.data(5);
  const ValueType* restrict hxz    = temp.data(6);
  const ValueType* restrict hyy    = temp.data(7);
  const ValueType* restrict hyz    = temp.data(8);
  const ValueType* restrict hzz    = temp.data(9);
  const ValueType* restrict gh_xxx = temp.data(10);
  const ValueType* restrict gh_xxy = temp.data(11);
  const ValueType* restrict gh_xxz = temp.data(12);
  const ValueType* restrict gh_xyy = temp.data(13);
  const ValueType* restrict gh_xyz = temp.data(14);
  const ValueType* restrict gh_xzz = temp.data(15);
  const ValueType* restrict gh_yyy = temp.data(16);
  const ValueType* restrict gh_yyz = temp.data(17);
  const ValueType* restrict gh_yzz = temp.data(18);
  const ValueType* restrict gh_zzz = temp.data(19);

  for (size_t j = 0; j < output_size; j++)
  {
    dpsi[i][j][0] = gx[j];
    dpsi[i][j][1] = gy[j];
    dpsi[i][j][2] = gz[j];

    d2psi[i][j](0, 0) = hxx[j];
    d2psi[i][j](0, 1) = d2psi[i][j](1, 0) = hxy[j];
    d2psi[i][j](0, 2) = d2psi[i][j](2, 0) = hxz[j];
    d2psi[i][j](1, 1)                     = hyy[j];
    d2psi[i][j](2, 1) = d2psi[i][j](1, 2) = hyz[j];
    d2psi[i][j](2, 2)                     = hzz[j];

    dghpsi[i][j][0](0, 0) = gh_xxx[j]; //x|xx
    dghpsi[i][j][0](0, 1) = gh_xxy[j]; //x|xy
    dghpsi[i][j][0](0, 2) = gh_xxz[j]; //x|xz
    dghpsi[i][j][0](1, 0) = gh_xxy[j]; //x|yx = xxy
    dghpsi[i][j][0](1, 1) = gh_xyy[j]; //x|yy
    dghpsi[i][j][0](1, 2) = gh_xyz[j]; //x|yz
    dghpsi[i][j][0](2, 0) = gh_xxz[j]; //x|zx = xxz
    dghpsi[i][j][0](2, 1) = gh_xyz[j]; //x|zy = xyz
    dghpsi[i][j][0](2, 2) = gh_xzz[j]; //x|zz

    dghpsi[i][j][1](0, 0) = gh_xxy[j]; //y|xx = xxy
    dghpsi[i][j][1](0, 1) = gh_xyy[j]; //y|xy = xyy
    dghpsi[i][j][1](0, 2) = gh_xyz[j]; //y|xz = xyz
    dghpsi[i][j][1](1, 0) = gh_xyy[j]; //y|yx = xyy
    dghpsi[i][j][1](1, 1) = gh_yyy[j]; //y|yy
    dghpsi[i][j][1](1, 2) = gh_yyz[j]; //y|yz
    dghpsi[i][j][1](2, 0) = gh_xyz[j]; //y|zx = xyz
    dghpsi[i][j][1](2, 1) = gh_yyz[j]; //y|zy = yyz
    dghpsi[i][j][1](2, 2) = gh_yzz[j]; //y|zz

    dghpsi[i][j][2](0, 0) = gh_xxz[j]; //z|xx = xxz
    dghpsi[i][j][2](0, 1) = gh_xyz[j]; //z|xy = xyz
    dghpsi[i][j][2](0, 2) = gh_xzz[j]; //z|xz = xzz
    dghpsi[i][j][2](1, 0) = gh_xyz[j]; //z|yx = xyz
    dghpsi[i][j][2](1, 1) = gh_yyz[j]; //z|yy = yyz
    dghpsi[i][j][2](1, 2) = gh_yzz[j]; //z|yz = yzz
    dghpsi[i][j][2](2, 0) = gh_xzz[j]; //z|zx = xzz
    dghpsi[i][j][2](2, 1) = gh_yzz[j]; //z|zy = yzz
    dghpsi[i][j][2](2, 2) = gh_zzz[j]; //z|zz
  }
}

inline void LCAOrbitalSet::evaluate_vghgh_impl(const vghgh_type& temp,
                                               ValueVector& psi,
                                               GradVector& dpsi,
                                               HessVector& d2psi,
                                               GGGVector& dghpsi) const
{
  const size_t output_size = psi.size();
  std::copy_n(temp.data(0), output_size, psi.data());
  const ValueType* restrict gx     = temp.data(1);
  const ValueType* restrict gy     = temp.data(2);
  const ValueType* restrict gz     = temp.data(3);
  const ValueType* restrict hxx    = temp.data(4);
  const ValueType* restrict hxy    = temp.data(5);
  const ValueType* restrict hxz    = temp.data(6);
  const ValueType* restrict hyy    = temp.data(7);
  const ValueType* restrict hyz    = temp.data(8);
  const ValueType* restrict hzz    = temp.data(9);
  const ValueType* restrict gh_xxx = temp.data(10);
  const ValueType* restrict gh_xxy = temp.data(11);
  const ValueType* restrict gh_xxz = temp.data(12);
  const ValueType* restrict gh_xyy = temp.data(13);
  const ValueType* restrict gh_xyz = temp.data(14);
  const ValueType* restrict gh_xzz = temp.data(15);
  const ValueType* restrict gh_yyy = temp.data(16);
  const ValueType* restrict gh_yyz = temp.data(17);
  const ValueType* restrict gh_yzz = temp.data(18);
  const ValueType* restrict gh_zzz = temp.data(19);

  for (size_t j = 0; j < output_size; j++)
  {
    dpsi[j][0] = gx[j];
    dpsi[j][1] = gy[j];
    dpsi[j][2] = gz[j];

    d2psi[j](0, 0) = hxx[j];
    d2psi[j](0, 1) = d2psi[j](1, 0) = hxy[j];
    d2psi[j](0, 2) = d2psi[j](2, 0) = hxz[j];
    d2psi[j](1, 1)                  = hyy[j];
    d2psi[j](2, 1) = d2psi[j](1, 2) = hyz[j];
    d2psi[j](2, 2)                  = hzz[j];

    dghpsi[j][0](0, 0) = gh_xxx[j]; //x|xx
    dghpsi[j][0](0, 1) = gh_xxy[j]; //x|xy
    dghpsi[j][0](0, 2) = gh_xxz[j]; //x|xz
    dghpsi[j][0](1, 0) = gh_xxy[j]; //x|yx = xxy
    dghpsi[j][0](1, 1) = gh_xyy[j]; //x|yy
    dghpsi[j][0](1, 2) = gh_xyz[j]; //x|yz
    dghpsi[j][0](2, 0) = gh_xxz[j]; //x|zx = xxz
    dghpsi[j][0](2, 1) = gh_xyz[j]; //x|zy = xyz
    dghpsi[j][0](2, 2) = gh_xzz[j]; //x|zz

    dghpsi[j][1](0, 0) = gh_xxy[j]; //y|xx = xxy
    dghpsi[j][1](0, 1) = gh_xyy[j]; //y|xy = xyy
    dghpsi[j][1](0, 2) = gh_xyz[j]; //y|xz = xyz
    dghpsi[j][1](1, 0) = gh_xyy[j]; //y|yx = xyy
    dghpsi[j][1](1, 1) = gh_yyy[j]; //y|yy
    dghpsi[j][1](1, 2) = gh_yyz[j]; //y|yz
    dghpsi[j][1](2, 0) = gh_xyz[j]; //y|zx = xyz
    dghpsi[j][1](2, 1) = gh_xyy[j]; //y|xy = xyy
    dghpsi[j][1](2, 2) = gh_yzz[j]; //y|zz

    dghpsi[j][2](0, 0) = gh_xzz[j]; //z|xx = xzz
    dghpsi[j][2](0, 1) = gh_xyz[j]; //z|xy = xyz
    dghpsi[j][2](0, 2) = gh_xzz[j]; //z|xz = xzz
    dghpsi[j][2](1, 0) = gh_xyz[j]; //z|yx = xyz
    dghpsi[j][2](1, 1) = gh_yyz[j]; //z|yy = yyz
    dghpsi[j][2](1, 2) = gh_yzz[j]; //z|yz = yzz
    dghpsi[j][2](2, 0) = gh_xzz[j]; //z|zx = xzz
    dghpsi[j][2](2, 1) = gh_yzz[j]; //z|zy = yzz
    dghpsi[j][2](2, 2) = gh_zzz[j]; //z|zz
  }
}

inline void LCAOrbitalSet::evaluate_ionderiv_v_row_impl(const vgl_type& temp, GradVector& dpsi) const
{
  const size_t output_size     = dpsi.size();
  const ValueType* restrict gx = temp.data(1);
  const ValueType* restrict gy = temp.data(2);
  const ValueType* restrict gz = temp.data(3);

  for (size_t j = 0; j < output_size; j++)
  {
    //As mentioned in SoaLocalizedBasisSet, LCAO's have a nice property that
    // for an atomic center, the ion gradient is the negative of the elecron gradient.
    // Hence minus signs for each of these.
    dpsi[j][0] = -gx[j];
    dpsi[j][1] = -gy[j];
    dpsi[j][2] = -gz[j];
  }
}


void LCAOrbitalSet::evaluateVGL(const ParticleSet& P, int iat, ValueVector& psi, GradVector& dpsi, ValueVector& d2psi)
{
  //TAKE CARE OF IDENTITY
  {
    ScopedTimer local(basis_timer_);
    myBasisSet->evaluateVGL(P, iat, Temp);
  }

  if (Identity)
    evaluate_vgl_impl(Temp, psi, dpsi, d2psi);
  else
  {
    assert(psi.size() <= OrbitalSetSize);
    {
      ScopedTimer local(mo_timer_);
      ValueMatrix C_partial_view(C->data(), psi.size(), BasisSetSize);
      Product_ABt(Temp, C_partial_view, Tempv);
    }
    evaluate_vgl_impl(Tempv, psi, dpsi, d2psi);
  }
}

void LCAOrbitalSet::mw_evaluateVGL(const RefVectorWithLeader<SPOSet>& spo_list,
                                   const RefVectorWithLeader<ParticleSet>& P_list,
                                   int iat,
                                   const RefVector<ValueVector>& psi_v_list,
                                   const RefVector<GradVector>& dpsi_v_list,
                                   const RefVector<ValueVector>& d2psi_v_list) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();
  auto& phi_vgl_v  = spo_leader.mw_mem_handle_.getResource().phi_vgl_v;

  phi_vgl_v.resize(DIM_VGL, spo_list.size(), OrbitalSetSize);
  mw_evaluateVGLImplGEMM(spo_list, P_list, iat, phi_vgl_v);

  const size_t nw = phi_vgl_v.size(1);
  phi_vgl_v.updateFrom();

  //TODO: make this cleaner?
  for (int iw = 0; iw < nw; iw++)
  {
    const size_t output_size = psi_v_list[iw].get().size();
    std::copy_n(phi_vgl_v.data_at(0, iw, 0), output_size, psi_v_list[iw].get().data());
    std::copy_n(phi_vgl_v.data_at(4, iw, 0), output_size, d2psi_v_list[iw].get().data());
    // grads are [dim, walker, orb] in phi_vgl_v
    //           [walker][orb, dim] in dpsi_v_list
    for (size_t idim = 0; idim < DIM; idim++)
      BLAS::copy(output_size, phi_vgl_v.data_at(idim + 1, iw, 0), 1, &dpsi_v_list[iw].get().data()[0][idim], DIM);
  }
}

void LCAOrbitalSet::mw_evaluateVGLImplGEMM(const RefVectorWithLeader<SPOSet>& spo_list,
                                           const RefVectorWithLeader<ParticleSet>& P_list,
                                           int iat,
                                           OffloadMWVGLArray& phi_vgl_v) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader   = spo_list.getCastedLeader<LCAOrbitalSet>();
  auto& basis_vgl_mw = spo_leader.mw_mem_handle_.getResource().basis_vgl_mw;
  basis_vgl_mw.resize(DIM_VGL, spo_list.size(), BasisSetSize);

  {
    ScopedTimer local(basis_timer_);
    auto basis_list = spo_leader.extractBasisRefList(spo_list);
    myBasisSet->mw_evaluateVGL(basis_list, P_list, iat, basis_vgl_mw);
  }
  // basis_vgl_mw correct on device
#if defined(ENABLE_OFFLOAD)
  int dummy_handle = 0;
  int success      = 0;

  if (Identity)
  {
    const size_t output_size = phi_vgl_v.size(2);
    const size_t nw          = phi_vgl_v.size(1);
    for (size_t idim = 0; idim < DIM_VGL; idim++)
    {
      success = ompBLAS::copy(dummy_handle, output_size * nw, basis_vgl_mw.device_data_at(idim, 0, 0), 1,
                              phi_vgl_v.device_data_at(idim, 0, 0), 1);
      if (success != 0)
        throw std::runtime_error("In LCAOrbitalSet::mw_evaluateVGLImplGEMM ompBLAS::copy failed.");
    }
  }
  else
  {
    const size_t requested_orb_size = phi_vgl_v.size(2);
    assert(requested_orb_size <= OrbitalSetSize);

    auto* c_devptr = C->device_data();
    success        = ompBLAS::gemm(dummy_handle, 'T', 'N',
                                   requested_orb_size,        // MOs
                                   spo_list.size() * DIM_VGL, // walkers * DIM_VGL
                                   BasisSetSize,              // AOs
                                   1, c_devptr, BasisSetSize, basis_vgl_mw.device_data(), BasisSetSize, 0,
                                   phi_vgl_v.device_data(), requested_orb_size);
    if (success != 0)
      throw std::runtime_error("In LCAOrbitalSet::mw_evaluateVGLImplGEMM ompBLAS::gemm failed.");
  }
#else
  if (Identity)
  {
    // output_size can be smaller than BasisSetSize
    const size_t output_size = phi_vgl_v.size(2);
    const size_t nw          = phi_vgl_v.size(1);

    for (size_t idim = 0; idim < DIM_VGL; idim++)
      for (int iw = 0; iw < nw; iw++)
        std::copy_n(basis_vgl_mw.data_at(idim, iw, 0), output_size, phi_vgl_v.data_at(idim, iw, 0));
  }
  else
  {
    const size_t requested_orb_size = phi_vgl_v.size(2);
    assert(requested_orb_size <= OrbitalSetSize);
    {
      ScopedTimer local(mo_timer_);
      ValueMatrix C_partial_view(C->data(), requested_orb_size, BasisSetSize);
      // TODO: make class for general blas interface in Platforms
      // have instance of that class as member of LCAOrbitalSet, call gemm through that
      BLAS::gemm('T', 'N',
                 requested_orb_size,        // MOs
                 spo_list.size() * DIM_VGL, // walkers * DIM_VGL
                 BasisSetSize,              // AOs
                 1, C_partial_view.data(), BasisSetSize, basis_vgl_mw.data(), BasisSetSize, 0, phi_vgl_v.data(),
                 requested_orb_size);
    }
  }
#endif
  // phi_vgl_v correct on device if ENABLE_OFFLOAD
}

void LCAOrbitalSet::mw_evaluateValueVPsImplGEMM(const RefVectorWithLeader<SPOSet>& spo_list,
                                                const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                                OffloadMWVArray& vp_phi_v) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();
  //const size_t nw  = spo_list.size();
  auto& vp_basis_v_mw = spo_leader.mw_mem_handle_.getResource().vp_basis_v_mw;
  //Splatter basis_v
  const size_t nVPs = vp_phi_v.size(0);
  vp_basis_v_mw.resize(nVPs, BasisSetSize);

  auto basis_list = spo_leader.extractBasisRefList(spo_list);
  myBasisSet->mw_evaluateValueVPs(basis_list, vp_list, vp_basis_v_mw);

#if defined(ENABLE_OFFLOAD)

  int dummy_handle = 0;
  int success      = 0;

  if (Identity)
  {
    success =
        ompBLAS::copy(dummy_handle, OrbitalSetSize * nVPs, vp_basis_v_mw.device_data(), 1, vp_phi_v.device_data(), 1);
    if (success != 0)
      throw std::runtime_error("In LCAOrbitalSet::mw_evaluateValueVPsImplGEMM ompBLAS::copy failed.");
  }
  else
  {
    const size_t requested_orb_size = vp_phi_v.size(1);
    assert(requested_orb_size <= OrbitalSetSize);

    auto* c_devptr = C->device_data();
    success        = ompBLAS::gemm(dummy_handle, 'T', 'N',
                                   requested_orb_size, // MOs
                                   nVPs,               // walkers * Virtual Particles
                                   BasisSetSize,       // AOs
                                   1, c_devptr, BasisSetSize, vp_basis_v_mw.device_data(), BasisSetSize, 0,
                                   vp_phi_v.device_data(), requested_orb_size);
    if (success != 0)
      throw std::runtime_error("In LCAOrbitalSet::mw_evaluateValueVPsImplGEMM ompBLAS::gemm failed.");
  }
#else
  if (Identity)
  {
    std::copy_n(vp_basis_v_mw.data_at(0, 0), OrbitalSetSize * nVPs, vp_phi_v.data_at(0, 0));
  }
  else
  {
    const size_t requested_orb_size = vp_phi_v.size(1);
    assert(requested_orb_size <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), requested_orb_size, BasisSetSize);
    BLAS::gemm('T', 'N',
               requested_orb_size, // MOs
               nVPs,               // walkers * Virtual Particles
               BasisSetSize,       // AOs
               1, C_partial_view.data(), BasisSetSize, vp_basis_v_mw.data(), BasisSetSize, 0, vp_phi_v.data(),
               requested_orb_size);
  }
#endif
}
void LCAOrbitalSet::mw_evaluateValue(const RefVectorWithLeader<SPOSet>& spo_list,
                                     const RefVectorWithLeader<ParticleSet>& P_list,
                                     int iat,
                                     const RefVector<ValueVector>& psi_v_list) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();
  auto& phi_v      = spo_leader.mw_mem_handle_.getResource().phi_v;
  phi_v.resize(spo_list.size(), OrbitalSetSize);
  mw_evaluateValueImplGEMM(spo_list, P_list, iat, phi_v);

  const size_t output_size = phi_v.size(1);
  const size_t nw          = phi_v.size(0);
  phi_v.updateFrom();

  for (int iw = 0; iw < nw; iw++)
    std::copy_n(phi_v.data_at(iw, 0), output_size, psi_v_list[iw].get().data());
}

void LCAOrbitalSet::mw_evaluateValueImplGEMM(const RefVectorWithLeader<SPOSet>& spo_list,
                                             const RefVectorWithLeader<ParticleSet>& P_list,
                                             int iat,
                                             OffloadMWVArray& phi_v) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();
  const size_t nw  = spo_list.size();
  auto& basis_v_mw = spo_leader.mw_mem_handle_.getResource().basis_v_mw;
  basis_v_mw.resize(nw, BasisSetSize);

  auto basis_list = spo_leader.extractBasisRefList(spo_list);
  myBasisSet->mw_evaluateValue(basis_list, P_list, iat, basis_v_mw);

#if defined(ENABLE_OFFLOAD)
  auto* basis_devptr = basis_v_mw.device_data_at(0, 0);
  auto* phi_devptr   = phi_v.device_data_at(0, 0);
  int dummy_handle   = 0;
  int success        = 0;

  if (Identity)
  {
    success = ompBLAS::copy(dummy_handle, OrbitalSetSize * nw, basis_devptr, 1, phi_devptr, 1);
    if (success != 0)
      throw std::runtime_error("In LCAOrbitalSet::mw_evaluateValueImplGEMM ompBLAS::copy failed.");
  }
  else
  {
    const size_t requested_orb_size = phi_v.size(1);
    assert(requested_orb_size <= OrbitalSetSize);

    auto* c_devptr = C->device_data();
    success        = ompBLAS::gemm(dummy_handle, 'T', 'N',
                                   requested_orb_size, // MOs
                                   nw,                 // walkers
                                   BasisSetSize,       // AOs
                                   1, c_devptr, BasisSetSize, basis_devptr, BasisSetSize, 0, phi_devptr, requested_orb_size);
    if (success != 0)
      throw std::runtime_error("In LCAOrbitalSet::mw_evaluateValueImplGEMM ompBLAS::gemm failed.");
  }
#else
  if (Identity)
  {
    std::copy_n(basis_v_mw.data_at(0, 0), OrbitalSetSize * nw, phi_v.data_at(0, 0));
  }
  else
  {
    const size_t requested_orb_size = phi_v.size(1);
    assert(requested_orb_size <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), requested_orb_size, BasisSetSize);
    BLAS::gemm('T', 'N',
               requested_orb_size, // MOs
               spo_list.size(),    // walkers
               BasisSetSize,       // AOs
               1, C_partial_view.data(), BasisSetSize, basis_v_mw.data(), BasisSetSize, 0, phi_v.data(),
               requested_orb_size);
  }
#endif
}

void LCAOrbitalSet::mw_evaluateDetRatios(const RefVectorWithLeader<SPOSet>& spo_list,
                                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                         const RefVector<ValueVector>& psi_list,
                                         const std::vector<const ValueType*>& invRow_ptr_list,
                                         std::vector<std::vector<ValueType>>& ratios_list) const
{
  assert(this == &spo_list.getLeader());
  auto& spo_leader = spo_list.getCastedLeader<LCAOrbitalSet>();
  auto& vp_phi_v   = spo_leader.mw_mem_handle_.getResource().vp_phi_v;

  const size_t nVPs               = VirtualParticleSet::countVPs(vp_list);
  const size_t requested_orb_size = psi_list[0].get().size();
  vp_phi_v.resize(nVPs, requested_orb_size);

  mw_evaluateValueVPsImplGEMM(spo_list, vp_list, vp_phi_v);

#if defined(ENABLE_OFFLOAD)
  const size_t nw             = vp_list.size();
  auto& invRow_deviceptr_list = spo_leader.mw_mem_handle_.getResource().invRow_deviceptr_list;
  auto& ratios_buffer         = spo_leader.mw_mem_handle_.getResource().ratios_buffer;

  invRow_deviceptr_list.resize(nVPs);
  ratios_buffer.resize(nVPs);

  std::vector<size_t> nVP_mw(nw, 0);  // number of VPs per walker
  std::vector<size_t> iVP0_mw(nw, 0); // first VP idx of each walker

  // fill with nVP per walker
  for (size_t iw = 0; iw < nw; iw++)
    nVP_mw[iw] = vp_list[iw].getTotalNum();

  // fill with cumulative number of VPs up to current walker
  std::exclusive_scan(nVP_mw.begin(), nVP_mw.end(), iVP0_mw.begin(), 0);

  
  for (size_t iw = 0; iw < nw; iw++)
  {
    auto istart = iVP0_mw[iw];
    auto nvp_i  = nVP_mw[iw];
    std::fill(invRow_deviceptr_list.begin() + istart, invRow_deviceptr_list.begin() + istart + nvp_i,
              invRow_ptr_list[iw]);
    //for (size_t idx = 0; idx< nvp_i; idx++)
    //  invRow_deviceptr_list[istart+idx] = invRow_ptr_list[iw];
  }

  std::cout << "\nnVP_mw:                   {";
  for (auto& i : nVP_mw){
    std::cout << i << ", ";
  }
  std::cout << "}\n";
  std::cout << "\ninvRow_ptr_list       {";
  for (auto& i : invRow_ptr_list){
    std::cout << i << ", ";
  }
  std::cout << "}\n";
  std::cout << "\ninvRow_deviceptr_list {";
  for (auto& i : invRow_deviceptr_list){
    std::cout << i << ", ";
  }
  std::cout << "}\n";

  auto* invRow_deviceptr_list_ptr = invRow_deviceptr_list.data();
  auto* vp_phi_v_ptr              = vp_phi_v.data();
  auto* ratios_buffer_ptr         = ratios_buffer.data();
  auto* ratios_buffer_devptr         = ratios_buffer.device_data();
  //ratios_buffer.updateTo();
  //invRow_deviceptr_list.updateTo();
  std::cout << "\nratios_buffer_ptr:    " << ratios_buffer_ptr;
  std::cout << "\nratios_buffer_devptr: " << ratios_buffer_devptr << std::endl;
  PRAGMA_OFFLOAD("omp target teams distribute parallel for \
      map(always,to: invRow_deviceptr_list_ptr[:nVPs]) \
      map(to: vp_phi_v_ptr[:nVPs*requested_orb_size]) \
      map(always,from: ratios_buffer_ptr[:nVPs])")
  for (size_t ivp = 0; ivp < nVPs; ivp++)
  {
    ratios_buffer_ptr[ivp] = 0;
    for (size_t iorb = 0; iorb < requested_orb_size; iorb++)
      ratios_buffer_ptr[ivp] += vp_phi_v_ptr[ivp * requested_orb_size + iorb] * invRow_deviceptr_list_ptr[ivp][iorb];
  }


  std::cout << "\n DONE \n";
  //ratios_buffer.updateFrom();

  size_t index = 0;
  for (size_t iw = 0; iw < nw; iw++)
    for (size_t iat = 0; iat < vp_list[iw].getTotalNum(); iat++)
      ratios_list[iw][iat] = ratios_buffer[index++];
#else
  size_t index = 0;
  for (size_t iw = 0; iw < vp_list.size(); iw++)
    for (size_t iat = 0; iat < vp_list[iw].getTotalNum(); iat++)
      ratios_list[iw][iat] = simd::dot(vp_phi_v.data_at(index++, 0), invRow_ptr_list[iw], requested_orb_size);
#endif
}

void LCAOrbitalSet::evaluateDetRatios(const VirtualParticleSet& VP,
                                      ValueVector& psi,
                                      const ValueVector& psiinv,
                                      std::vector<ValueType>& ratios)
{
  Vector<ValueType> vTemp(Temp.data(0), BasisSetSize);
  Vector<ValueType> invTemp(Temp.data(1), BasisSetSize);

  if (Identity)
    std::copy_n(psiinv.data(), psiinv.size(), invTemp.data());
  else
  {
    ScopedTimer local(mo_timer_);
    // when only a subset of orbitals is used, extract limited rows of C.
    Matrix<ValueType> C_occupied(C->data(), psiinv.size(), BasisSetSize);
    MatrixOperators::product_Atx(C_occupied, psiinv, invTemp);
  }

  for (size_t j = 0; j < VP.getTotalNum(); j++)
  {
    {
      ScopedTimer local(basis_timer_);
      myBasisSet->evaluateV(VP, j, vTemp.data());
    }
    ratios[j] = simd::dot(vTemp.data(), invTemp.data(), BasisSetSize);
  }
}

void LCAOrbitalSet::mw_evaluateVGLandDetRatioGrads(const RefVectorWithLeader<SPOSet>& spo_list,
                                                   const RefVectorWithLeader<ParticleSet>& P_list,
                                                   int iat,
                                                   const std::vector<const ValueType*>& invRow_ptr_list,
                                                   OffloadMWVGLArray& phi_vgl_v,
                                                   std::vector<ValueType>& ratios,
                                                   std::vector<GradType>& grads) const
{
  assert(this == &spo_list.getLeader());
  assert(phi_vgl_v.size(0) == DIM_VGL);
  assert(phi_vgl_v.size(1) == spo_list.size());

  mw_evaluateVGLImplGEMM(spo_list, P_list, iat, phi_vgl_v);
  // Device data of phi_vgl_v must be up-to-date upon return
  // phi_vgl_v.updateTo(); // moved updateTo to mw_evaluateVGLImplGEMM

  const size_t nw             = spo_list.size();
  const size_t norb_requested = phi_vgl_v.size(2);
  for (int iw = 0; iw < nw; iw++)
  {
    ratios[iw] = simd::dot(invRow_ptr_list[iw], phi_vgl_v.data_at(0, iw, 0), norb_requested);
    GradType dphi;
    for (size_t idim = 0; idim < DIM; idim++)
      dphi[idim] = simd::dot(invRow_ptr_list[iw], phi_vgl_v.data_at(idim + 1, iw, 0), norb_requested) / ratios[iw];
    grads[iw] = dphi;
  }
}

void LCAOrbitalSet::evaluateVGH(const ParticleSet& P, int iat, ValueVector& psi, GradVector& dpsi, HessVector& dhpsi)
{
  //TAKE CARE OF IDENTITY
  myBasisSet->evaluateVGH(P, iat, Temph);
  if (Identity)
    evaluate_vgh_impl(Temph, psi, dpsi, dhpsi);
  else
  {
    assert(psi.size() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), psi.size(), BasisSetSize);
    Product_ABt(Temph, C_partial_view, Temphv);
    evaluate_vgh_impl(Temphv, psi, dpsi, dhpsi);
  }
}

void LCAOrbitalSet::evaluateVGHGH(const ParticleSet& P,
                                  int iat,
                                  ValueVector& psi,
                                  GradVector& dpsi,
                                  HessVector& dhpsi,
                                  GGGVector& dghpsi)
{
  // APP_ABORT("LCAORbitalSet::evaluate(psi,gpsi,hpsi,ghpsi) not implemented\n");

  //TAKE CARE OF IDENTITY
  myBasisSet->evaluateVGHGH(P, iat, Tempgh);
  if (Identity)
    evaluate_vghgh_impl(Tempgh, psi, dpsi, dhpsi, dghpsi);
  else
  {
    assert(psi.size() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), psi.size(), BasisSetSize);
    Product_ABt(Tempgh, C_partial_view, Tempghv);
    evaluate_vghgh_impl(Tempghv, psi, dpsi, dhpsi, dghpsi);
  }
}

/* implement using gemm algorithm */
inline void LCAOrbitalSet::evaluate_vgl_impl(const vgl_type& temp,
                                             int i,
                                             ValueMatrix& logdet,
                                             GradMatrix& dlogdet,
                                             ValueMatrix& d2logdet) const
{
  const size_t output_size = logdet.cols();
  std::copy_n(temp.data(0), output_size, logdet[i]);
  const ValueType* restrict gx = temp.data(1);
  const ValueType* restrict gy = temp.data(2);
  const ValueType* restrict gz = temp.data(3);
  for (size_t j = 0; j < output_size; j++)
  {
    dlogdet[i][j][0] = gx[j];
    dlogdet[i][j][1] = gy[j];
    dlogdet[i][j][2] = gz[j];
  }
  std::copy_n(temp.data(4), output_size, d2logdet[i]);
}

inline void LCAOrbitalSet::evaluate_vgh_impl(const vgh_type& temp,
                                             int i,
                                             ValueMatrix& psi,
                                             GradMatrix& dpsi,
                                             HessMatrix& d2psi) const
{
  const size_t output_size = psi.cols();
  std::copy_n(temp.data(0), output_size, psi[i]);
  const ValueType* restrict gx  = temp.data(1);
  const ValueType* restrict gy  = temp.data(2);
  const ValueType* restrict gz  = temp.data(3);
  const ValueType* restrict hxx = temp.data(4);
  const ValueType* restrict hxy = temp.data(5);
  const ValueType* restrict hxz = temp.data(6);
  const ValueType* restrict hyy = temp.data(7);
  const ValueType* restrict hyz = temp.data(8);
  const ValueType* restrict hzz = temp.data(9);

  for (size_t j = 0; j < output_size; j++)
  {
    dpsi[i][j][0] = gx[j];
    dpsi[i][j][1] = gy[j];
    dpsi[i][j][2] = gz[j];

    d2psi[i][j](0, 0) = hxx[j];
    d2psi[i][j](0, 1) = d2psi[i][j](1, 0) = hxy[j];
    d2psi[i][j](0, 2) = d2psi[i][j](2, 0) = hxz[j];
    d2psi[i][j](1, 1)                     = hyy[j];
    d2psi[i][j](2, 1) = d2psi[i][j](1, 2) = hyz[j];
    d2psi[i][j](2, 2)                     = hzz[j];
  }
}

inline void LCAOrbitalSet::evaluate_ionderiv_v_impl(const vgl_type& temp, int i, GradMatrix& dpsi) const
{
  const size_t output_size     = dpsi.cols();
  const ValueType* restrict gx = temp.data(1);
  const ValueType* restrict gy = temp.data(2);
  const ValueType* restrict gz = temp.data(3);

  for (size_t j = 0; j < output_size; j++)
  {
    //As mentioned in SoaLocalizedBasisSet, LCAO's have a nice property that
    // for an atomic center, the ion gradient is the negative of the elecron gradient.
    // Hence minus signs for each of these.
    dpsi[i][j][0] = -gx[j];
    dpsi[i][j][1] = -gy[j];
    dpsi[i][j][2] = -gz[j];
  }
}

inline void LCAOrbitalSet::evaluate_ionderiv_vgl_impl(const vghgh_type& temp,
                                                      int i,
                                                      GradMatrix& dpsi,
                                                      HessMatrix& dgpsi,
                                                      GradMatrix& dlpsi) const
{
  const size_t output_size         = dpsi.cols();
  const ValueType* restrict gx     = temp.data(1);
  const ValueType* restrict gy     = temp.data(2);
  const ValueType* restrict gz     = temp.data(3);
  const ValueType* restrict hxx    = temp.data(4);
  const ValueType* restrict hxy    = temp.data(5);
  const ValueType* restrict hxz    = temp.data(6);
  const ValueType* restrict hyy    = temp.data(7);
  const ValueType* restrict hyz    = temp.data(8);
  const ValueType* restrict hzz    = temp.data(9);
  const ValueType* restrict gh_xxx = temp.data(10);
  const ValueType* restrict gh_xxy = temp.data(11);
  const ValueType* restrict gh_xxz = temp.data(12);
  const ValueType* restrict gh_xyy = temp.data(13);
  const ValueType* restrict gh_xzz = temp.data(15);
  const ValueType* restrict gh_yyy = temp.data(16);
  const ValueType* restrict gh_yyz = temp.data(17);
  const ValueType* restrict gh_yzz = temp.data(18);
  const ValueType* restrict gh_zzz = temp.data(19);

  for (size_t j = 0; j < output_size; j++)
  {
    //As mentioned in SoaLocalizedBasisSet, LCAO's have a nice property that
    // for an atomic center, the ion gradient is the negative of the elecron gradient.
    // Hence minus signs for each of these.
    dpsi[i][j][0] = -gx[j];
    dpsi[i][j][1] = -gy[j];
    dpsi[i][j][2] = -gz[j];

    dgpsi[i][j](0, 0) = -hxx[j];
    dgpsi[i][j](0, 1) = dgpsi[i][j](1, 0) = -hxy[j];
    dgpsi[i][j](0, 2) = dgpsi[i][j](2, 0) = -hxz[j];
    dgpsi[i][j](1, 1)                     = -hyy[j];
    dgpsi[i][j](2, 1) = dgpsi[i][j](1, 2) = -hyz[j];
    dgpsi[i][j](2, 2)                     = -hzz[j];

    //Since this returns the ion gradient of the laplacian, we have to trace the grad hessian vector.
    dlpsi[i][j][0] = -(gh_xxx[j] + gh_xyy[j] + gh_xzz[j]);
    dlpsi[i][j][1] = -(gh_xxy[j] + gh_yyy[j] + gh_yzz[j]);
    dlpsi[i][j][2] = -(gh_xxz[j] + gh_yyz[j] + gh_zzz[j]);
  }
}

void LCAOrbitalSet::evaluate_notranspose(const ParticleSet& P,
                                         int first,
                                         int last,
                                         ValueMatrix& logdet,
                                         GradMatrix& dlogdet,
                                         ValueMatrix& d2logdet)
{
  if (Identity)
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGL(P, iat, Temp);
      evaluate_vgl_impl(Temp, i, logdet, dlogdet, d2logdet);
    }
  }
  else
  {
    assert(logdet.cols() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), logdet.cols(), BasisSetSize);
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGL(P, iat, Temp);
      Product_ABt(Temp, C_partial_view, Tempv);
      evaluate_vgl_impl(Tempv, i, logdet, dlogdet, d2logdet);
    }
  }
}

void LCAOrbitalSet::evaluate_notranspose(const ParticleSet& P,
                                         int first,
                                         int last,
                                         ValueMatrix& logdet,
                                         GradMatrix& dlogdet,
                                         HessMatrix& grad_grad_logdet)
{
  if (Identity)
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGH(P, iat, Temph);
      evaluate_vgh_impl(Temph, i, logdet, dlogdet, grad_grad_logdet);
    }
  }
  else
  {
    assert(logdet.cols() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), logdet.cols(), BasisSetSize);
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGH(P, iat, Temph);
      Product_ABt(Temph, C_partial_view, Temphv);
      evaluate_vgh_impl(Temphv, i, logdet, dlogdet, grad_grad_logdet);
    }
  }
}

void LCAOrbitalSet::evaluate_notranspose(const ParticleSet& P,
                                         int first,
                                         int last,
                                         ValueMatrix& logdet,
                                         GradMatrix& dlogdet,
                                         HessMatrix& grad_grad_logdet,
                                         GGGMatrix& grad_grad_grad_logdet)
{
  if (Identity)
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGHGH(P, iat, Tempgh);
      evaluate_vghgh_impl(Tempgh, i, logdet, dlogdet, grad_grad_logdet, grad_grad_grad_logdet);
    }
  }
  else
  {
    assert(logdet.cols() <= OrbitalSetSize);
    ValueMatrix C_partial_view(C->data(), logdet.cols(), BasisSetSize);
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateVGHGH(P, iat, Tempgh);
      Product_ABt(Tempgh, C_partial_view, Tempghv);
      evaluate_vghgh_impl(Tempghv, i, logdet, dlogdet, grad_grad_logdet, grad_grad_grad_logdet);
    }
  }
}

void LCAOrbitalSet::evaluateGradSource(const ParticleSet& P,
                                       int first,
                                       int last,
                                       const ParticleSet& source,
                                       int iat_src,
                                       GradMatrix& gradphi)
{
  if (Identity)
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateGradSourceV(P, iat, source, iat_src, Temp);
      evaluate_ionderiv_v_impl(Temp, i, gradphi);
    }
  }
  else
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateGradSourceV(P, iat, source, iat_src, Temp);
      Product_ABt(Temp, *C, Tempv);
      evaluate_ionderiv_v_impl(Tempv, i, gradphi);
    }
  }
}

void LCAOrbitalSet::evaluateGradSource(const ParticleSet& P,
                                       int first,
                                       int last,
                                       const ParticleSet& source,
                                       int iat_src,
                                       GradMatrix& grad_phi,
                                       HessMatrix& grad_grad_phi,
                                       GradMatrix& grad_lapl_phi)
{
  if (Identity)
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateGradSourceVGL(P, iat, source, iat_src, Tempgh);
      evaluate_ionderiv_vgl_impl(Tempgh, i, grad_phi, grad_grad_phi, grad_lapl_phi);
    }
  }
  else
  {
    for (size_t i = 0, iat = first; iat < last; i++, iat++)
    {
      myBasisSet->evaluateGradSourceVGL(P, iat, source, iat_src, Tempgh);
      Product_ABt(Tempgh, *C, Tempghv);
      evaluate_ionderiv_vgl_impl(Tempghv, i, grad_phi, grad_grad_phi, grad_lapl_phi);
      //  evaluate_vghgh_impl(Tempghv, i, logdet, dlogdet, grad_grad_logdet, grad_grad_grad_logdet);
    }
  }
}

void LCAOrbitalSet::evaluateGradSourceRow(const ParticleSet& P,
                                          int iel,
                                          const ParticleSet& source,
                                          int iat_src,
                                          GradVector& gradphi)
{
  if (Identity)
  {
    myBasisSet->evaluateGradSourceV(P, iel, source, iat_src, Temp);
    evaluate_ionderiv_v_row_impl(Temp, gradphi);
  }
  else
  {
    myBasisSet->evaluateGradSourceV(P, iel, source, iat_src, Temp);
    Product_ABt(Temp, *C, Tempv);
    evaluate_ionderiv_v_row_impl(Tempv, gradphi);
  }
}

void LCAOrbitalSet::applyRotation(const ValueMatrix& rot_mat, bool use_stored_copy)
{
  if (!use_stored_copy)
    *C_copy = *C;
  //gemm is out-of-place
  BLAS::gemm('N', 'T', BasisSetSize, OrbitalSetSize, OrbitalSetSize, RealType(1.0), C_copy->data(), BasisSetSize,
             rot_mat.data(), OrbitalSetSize, RealType(0.0), C->data(), BasisSetSize);
  C->updateTo();

  /* debugging code
  app_log() << "PRINTING MO COEFFICIENTS AFTER ROTATION " << objectName << std::endl;
  for (int j = 0; j < OrbitalSetSize; j++)
    for (int i = 0; i < BasisSetSize; i++)
    {
      app_log() << " " << std::right << std::fixed << std::setprecision(16) << std::setw(23) << std::scientific
                << *(C->data() + j * BasisSetSize + i);

      if ((j * BasisSetSize + i + 1) % 4 == 0)
        app_log() << std::endl;
    }
  */
}

} // namespace qmcplusplus
