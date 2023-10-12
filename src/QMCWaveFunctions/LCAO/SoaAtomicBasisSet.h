//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////

/** @file SoaAtomicBasisSet.h
 */
#ifndef QMCPLUSPLUS_SOA_SPHERICALORBITAL_BASISSET_H
#define QMCPLUSPLUS_SOA_SPHERICALORBITAL_BASISSET_H

#include "CPU/math.hpp"
#include "OptimizableObject.h"
#include <ResourceCollection.h>

namespace qmcplusplus
{
/* A basis set for a center type 
   *
   * @tparam ROT : radial function type, e.g.,NGFunctor<T>
   * @tparam SH : spherical or carteisan Harmonics for (l,m) expansion
   *
   * \f$ \phi_{n,l,m}({\bf r})=R_{n,l}(r) Y_{l,m}(\theta) \f$
   */
template<typename ROT, typename SH>
struct SoaAtomicBasisSet
{
  using RadialOrbital_t = ROT;
  using RealType        = typename ROT::RealType;
  using GridType        = typename ROT::GridType;
  using ValueType       = typename QMCTraits::ValueType;
  using OffloadArray4D  = Array<ValueType, 4, OffloadPinnedAllocator<ValueType>>;
  using OffloadArray3D  = Array<ValueType, 3, OffloadPinnedAllocator<ValueType>>;
  using OffloadArray2D  = Array<ValueType, 2, OffloadPinnedAllocator<ValueType>>;
  using OffloadVector   = Vector<ValueType, OffloadPinnedAllocator<ValueType>>;

  ///size of the basis set
  int BasisSetSize;
  ///Number of Cell images for the evaluation of the orbital with PBC. If No PBC, should be 0;
  TinyVector<int, 3> PBCImages;
  ///Coordinates of SuperTwist
  TinyVector<double, 3> SuperTwist;
  ///Phase Factor array
  std::vector<QMCTraits::ValueType> periodic_image_phase_factors;
  ///maximum radius of this center
  RealType Rmax;
  ///spherical harmonics
  SH Ylm;
  ///radial orbitals
  ROT MultiRnl;
  ///index of the corresponding real Spherical Harmonic with quantum numbers \f$ (l,m) \f$
  aligned_vector<int> LM;
  /**index of the corresponding radial orbital with quantum numbers \f$ (n,l) \f$ */
  aligned_vector<int> NL;
  ///container for the quantum-numbers
  std::vector<QuantumNumberType> RnlID;
  ///temporary storage
  //VectorSoaContainer<RealType, 4> tempS;
  VectorSoaContainer<RealType, 4, OffloadPinnedAllocator<RealType>> tempS;
  NewTimer& ylm_timer_;
  NewTimer& rnl_timer_;
  NewTimer& pbc_timer_;
  NewTimer& nelec_pbc_timer_;
  NewTimer& phase_timer_;
  NewTimer& psi_timer_;
  struct SoaAtomicBSetMultiWalkerMem;
  ResourceHandle<SoaAtomicBSetMultiWalkerMem> mw_mem_handle_;
  // void createResource(ResourceCollection& collection) const;
  // void acquireResource(ResourceCollection& collection, const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list) const;
  // void releaseResource(ResourceCollection& collection, const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list) const;

  ///the constructor
  explicit SoaAtomicBasisSet(int lmax, bool addsignforM = false)
      : Ylm(lmax, addsignforM),
        ylm_timer_(createGlobalTimer("SoaAtomicBasisSet::Ylm", timer_level_fine)),
        rnl_timer_(createGlobalTimer("SoaAtomicBasisSet::Rnl", timer_level_fine)),
        pbc_timer_(createGlobalTimer("SoaAtomicBasisSet::pbc_images", timer_level_fine)),
        nelec_pbc_timer_(createGlobalTimer("SoaAtomicBasisSet::nelec_pbc_images", timer_level_fine)),
        phase_timer_(createGlobalTimer("SoaAtomicBasisSet::phase", timer_level_fine)),
        psi_timer_(createGlobalTimer("SoaAtomicBasisSet::psi", timer_level_fine))
  {}

  void checkInVariables(opt_variables_type& active)
  {
    //for(size_t nl=0; nl<Rnl.size(); nl++)
    //  Rnl[nl]->checkInVariables(active);
  }

  void checkOutVariables(const opt_variables_type& active)
  {
    //for(size_t nl=0; nl<Rnl.size(); nl++)
    //  Rnl[nl]->checkOutVariables(active);
  }

  void resetParameters(const opt_variables_type& active)
  {
    //for(size_t nl=0; nl<Rnl.size(); nl++)
    //  Rnl[nl]->resetParameters(active);
  }

  /** return the number of basis functions
      */
  inline int getBasisSetSize() const
  {
    //=NL.size();
    return BasisSetSize;
  }

  /** Set the number of periodic image for the evaluation of the orbitals and the phase factor.
   * In the case of Non-PBC, PBCImages=(1,1,1), SuperTwist(0,0,0) and the PhaseFactor=1.
   */
  void setPBCParams(const TinyVector<int, 3>& pbc_images,
                    const TinyVector<double, 3> supertwist,
                    const std::vector<QMCTraits::ValueType>& PeriodicImagePhaseFactors)
  {
    PBCImages                    = pbc_images;
    periodic_image_phase_factors = PeriodicImagePhaseFactors;
    SuperTwist                   = supertwist;
  }


  /** implement a BasisSetBase virtual function
   *
   * Set Rmax and BasisSetSize
   * @todo Should be able to overwrite Rmax to be much smaller than the maximum grid
   */
  inline void setBasisSetSize(int n)
  {
    BasisSetSize = LM.size();
    tempS.resize(std::max(Ylm.size(), RnlID.size()));
  }

  /** Set Rmax */
  template<typename T>
  inline void setRmax(T rmax)
  {
    Rmax = (rmax > 0) ? rmax : MultiRnl.rmax();
  }

  ///set the current offset
  inline void setCenter(int c, int offset) {}

  /// Sets a boolean vector for S-type orbitals.  Used for cusp correction.
  void queryOrbitalsForSType(std::vector<bool>& s_orbitals) const
  {
    for (int i = 0; i < BasisSetSize; i++)
    {
      s_orbitals[i] = (RnlID[NL[i]][1] == 0);
    }
  }

  /** evaluate VGL
   */
  template<typename LAT, typename T, typename PosType, typename VGL>
  inline void evaluateVGL(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGL& vgl, PosType Tv)
  {
    int TransX, TransY, TransZ;

    PosType dr_new;
    T r_new;
    // T psi_new, dpsi_x_new, dpsi_y_new, dpsi_z_new,d2psi_new;

#if not defined(QMC_COMPLEX)
    const ValueType correctphase = 1;
#else

    RealType phasearg = SuperTwist[0] * Tv[0] + SuperTwist[1] * Tv[1] + SuperTwist[2] * Tv[2];
    RealType s, c;
    qmcplusplus::sincos(-phasearg, &s, &c);
    const ValueType correctphase(c, s);
#endif

    constexpr T cone(1);
    constexpr T ctwo(2);


    //one can assert the alignment
    RealType* restrict phi   = tempS.data(0);
    RealType* restrict dphi  = tempS.data(1);
    RealType* restrict d2phi = tempS.data(2);

    //V,Gx,Gy,Gz,L
    auto* restrict psi      = vgl.data(0) + offset;
    const T* restrict ylm_v = Ylm[0]; //value
    auto* restrict dpsi_x   = vgl.data(1) + offset;
    const T* restrict ylm_x = Ylm[1]; //gradX
    auto* restrict dpsi_y   = vgl.data(2) + offset;
    const T* restrict ylm_y = Ylm[2]; //gradY
    auto* restrict dpsi_z   = vgl.data(3) + offset;
    const T* restrict ylm_z = Ylm[3]; //gradZ
    auto* restrict d2psi    = vgl.data(4) + offset;
    const T* restrict ylm_l = Ylm[4]; //lap

    for (size_t ib = 0; ib < BasisSetSize; ++ib)
    {
      psi[ib]    = 0;
      dpsi_x[ib] = 0;
      dpsi_y[ib] = 0;
      dpsi_z[ib] = 0;
      d2psi[ib]  = 0;
    }
    //Phase_idx (iter) needs to be initialized at -1 as it has to be incremented first to comply with the if statement (r_new >=Rmax)
    int iter = -1;
    for (int i = 0; i <= PBCImages[0]; i++) //loop Translation over X
    {
      //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
      TransX = ((i % 2) * 2 - 1) * ((i + 1) / 2);
      for (int j = 0; j <= PBCImages[1]; j++) //loop Translation over Y
      {
        //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
        TransY = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        for (int k = 0; k <= PBCImages[2]; k++) //loop Translation over Z
        {
          //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
          TransZ = ((k % 2) * 2 - 1) * ((k + 1) / 2);

          dr_new[0] = dr[0] + (TransX * lattice.R(0, 0) + TransY * lattice.R(1, 0) + TransZ * lattice.R(2, 0));
          dr_new[1] = dr[1] + (TransX * lattice.R(0, 1) + TransY * lattice.R(1, 1) + TransZ * lattice.R(2, 1));
          dr_new[2] = dr[2] + (TransX * lattice.R(0, 2) + TransY * lattice.R(1, 2) + TransZ * lattice.R(2, 2));

          r_new = std::sqrt(dot(dr_new, dr_new));

          iter++;
          if (r_new >= Rmax)
            continue;

          //SIGN Change!!
          const T x = -dr_new[0], y = -dr_new[1], z = -dr_new[2];
          Ylm.evaluateVGL(x, y, z);

          MultiRnl.evaluate(r_new, phi, dphi, d2phi);

          const T rinv = cone / r_new;

          ///Phase for PBC containing the phase for the nearest image displacement and the correction due to the Distance table.
          const ValueType Phase = periodic_image_phase_factors[iter] * correctphase;

          for (size_t ib = 0; ib < BasisSetSize; ++ib)
          {
            const int nl(NL[ib]);
            const int lm(LM[ib]);
            const T drnloverr = rinv * dphi[nl];
            const T ang       = ylm_v[lm];
            const T gr_x      = drnloverr * x;
            const T gr_y      = drnloverr * y;
            const T gr_z      = drnloverr * z;
            const T ang_x     = ylm_x[lm];
            const T ang_y     = ylm_y[lm];
            const T ang_z     = ylm_z[lm];
            const T vr        = phi[nl];

            psi[ib] += ang * vr * Phase;
            dpsi_x[ib] += (ang * gr_x + vr * ang_x) * Phase;
            dpsi_y[ib] += (ang * gr_y + vr * ang_y) * Phase;
            dpsi_z[ib] += (ang * gr_z + vr * ang_z) * Phase;
            d2psi[ib] += (ang * (ctwo * drnloverr + d2phi[nl]) + ctwo * (gr_x * ang_x + gr_y * ang_y + gr_z * ang_z) +
                          vr * ylm_l[lm]) *
                Phase;
          }
        }
      }
    }
  }

  template<typename LAT, typename T, typename PosType, typename VGH>
  inline void evaluateVGH(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGH& vgh)
  {
    int TransX, TransY, TransZ;

    PosType dr_new;
    T r_new;

    constexpr T cone(1);

    //one can assert the alignment
    RealType* restrict phi   = tempS.data(0);
    RealType* restrict dphi  = tempS.data(1);
    RealType* restrict d2phi = tempS.data(2);

    //V,Gx,Gy,Gz,L
    auto* restrict psi      = vgh.data(0) + offset;
    const T* restrict ylm_v = Ylm[0]; //value
    auto* restrict dpsi_x   = vgh.data(1) + offset;
    const T* restrict ylm_x = Ylm[1]; //gradX
    auto* restrict dpsi_y   = vgh.data(2) + offset;
    const T* restrict ylm_y = Ylm[2]; //gradY
    auto* restrict dpsi_z   = vgh.data(3) + offset;
    const T* restrict ylm_z = Ylm[3]; //gradZ

    auto* restrict dhpsi_xx  = vgh.data(4) + offset;
    const T* restrict ylm_xx = Ylm[4];
    auto* restrict dhpsi_xy  = vgh.data(5) + offset;
    const T* restrict ylm_xy = Ylm[5];
    auto* restrict dhpsi_xz  = vgh.data(6) + offset;
    const T* restrict ylm_xz = Ylm[6];
    auto* restrict dhpsi_yy  = vgh.data(7) + offset;
    const T* restrict ylm_yy = Ylm[7];
    auto* restrict dhpsi_yz  = vgh.data(8) + offset;
    const T* restrict ylm_yz = Ylm[8];
    auto* restrict dhpsi_zz  = vgh.data(9) + offset;
    const T* restrict ylm_zz = Ylm[9];

    for (size_t ib = 0; ib < BasisSetSize; ++ib)
    {
      psi[ib]      = 0;
      dpsi_x[ib]   = 0;
      dpsi_y[ib]   = 0;
      dpsi_z[ib]   = 0;
      dhpsi_xx[ib] = 0;
      dhpsi_xy[ib] = 0;
      dhpsi_xz[ib] = 0;
      dhpsi_yy[ib] = 0;
      dhpsi_yz[ib] = 0;
      dhpsi_zz[ib] = 0;
      //      d2psi[ib]  = 0;
    }

    for (int i = 0; i <= PBCImages[0]; i++) //loop Translation over X
    {
      //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
      TransX = ((i % 2) * 2 - 1) * ((i + 1) / 2);
      for (int j = 0; j <= PBCImages[1]; j++) //loop Translation over Y
      {
        //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
        TransY = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        for (int k = 0; k <= PBCImages[2]; k++) //loop Translation over Z
        {
          //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
          TransZ    = ((k % 2) * 2 - 1) * ((k + 1) / 2);
          dr_new[0] = dr[0] + TransX * lattice.R(0, 0) + TransY * lattice.R(1, 0) + TransZ * lattice.R(2, 0);
          dr_new[1] = dr[1] + TransX * lattice.R(0, 1) + TransY * lattice.R(1, 1) + TransZ * lattice.R(2, 1);
          dr_new[2] = dr[2] + TransX * lattice.R(0, 2) + TransY * lattice.R(1, 2) + TransZ * lattice.R(2, 2);
          r_new     = std::sqrt(dot(dr_new, dr_new));

          //const size_t ib_max=NL.size();
          if (r_new >= Rmax)
            continue;

          //SIGN Change!!
          const T x = -dr_new[0], y = -dr_new[1], z = -dr_new[2];
          Ylm.evaluateVGH(x, y, z);

          MultiRnl.evaluate(r_new, phi, dphi, d2phi);

          const T rinv = cone / r_new;


          for (size_t ib = 0; ib < BasisSetSize; ++ib)
          {
            const int nl(NL[ib]);
            const int lm(LM[ib]);
            const T drnloverr = rinv * dphi[nl];
            const T ang       = ylm_v[lm];
            const T gr_x      = drnloverr * x;
            const T gr_y      = drnloverr * y;
            const T gr_z      = drnloverr * z;

            //The non-strictly diagonal term in \partial_i \partial_j R_{nl} is
            // \frac{x_i x_j}{r^2}\left(\frac{\partial^2 R_{nl}}{\partial r^2} - \frac{1}{r}\frac{\partial R_{nl}}{\partial r})
            // To save recomputation, I evaluate everything except the x_i*x_j term once, and store it in
            // gr2_tmp.  The full term is obtained by x_i*x_j*gr2_tmp.
            const T gr2_tmp = rinv * rinv * (d2phi[nl] - drnloverr);
            const T gr_xx   = x * x * gr2_tmp + drnloverr;
            const T gr_xy   = x * y * gr2_tmp;
            const T gr_xz   = x * z * gr2_tmp;
            const T gr_yy   = y * y * gr2_tmp + drnloverr;
            const T gr_yz   = y * z * gr2_tmp;
            const T gr_zz   = z * z * gr2_tmp + drnloverr;

            const T ang_x  = ylm_x[lm];
            const T ang_y  = ylm_y[lm];
            const T ang_z  = ylm_z[lm];
            const T ang_xx = ylm_xx[lm];
            const T ang_xy = ylm_xy[lm];
            const T ang_xz = ylm_xz[lm];
            const T ang_yy = ylm_yy[lm];
            const T ang_yz = ylm_yz[lm];
            const T ang_zz = ylm_zz[lm];

            const T vr = phi[nl];

            psi[ib] += ang * vr;
            dpsi_x[ib] += ang * gr_x + vr * ang_x;
            dpsi_y[ib] += ang * gr_y + vr * ang_y;
            dpsi_z[ib] += ang * gr_z + vr * ang_z;


            // \partial_i \partial_j (R*Y) = Y \partial_i \partial_j R + R \partial_i \partial_j Y
            //                             + (\partial_i R) (\partial_j Y) + (\partial_j R)(\partial_i Y)
            dhpsi_xx[ib] += gr_xx * ang + ang_xx * vr + 2.0 * gr_x * ang_x;
            dhpsi_xy[ib] += gr_xy * ang + ang_xy * vr + gr_x * ang_y + gr_y * ang_x;
            dhpsi_xz[ib] += gr_xz * ang + ang_xz * vr + gr_x * ang_z + gr_z * ang_x;
            dhpsi_yy[ib] += gr_yy * ang + ang_yy * vr + 2.0 * gr_y * ang_y;
            dhpsi_yz[ib] += gr_yz * ang + ang_yz * vr + gr_y * ang_z + gr_z * ang_y;
            dhpsi_zz[ib] += gr_zz * ang + ang_zz * vr + 2.0 * gr_z * ang_z;
          }
        }
      }
    }
  }

  template<typename LAT, typename T, typename PosType, typename VGHGH>
  inline void evaluateVGHGH(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGHGH& vghgh)
  {
    int TransX, TransY, TransZ;

    PosType dr_new;
    T r_new;

    constexpr T cone(1);

    //one can assert the alignment
    RealType* restrict phi   = tempS.data(0);
    RealType* restrict dphi  = tempS.data(1);
    RealType* restrict d2phi = tempS.data(2);
    RealType* restrict d3phi = tempS.data(3);

    //V,Gx,Gy,Gz,L
    auto* restrict psi      = vghgh.data(0) + offset;
    const T* restrict ylm_v = Ylm[0]; //value
    auto* restrict dpsi_x   = vghgh.data(1) + offset;
    const T* restrict ylm_x = Ylm[1]; //gradX
    auto* restrict dpsi_y   = vghgh.data(2) + offset;
    const T* restrict ylm_y = Ylm[2]; //gradY
    auto* restrict dpsi_z   = vghgh.data(3) + offset;
    const T* restrict ylm_z = Ylm[3]; //gradZ

    auto* restrict dhpsi_xx  = vghgh.data(4) + offset;
    const T* restrict ylm_xx = Ylm[4];
    auto* restrict dhpsi_xy  = vghgh.data(5) + offset;
    const T* restrict ylm_xy = Ylm[5];
    auto* restrict dhpsi_xz  = vghgh.data(6) + offset;
    const T* restrict ylm_xz = Ylm[6];
    auto* restrict dhpsi_yy  = vghgh.data(7) + offset;
    const T* restrict ylm_yy = Ylm[7];
    auto* restrict dhpsi_yz  = vghgh.data(8) + offset;
    const T* restrict ylm_yz = Ylm[8];
    auto* restrict dhpsi_zz  = vghgh.data(9) + offset;
    const T* restrict ylm_zz = Ylm[9];

    auto* restrict dghpsi_xxx = vghgh.data(10) + offset;
    const T* restrict ylm_xxx = Ylm[10];
    auto* restrict dghpsi_xxy = vghgh.data(11) + offset;
    const T* restrict ylm_xxy = Ylm[11];
    auto* restrict dghpsi_xxz = vghgh.data(12) + offset;
    const T* restrict ylm_xxz = Ylm[12];
    auto* restrict dghpsi_xyy = vghgh.data(13) + offset;
    const T* restrict ylm_xyy = Ylm[13];
    auto* restrict dghpsi_xyz = vghgh.data(14) + offset;
    const T* restrict ylm_xyz = Ylm[14];
    auto* restrict dghpsi_xzz = vghgh.data(15) + offset;
    const T* restrict ylm_xzz = Ylm[15];
    auto* restrict dghpsi_yyy = vghgh.data(16) + offset;
    const T* restrict ylm_yyy = Ylm[16];
    auto* restrict dghpsi_yyz = vghgh.data(17) + offset;
    const T* restrict ylm_yyz = Ylm[17];
    auto* restrict dghpsi_yzz = vghgh.data(18) + offset;
    const T* restrict ylm_yzz = Ylm[18];
    auto* restrict dghpsi_zzz = vghgh.data(19) + offset;
    const T* restrict ylm_zzz = Ylm[19];

    for (size_t ib = 0; ib < BasisSetSize; ++ib)
    {
      psi[ib] = 0;

      dpsi_x[ib] = 0;
      dpsi_y[ib] = 0;
      dpsi_z[ib] = 0;

      dhpsi_xx[ib] = 0;
      dhpsi_xy[ib] = 0;
      dhpsi_xz[ib] = 0;
      dhpsi_yy[ib] = 0;
      dhpsi_yz[ib] = 0;
      dhpsi_zz[ib] = 0;

      dghpsi_xxx[ib] = 0;
      dghpsi_xxy[ib] = 0;
      dghpsi_xxz[ib] = 0;
      dghpsi_xyy[ib] = 0;
      dghpsi_xyz[ib] = 0;
      dghpsi_xzz[ib] = 0;
      dghpsi_yyy[ib] = 0;
      dghpsi_yyz[ib] = 0;
      dghpsi_yzz[ib] = 0;
      dghpsi_zzz[ib] = 0;
    }

    for (int i = 0; i <= PBCImages[0]; i++) //loop Translation over X
    {
      //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
      TransX = ((i % 2) * 2 - 1) * ((i + 1) / 2);
      for (int j = 0; j <= PBCImages[1]; j++) //loop Translation over Y
      {
        //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
        TransY = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        for (int k = 0; k <= PBCImages[2]; k++) //loop Translation over Z
        {
          //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
          TransZ    = ((k % 2) * 2 - 1) * ((k + 1) / 2);
          dr_new[0] = dr[0] + TransX * lattice.R(0, 0) + TransY * lattice.R(1, 0) + TransZ * lattice.R(2, 0);
          dr_new[1] = dr[1] + TransX * lattice.R(0, 1) + TransY * lattice.R(1, 1) + TransZ * lattice.R(2, 1);
          dr_new[2] = dr[2] + TransX * lattice.R(0, 2) + TransY * lattice.R(1, 2) + TransZ * lattice.R(2, 2);
          r_new     = std::sqrt(dot(dr_new, dr_new));

          //const size_t ib_max=NL.size();
          if (r_new >= Rmax)
            continue;

          //SIGN Change!!
          const T x = -dr_new[0], y = -dr_new[1], z = -dr_new[2];
          Ylm.evaluateVGHGH(x, y, z);

          MultiRnl.evaluate(r_new, phi, dphi, d2phi, d3phi);

          const T rinv = cone / r_new;
          const T xu = x * rinv, yu = y * rinv, zu = z * rinv;
          for (size_t ib = 0; ib < BasisSetSize; ++ib)
          {
            const int nl(NL[ib]);
            const int lm(LM[ib]);
            const T drnloverr = rinv * dphi[nl];
            const T ang       = ylm_v[lm];
            const T gr_x      = drnloverr * x;
            const T gr_y      = drnloverr * y;
            const T gr_z      = drnloverr * z;

            //The non-strictly diagonal term in \partial_i \partial_j R_{nl} is
            // \frac{x_i x_j}{r^2}\left(\frac{\partial^2 R_{nl}}{\partial r^2} - \frac{1}{r}\frac{\partial R_{nl}}{\partial r})
            // To save recomputation, I evaluate everything except the x_i*x_j term once, and store it in
            // gr2_tmp.  The full term is obtained by x_i*x_j*gr2_tmp.  This is p(r) in the notes.
            const T gr2_tmp = rinv * (d2phi[nl] - drnloverr);

            const T gr_xx = x * xu * gr2_tmp + drnloverr;
            const T gr_xy = x * yu * gr2_tmp;
            const T gr_xz = x * zu * gr2_tmp;
            const T gr_yy = y * yu * gr2_tmp + drnloverr;
            const T gr_yz = y * zu * gr2_tmp;
            const T gr_zz = z * zu * gr2_tmp + drnloverr;

            //This is q(r) in the notes.
            const T gr3_tmp = d3phi[nl] - 3.0 * gr2_tmp;

            const T gr_xxx = xu * xu * xu * gr3_tmp + gr2_tmp * (3. * xu);
            const T gr_xxy = xu * xu * yu * gr3_tmp + gr2_tmp * yu;
            const T gr_xxz = xu * xu * zu * gr3_tmp + gr2_tmp * zu;
            const T gr_xyy = xu * yu * yu * gr3_tmp + gr2_tmp * xu;
            const T gr_xyz = xu * yu * zu * gr3_tmp;
            const T gr_xzz = xu * zu * zu * gr3_tmp + gr2_tmp * xu;
            const T gr_yyy = yu * yu * yu * gr3_tmp + gr2_tmp * (3. * yu);
            const T gr_yyz = yu * yu * zu * gr3_tmp + gr2_tmp * zu;
            const T gr_yzz = yu * zu * zu * gr3_tmp + gr2_tmp * yu;
            const T gr_zzz = zu * zu * zu * gr3_tmp + gr2_tmp * (3. * zu);


            //Angular derivatives up to third
            const T ang_x = ylm_x[lm];
            const T ang_y = ylm_y[lm];
            const T ang_z = ylm_z[lm];

            const T ang_xx = ylm_xx[lm];
            const T ang_xy = ylm_xy[lm];
            const T ang_xz = ylm_xz[lm];
            const T ang_yy = ylm_yy[lm];
            const T ang_yz = ylm_yz[lm];
            const T ang_zz = ylm_zz[lm];

            const T ang_xxx = ylm_xxx[lm];
            const T ang_xxy = ylm_xxy[lm];
            const T ang_xxz = ylm_xxz[lm];
            const T ang_xyy = ylm_xyy[lm];
            const T ang_xyz = ylm_xyz[lm];
            const T ang_xzz = ylm_xzz[lm];
            const T ang_yyy = ylm_yyy[lm];
            const T ang_yyz = ylm_yyz[lm];
            const T ang_yzz = ylm_yzz[lm];
            const T ang_zzz = ylm_zzz[lm];

            const T vr = phi[nl];

            psi[ib] += ang * vr;
            dpsi_x[ib] += ang * gr_x + vr * ang_x;
            dpsi_y[ib] += ang * gr_y + vr * ang_y;
            dpsi_z[ib] += ang * gr_z + vr * ang_z;


            // \partial_i \partial_j (R*Y) = Y \partial_i \partial_j R + R \partial_i \partial_j Y
            //                             + (\partial_i R) (\partial_j Y) + (\partial_j R)(\partial_i Y)
            dhpsi_xx[ib] += gr_xx * ang + ang_xx * vr + 2.0 * gr_x * ang_x;
            dhpsi_xy[ib] += gr_xy * ang + ang_xy * vr + gr_x * ang_y + gr_y * ang_x;
            dhpsi_xz[ib] += gr_xz * ang + ang_xz * vr + gr_x * ang_z + gr_z * ang_x;
            dhpsi_yy[ib] += gr_yy * ang + ang_yy * vr + 2.0 * gr_y * ang_y;
            dhpsi_yz[ib] += gr_yz * ang + ang_yz * vr + gr_y * ang_z + gr_z * ang_y;
            dhpsi_zz[ib] += gr_zz * ang + ang_zz * vr + 2.0 * gr_z * ang_z;

            dghpsi_xxx[ib] += gr_xxx * ang + vr * ang_xxx + 3.0 * gr_xx * ang_x + 3.0 * gr_x * ang_xx;
            dghpsi_xxy[ib] +=
                gr_xxy * ang + vr * ang_xxy + gr_xx * ang_y + ang_xx * gr_y + 2.0 * gr_xy * ang_x + 2.0 * ang_xy * gr_x;
            dghpsi_xxz[ib] +=
                gr_xxz * ang + vr * ang_xxz + gr_xx * ang_z + ang_xx * gr_z + 2.0 * gr_xz * ang_x + 2.0 * ang_xz * gr_x;
            dghpsi_xyy[ib] +=
                gr_xyy * ang + vr * ang_xyy + gr_yy * ang_x + ang_yy * gr_x + 2.0 * gr_xy * ang_y + 2.0 * ang_xy * gr_y;
            dghpsi_xyz[ib] += gr_xyz * ang + vr * ang_xyz + gr_xy * ang_z + ang_xy * gr_z + gr_yz * ang_x +
                ang_yz * gr_x + gr_xz * ang_y + ang_xz * gr_y;
            dghpsi_xzz[ib] +=
                gr_xzz * ang + vr * ang_xzz + gr_zz * ang_x + ang_zz * gr_x + 2.0 * gr_xz * ang_z + 2.0 * ang_xz * gr_z;
            dghpsi_yyy[ib] += gr_yyy * ang + vr * ang_yyy + 3.0 * gr_yy * ang_y + 3.0 * gr_y * ang_yy;
            dghpsi_yyz[ib] +=
                gr_yyz * ang + vr * ang_yyz + gr_yy * ang_z + ang_yy * gr_z + 2.0 * gr_yz * ang_y + 2.0 * ang_yz * gr_y;
            dghpsi_yzz[ib] +=
                gr_yzz * ang + vr * ang_yzz + gr_zz * ang_y + ang_zz * gr_y + 2.0 * gr_yz * ang_z + 2.0 * ang_yz * gr_z;
            dghpsi_zzz[ib] += gr_zzz * ang + vr * ang_zzz + 3.0 * gr_zz * ang_z + 3.0 * gr_z * ang_zz;
          }
        }
      }
    }
  }

  /** evaluate V
  */
  template<typename LAT, typename T, typename PosType, typename VT>
  inline void evaluateV(const LAT& lattice, const T r, const PosType& dr, VT* restrict psi, PosType Tv)
  {
    int TransX, TransY, TransZ;

    PosType dr_new;
    T r_new;

#if not defined(QMC_COMPLEX)
    const ValueType correctphase = 1.0;
#else

    RealType phasearg = SuperTwist[0] * Tv[0] + SuperTwist[1] * Tv[1] + SuperTwist[2] * Tv[2];
    RealType s, c;
    qmcplusplus::sincos(-phasearg, &s, &c);
    const ValueType correctphase(c, s);

#endif

    RealType* restrict ylm_v = tempS.data(0);
    RealType* restrict phi_r = tempS.data(1);

    for (size_t ib = 0; ib < BasisSetSize; ++ib)
      psi[ib] = 0;
    //Phase_idx (iter) needs to be initialized at -1 as it has to be incremented first to comply with the if statement (r_new >=Rmax)
    int iter = -1;
    for (int i = 0; i <= PBCImages[0]; i++) //loop Translation over X
    {
      //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
      TransX = ((i % 2) * 2 - 1) * ((i + 1) / 2);
      for (int j = 0; j <= PBCImages[1]; j++) //loop Translation over Y
      {
        //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
        TransY = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        for (int k = 0; k <= PBCImages[2]; k++) //loop Translation over Z
        {
          //Allows to increment cells from 0,1,-1,2,-2,3,-3 etc...
          TransZ = ((k % 2) * 2 - 1) * ((k + 1) / 2);

          dr_new[0] = dr[0] + (TransX * lattice.R(0, 0) + TransY * lattice.R(1, 0) + TransZ * lattice.R(2, 0));
          dr_new[1] = dr[1] + (TransX * lattice.R(0, 1) + TransY * lattice.R(1, 1) + TransZ * lattice.R(2, 1));
          dr_new[2] = dr[2] + (TransX * lattice.R(0, 2) + TransY * lattice.R(1, 2) + TransZ * lattice.R(2, 2));

          r_new = std::sqrt(dot(dr_new, dr_new));
          iter++;
          if (r_new >= Rmax)
            continue;

          Ylm.evaluateV(-dr_new[0], -dr_new[1], -dr_new[2], ylm_v);
          MultiRnl.evaluate(r_new, phi_r);
          ///Phase for PBC containing the phase for the nearest image displacement and the correction due to the Distance table.
          const ValueType Phase = periodic_image_phase_factors[iter] * correctphase;
          for (size_t ib = 0; ib < BasisSetSize; ++ib)
            psi[ib] += ylm_v[LM[ib]] * phi_r[NL[ib]] * Phase;
        }
      }
    }
  }


  /** evaluate VGL
   */
  template<typename LAT, typename VT>
  inline void mw_evaluateVGL(const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list,
                             const LAT& lattice,
                             Array<VT, 3, OffloadPinnedAllocator<VT>>& psi_vgl,
                             const Vector<RealType, OffloadPinnedAllocator<RealType>>& displ_list,
                             const Vector<RealType, OffloadPinnedAllocator<RealType>>& Tv_list,
                             const size_t nElec,
                             const size_t nBasTot,
                             const size_t c,
                             const size_t BasisOffset,
                             const size_t NumCenters)
  {
    assert(this == &atom_bs_list.getLeader());
    auto& atom_bs_leader = atom_bs_list.template getCastedLeader<SoaAtomicBasisSet<ROT, SH>>();

    int Nx   = PBCImages[0] + 1;
    int Ny   = PBCImages[1] + 1;
    int Nz   = PBCImages[2] + 1;
    int Nyz  = Ny * Nz;
    int Nxyz = Nx * Nyz;

    //assert(psi_vgl.size(0) == 5);
    assert(psi_vgl.size(1) == nElec);
    assert(psi_vgl.size(2) == nBasTot);


    auto& ylm_vgl = atom_bs_leader.mw_mem_handle_.getResource().ylm_vgl;
    auto& rnl_vgl = atom_bs_leader.mw_mem_handle_.getResource().rnl_vgl;
    auto& dr      = atom_bs_leader.mw_mem_handle_.getResource().dr;
    auto& r       = atom_bs_leader.mw_mem_handle_.getResource().r;

    size_t nRnl = RnlID.size();
    size_t nYlm = Ylm.size();

    ylm_vgl.resize(5, nElec, Nxyz, nYlm);
    rnl_vgl.resize(3, nElec, Nxyz, nRnl);
    dr.resize(nElec, Nxyz, 3);
    r.resize(nElec, Nxyz);


    // TODO: move these outside?
    auto& dr_pbc       = atom_bs_leader.mw_mem_handle_.getResource().dr_pbc;
    auto& correctphase = atom_bs_leader.mw_mem_handle_.getResource().correctphase;
    dr_pbc.resize(Nxyz, 3);
    correctphase.resize(nElec);

    // TODO: should this be VT?
    constexpr RealType cone(1);
    constexpr RealType ctwo(2);

    //V,Gx,Gy,Gz,L
    auto* restrict psi_devptr             = psi_vgl.device_data_at(0, 0, 0);
    auto* restrict dpsi_x_devptr          = psi_vgl.device_data_at(1, 0, 0);
    auto* restrict dpsi_y_devptr          = psi_vgl.device_data_at(2, 0, 0);
    auto* restrict dpsi_z_devptr          = psi_vgl.device_data_at(3, 0, 0);
    auto* restrict d2psi_devptr           = psi_vgl.device_data_at(4, 0, 0);


    auto* dr_pbc_devptr = dr_pbc.device_data();
    auto* latR_ptr      = lattice.R.data();

    // build dr_pbc: translation vectors to all images
    // should just do this once and store it (like with phase)
    {
      ScopedTimer local(pbc_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) map(to:latR_ptr[:9]) \
        is_device_ptr(dr_pbc_devptr) ")
      for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
      {
        // is std::div any better than just % and / separately?
        auto div_k  = std::div(i_xyz, Nz);
        int k       = div_k.rem;
        int ij      = div_k.quot;
        auto div_ij = std::div(ij, Ny);
        int j       = div_ij.rem;
        int i       = div_ij.quot;
        int TransX  = ((i % 2) * 2 - 1) * ((i + 1) / 2);
        int TransY  = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        int TransZ  = ((k % 2) * 2 - 1) * ((k + 1) / 2);
        //TODO:
        //  Trans = {TransX,Transy,TransZ};
        //  dr_pbc[i_xyz] = dot(Trans, lattice.R);
        for (size_t i_dim = 0; i_dim < 3; i_dim++)
          dr_pbc_devptr[i_dim + 3 * i_xyz] =
              (TransX * latR_ptr[i_dim + 0 * 3] + TransY * latR_ptr[i_dim + 1 * 3] + TransZ * latR_ptr[i_dim + 2 * 3]);
      }
    }


    auto* correctphase_devptr = correctphase.device_data();
    auto* Tv_list_devptr      = Tv_list.device_data();
    {
      ScopedTimer local_timer(phase_timer_);
#if not defined(QMC_COMPLEX)

      PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(correctphase_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
        correctphase_devptr[i_vp] = 1.0;

#else
      auto* SuperTwist_ptr = SuperTwist.data();

      PRAGMA_OFFLOAD("omp target teams distribute parallel for map(to:SuperTwist_ptr[:9]) \
		    is_device_ptr(Tv_list_devptr, correctphase_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
      {
        //RealType phasearg = dot(3, SuperTwist.data(), 1, Tv_list.data() + 3 * i_vp, 1);
        RealType phasearg = 0;
        for (size_t i_dim = 0; i_dim < 3; i_dim++)
          phasearg += SuperTwist[i_dim] * Tv_list_devptr[i_dim + 3 * i_vp];
        RealType s, c;
        qmcplusplus::sincos(-phasearg, &s, &c);
        correctphase_devptr[i_vp] = ValueType(c, s);
      }
#endif
    }

    auto* dr_new_devptr     = dr.device_data();
    auto* r_new_devptr      = r.device_data();
    auto* displ_list_devptr = displ_list.device_data();
    {
      ScopedTimer local_timer(nelec_pbc_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) \
		    is_device_ptr(dr_new_devptr, dr_pbc_devptr, r_new_devptr, displ_list_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
      {
        for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
        {
          RealType tmp_r2 = 0.0;
          for (size_t i_dim = 0; i_dim < 3; i_dim++)
          {
            dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)] =
                -(displ_list_devptr[i_dim + 3 * (i_vp + c * nElec)] + dr_pbc_devptr[i_dim + 3 * i_xyz]);
            tmp_r2 +=
                dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)] * dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)];
          }
          r_new_devptr[i_xyz + Nxyz * i_vp] = std::sqrt(tmp_r2);
        }
      }
    }
    {
      ScopedTimer local(rnl_timer_);
      MultiRnl.batched_evaluateVGL(r, rnl_vgl, Rmax);
    }
    {
      ScopedTimer local(ylm_timer_);
      Ylm.batched_evaluateVGL(dr, ylm_vgl);
    }


    auto* phase_fac_ptr = periodic_image_phase_factors.data();
    auto* LM_ptr        = LM.data();
    auto* NL_ptr        = NL.data();

    RealType* restrict phi_devptr   = rnl_vgl.device_data_at(0, 0, 0, 0);
    RealType* restrict dphi_devptr  = rnl_vgl.device_data_at(1, 0, 0, 0);
    RealType* restrict d2phi_devptr = rnl_vgl.device_data_at(2, 0, 0, 0);


    const RealType* restrict ylm_v_devptr = ylm_vgl.device_data_at(0, 0, 0, 0); //value
    const RealType* restrict ylm_x_devptr = ylm_vgl.device_data_at(1, 0, 0, 0); //gradX
    const RealType* restrict ylm_y_devptr = ylm_vgl.device_data_at(2, 0, 0, 0); //gradY
    const RealType* restrict ylm_z_devptr = ylm_vgl.device_data_at(3, 0, 0, 0); //gradZ
    const RealType* restrict ylm_l_devptr = ylm_vgl.device_data_at(4, 0, 0, 0); //lap
    {
      ScopedTimer local_timer(psi_timer_);
      PRAGMA_OFFLOAD(
          "omp target teams distribute parallel for collapse(2) map(to:phase_fac_ptr[:Nxyz], LM_ptr[:BasisSetSize], NL_ptr[:BasisSetSize]) \
		       is_device_ptr(ylm_v_devptr, ylm_x_devptr, ylm_y_devptr, ylm_z_devptr, ylm_l_devptr, \
                         phi_devptr, dphi_devptr, d2phi_devptr, \
                         psi_devptr, dpsi_x_devptr, dpsi_y_devptr, dpsi_z_devptr, d2psi_devptr, \
                         correctphase_devptr, r_new_devptr, dr_new_devptr) ")
      for (size_t i_e = 0; i_e < nElec; i_e++)
      {
        for (size_t ib = 0; ib < BasisSetSize; ++ib)
        {
          const int nl(NL_ptr[ib]);
          const int lm(LM_ptr[ib]);
          for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
          {
            const ValueType Phase    = phase_fac_ptr[i_xyz] * correctphase_devptr[i_e];
            const RealType rinv      = cone / r_new_devptr[i_xyz + Nxyz * i_e];
            const RealType x         = dr_new_devptr[0 + 3 * (i_xyz + Nxyz * i_e)];
            const RealType y         = dr_new_devptr[1 + 3 * (i_xyz + Nxyz * i_e)];
            const RealType z         = dr_new_devptr[2 + 3 * (i_xyz + Nxyz * i_e)];
            const RealType drnloverr = rinv * dphi_devptr[nl + nRnl * (i_xyz + Nxyz * i_e)];
            const RealType ang       = ylm_v_devptr[lm + nYlm * (i_xyz + Nxyz * i_e)];
            const RealType gr_x      = drnloverr * x;
            const RealType gr_y      = drnloverr * y;
            const RealType gr_z      = drnloverr * z;
            const RealType ang_x     = ylm_x_devptr[lm + nYlm * (i_xyz + Nxyz * i_e)];
            const RealType ang_y     = ylm_y_devptr[lm + nYlm * (i_xyz + Nxyz * i_e)];
            const RealType ang_z     = ylm_z_devptr[lm + nYlm * (i_xyz + Nxyz * i_e)];
            const RealType vr        = phi_devptr[nl + nRnl * (i_xyz + Nxyz * i_e)];

            psi_devptr[BasisOffset + ib + i_e * nBasTot] += ang * vr * Phase;
            dpsi_x_devptr[BasisOffset + ib + i_e * nBasTot] += (ang * gr_x + vr * ang_x) * Phase;
            dpsi_y_devptr[BasisOffset + ib + i_e * nBasTot] += (ang * gr_y + vr * ang_y) * Phase;
            dpsi_z_devptr[BasisOffset + ib + i_e * nBasTot] += (ang * gr_z + vr * ang_z) * Phase;
            d2psi_devptr[BasisOffset + ib + i_e * nBasTot] +=
                (ang * (ctwo * drnloverr + d2phi_devptr[nl + nRnl * (i_xyz + Nxyz * i_e)]) +
                 ctwo * (gr_x * ang_x + gr_y * ang_y + gr_z * ang_z) +
                 vr * ylm_l_devptr[lm + nYlm * (i_xyz + Nxyz * i_e)]) *
                Phase;
          }
        }
      }
    }
    // for (size_t i_e = 0; i_e < nElec; i_e++)
    // {
    //   for (size_t ib = 0; ib < BasisSetSize; ++ib)
    //   {
    //     const int nl(NL_ptr[ib]);
    //     const int lm(LM_ptr[ib]);
    //     for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
    //     {
    //       auto div_k  = std::div(i_xyz, Nz);
    //       int k       = div_k.rem;
    //       int ij      = div_k.quot;
    //       auto div_ij = std::div(ij, Ny);
    //       int j       = div_ij.rem;
    //       int i       = div_ij.quot;
    //       int TransX  = ((i % 2) * 2 - 1) * ((i + 1) / 2);
    //       int TransY  = ((j % 2) * 2 - 1) * ((j + 1) / 2);
    //       int TransZ  = ((k % 2) * 2 - 1) * ((k + 1) / 2);
    //       std::cout << i << j << k;
    //       rnl_vgl[0, i_e, i_xyz, nl] << rnl_vgl[1, i_e, i_xyz, nl] << rnl_vgl[2, i_e, i_xyz, nl]
    //     }
    //   }
    // }
  }
  template<typename LAT, typename VT>
  inline void mw_evaluateV(const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list,
                           const LAT& lattice,
                           Array<VT, 2, OffloadPinnedAllocator<VT>>& psi,
                           const Vector<RealType, OffloadPinnedAllocator<RealType>>& displ_list,
                           const Vector<RealType, OffloadPinnedAllocator<RealType>>& Tv_list,
                           const size_t nElec,
                           const size_t nBasTot,
                           const size_t c,
                           const size_t BasisOffset,
                           const size_t NumCenters)
  {
    assert(this == &atom_bs_list.getLeader());
    auto& atom_bs_leader = atom_bs_list.template getCastedLeader<SoaAtomicBasisSet<ROT, SH>>();
    /*
      psi [nElec, nBasTot] (start at [0, BasisOffset])
      displ_list [3 * nElec * NumCenters] (start at [3*nElec*c])
      Tv_list [3 * nElec * NumCenters]
    */
    //TODO: use QMCTraits::DIM instead of 3?
    int Nx   = PBCImages[0] + 1;
    int Ny   = PBCImages[1] + 1;
    int Nz   = PBCImages[2] + 1;
    int Nyz  = Ny * Nz;
    int Nxyz = Nx * Nyz;
    assert(psi.size(0) == nElec);
    assert(psi.size(1) == nBasTot);


    auto& ylm_v = atom_bs_leader.mw_mem_handle_.getResource().ylm_v;
    auto& rnl_v = atom_bs_leader.mw_mem_handle_.getResource().rnl_v;
    auto& dr    = atom_bs_leader.mw_mem_handle_.getResource().dr;
    auto& r     = atom_bs_leader.mw_mem_handle_.getResource().r;

    size_t nRnl = RnlID.size();
    size_t nYlm = Ylm.size();

    ylm_v.resize(nElec, Nxyz, nYlm);
    rnl_v.resize(nElec, Nxyz, nRnl);
    dr.resize(nElec, Nxyz, 3);
    r.resize(nElec, Nxyz);

    // TODO: move these outside?
    auto& dr_pbc       = atom_bs_leader.mw_mem_handle_.getResource().dr_pbc;
    auto& correctphase = atom_bs_leader.mw_mem_handle_.getResource().correctphase;
    dr_pbc.resize(Nxyz, 3);
    correctphase.resize(nElec);


    auto* dr_pbc_ptr    = dr_pbc.data();
    auto* dr_pbc_devptr = dr_pbc.device_data();

    auto* dr_new_ptr    = dr.data();
    auto* dr_new_devptr = dr.device_data();

    auto* r_new_ptr    = r.data();
    auto* r_new_devptr = r.device_data();

    // how to map Tensor<T,3> to device?
    auto* latR_ptr = lattice.R.data();

    auto* psi_devptr = psi.device_data();

    auto* correctphase_devptr = correctphase.device_data();

    auto* Tv_list_devptr    = Tv_list.device_data();
    auto* displ_list_devptr = displ_list.device_data();

    // build dr_pbc: translation vectors to all images
    // should just do this once and store it (like with phase)
    {
      ScopedTimer local(pbc_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) map(to:latR_ptr[:9]) \
        is_device_ptr(dr_pbc_devptr) ")
      for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
      {
        // is std::div any better than just % and / separately?
        auto div_k  = std::div(i_xyz, Nz);
        int k       = div_k.rem;
        int ij      = div_k.quot;
        auto div_ij = std::div(ij, Ny);
        int j       = div_ij.rem;
        int i       = div_ij.quot;
        int TransX  = ((i % 2) * 2 - 1) * ((i + 1) / 2);
        int TransY  = ((j % 2) * 2 - 1) * ((j + 1) / 2);
        int TransZ  = ((k % 2) * 2 - 1) * ((k + 1) / 2);
        //TODO:
        //  Trans = {TransX,Transy,TransZ};
        //  dr_pbc[i_xyz] = dot(Trans, lattice.R);
        for (size_t i_dim = 0; i_dim < 3; i_dim++)
          dr_pbc_devptr[i_dim + 3 * i_xyz] =
              (TransX * latR_ptr[i_dim + 0 * 3] + TransY * latR_ptr[i_dim + 1 * 3] + TransZ * latR_ptr[i_dim + 2 * 3]);
        //(TransX * lattice.R(0, i_dim) + TransY * lattice.R(1, i_dim) + TransZ * lattice.R(2, i_dim));
      }
    }


    {
      ScopedTimer local_timer(phase_timer_);
#if not defined(QMC_COMPLEX)

      PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(correctphase_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
        correctphase_devptr[i_vp] = 1.0;

#else
      auto* SuperTwist_ptr = SuperTwist.data();

      PRAGMA_OFFLOAD("omp target teams distribute parallel for map(to:SuperTwist_ptr[:9]) \
		    is_device_ptr(Tv_list_devptr, correctphase_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
      {
        //RealType phasearg = dot(3, SuperTwist.data(), 1, Tv_list.data() + 3 * i_vp, 1);
        RealType phasearg = 0;
        for (size_t i_dim = 0; i_dim < 3; i_dim++)
          phasearg += SuperTwist[i_dim] * Tv_list_devptr[i_dim + 3 * i_vp];
        RealType s, c;
        qmcplusplus::sincos(-phasearg, &s, &c);
        correctphase_devptr[i_vp] = ValueType(c, s);
      }
#endif
    }

    {
      ScopedTimer local_timer(nelec_pbc_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) \
		    is_device_ptr(dr_new_devptr, dr_pbc_devptr, r_new_devptr, displ_list_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
      {
        for (int i_xyz = 0; i_xyz < Nxyz; i_xyz++)
        {
          RealType tmp_r2 = 0.0;
          for (size_t i_dim = 0; i_dim < 3; i_dim++)
          {
            dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)] =
                -(displ_list_devptr[i_dim + 3 * (i_vp + c * nElec)] + dr_pbc_devptr[i_dim + 3 * i_xyz]);
            tmp_r2 +=
                dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)] * dr_new_devptr[i_dim + 3 * (i_xyz + Nxyz * i_vp)];
          }
          r_new_devptr[i_xyz + Nxyz * i_vp] = std::sqrt(tmp_r2);
        }
      }
    }

    // TODO: remove this when Rnl/Ylm offloaded
    // r.updateFrom();
    // dr.updateFrom();
    // TODO: manage this better? set in builder?
    // maybe just do one image at a time?
    // size_t tmpSize = std::max(Ylm.size(), RnlID.size());
    // tempS.resize(nElec * Nxyz * tmpSize);

    // RealType* restrict ylm_v = tempS.data(0);
    // RealType* restrict phi_r = tempS.data(1);

    // auto* ylm_devptr = tempS.device_data(0);
    // auto* rnl_devptr = tempS.device_data(1);


    // ylm_v and phi_r are nElec * Nxyz * tmpSize
    // tmpSize is max(Ylm.size(), RnlID.size())
    {
      ScopedTimer local(rnl_timer_);
      //MultiRnl.mw_evaluate(r.data(), rnl_v.data(), Nxyz * nElec, nRnl, Rmax);
      MultiRnl.batched_evaluate(r, rnl_v, Rmax);

      // serial?
      // for (size_t i_r = 0; i_r < Nxyz * NElec; i_r++)
      // {
      //   MultiRnl.evaluate_single_offload()
      // }
    }
    // dr_new is [3 * Nxyz * nElec] realtype
    {
      ScopedTimer local(ylm_timer_);
      //Ylm.mw_evaluateV(dr.data(), ylm_v.data(), Nxyz * nElec, nYlm);
      Ylm.batched_evaluateV(dr, ylm_v);
    }
    ///Phase for PBC containing the phase for the nearest image displacement and the correction due to the Distance table.

    //TODO: remove this when Ylm/MultiRnl mw_evaluateV are offloaded
    // tempS.updateTo();
    // ylm_v.updateTo();
    // rnl_v.updateTo();

    auto* phase_fac_ptr = periodic_image_phase_factors.data();
    auto* LM_ptr        = LM.data();
    auto* NL_ptr        = NL.data();
    auto* psi_ptr       = psi.data();

    auto* ylm_devptr = ylm_v.device_data();
    auto* rnl_devptr = rnl_v.device_data();
    {
      ScopedTimer local_timer(psi_timer_);
      PRAGMA_OFFLOAD(
          "omp target teams distribute parallel for collapse(2) map(to:phase_fac_ptr[:Nxyz], LM_ptr[:BasisSetSize], NL_ptr[:BasisSetSize]) \
		    is_device_ptr(ylm_devptr, rnl_devptr, psi_devptr, correctphase_devptr) ")
      for (size_t i_vp = 0; i_vp < nElec; i_vp++)
      {
        for (size_t ib = 0; ib < BasisSetSize; ++ib)
        {
          for (size_t i_xyz = 0; i_xyz < Nxyz; i_xyz++)
          {
            const ValueType Phase = phase_fac_ptr[i_xyz] * correctphase_devptr[i_vp];
            psi_devptr[BasisOffset + ib + i_vp * nBasTot] += ylm_devptr[(i_xyz + Nxyz * i_vp) * nYlm + LM_ptr[ib]] *
                rnl_devptr[(i_xyz + Nxyz * i_vp) * nRnl + NL_ptr[ib]] * Phase;
            //const ValueType Phase = periodic_image_phase_factors[i_xyz] * correctphase[i_vp];
            //psi(i_vp, BasisOffset + ib) +=
            //psi_ptr[BasisOffset + ib + i_vp*nBasTot] +=
            //    ylm_v[(i_xyz + Nxyz * i_vp) * tmpSize + LM[ib]] * phi_r[(i_xyz + Nxyz * i_vp) * tmpSize + NL[ib]] * Phase;
          }
        }
      }
    }
  }


  void createResource(ResourceCollection& collection) const
  {
    // Ylm.createResource(collection);
    // MultiRnl.createResource(collection);
    auto resource_index = collection.addResource(std::make_unique<SoaAtomicBSetMultiWalkerMem>());
  }

  void acquireResource(ResourceCollection& collection, const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list) const
  {
    assert(this == &atom_bs_list.getLeader());
    // auto& atom_bs_leader          = atom_bs_list.getCastedLeader<SoaAtomicBasisSet>();
    // auto& atom_bs_leader          = atom_bs_list.getCastedLeader();
    // SoaAtomicBasisSet& atom_bs_leader          = atom_bs_list.getCastedLeader();
    // const SoaAtomicBasisSet& atom_bs_leader          = atom_bs_list.getCastedLeader();
    // const auto ylm_list(extractYlmRefList(atom_bs_list));
    auto& atom_bs_leader          = atom_bs_list.template getCastedLeader<SoaAtomicBasisSet>();
    atom_bs_leader.mw_mem_handle_ = collection.lendResource<SoaAtomicBSetMultiWalkerMem>();
  }

  void releaseResource(ResourceCollection& collection, const RefVectorWithLeader<SoaAtomicBasisSet>& atom_bs_list) const
  {
    assert(this == &atom_bs_list.getLeader());
    // auto& atom_bs_leader = atom_bs_list.getCastedLeader();
    // const SoaAtomicBasisSet& atom_bs_leader          = atom_bs_list.getCastedLeader();
    auto& atom_bs_leader = atom_bs_list.template getCastedLeader<SoaAtomicBasisSet>();
    collection.takebackResource(atom_bs_leader.mw_mem_handle_);
  }

  struct SoaAtomicBSetMultiWalkerMem : public Resource
  {
    SoaAtomicBSetMultiWalkerMem() : Resource("SoaAtomicBasisSet") {}

    SoaAtomicBSetMultiWalkerMem(const SoaAtomicBSetMultiWalkerMem&) : SoaAtomicBSetMultiWalkerMem() {}

    std::unique_ptr<Resource> makeClone() const override
    {
      return std::make_unique<SoaAtomicBSetMultiWalkerMem>(*this);
    }

    OffloadArray3D ylm_v;       // [Nelec][PBC][NYlm]
    OffloadArray3D rnl_v;       // [Nelec][PBC][NRnl]
    OffloadArray4D ylm_vgl;     // [5][Nelec][PBC][NYlm]
    OffloadArray4D rnl_vgl;     // [5][Nelec][PBC][NRnl]
    OffloadArray2D dr_pbc;      // [PBC][xyz]
    OffloadArray3D dr;          // [Nelec][PBC][xyz]
    OffloadArray2D r;           // [Nelec][PBC]
    OffloadVector correctphase; // [Nelec]
  };
};
} // namespace qmcplusplus
#endif
