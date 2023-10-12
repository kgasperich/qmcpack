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


#ifndef QMCPLUSPLUS_SOA_SPHERICAL_CARTESIAN_TENSOR_H
#define QMCPLUSPLUS_SOA_SPHERICAL_CARTESIAN_TENSOR_H

#include <stdexcept>
#include <limits>
#include "OhmmsSoA/VectorSoaContainer.h"
#include "OhmmsPETE/Tensor.h"
#include "OhmmsPETE/OhmmsArray.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"

namespace qmcplusplus
{
/** SoaSphericalTensor that evaluates the Real Spherical Harmonics
 *
 * The template parameters
 * - T, the value_type, e.g. double
 * - Point_t, a vector type to provide xyz coordinate.
 * Point_t must have the operator[] defined, e.g., TinyVector\<double,3\>.
 *
 * Real Spherical Harmonics Ylm\f$=r^l S_l^m(x,y,z) \f$ is stored
 * in an array ordered as [0,-1 0 1,-2 -1 0 1 2, -Lmax,-Lmax+1,..., Lmax-1,Lmax]
 * where Lmax is the maximum angular momentum of a center.
 * All the data members, e.g, Ylm and pre-calculated factors,
 * can be accessed by index(l,m) which returns the
 * locator of the combination for l and m.
 */
template<typename T>
struct SoaSphericalTensor
{
  ///maximum angular momentum for the center
  int Lmax;
  /// Normalization factors
  aligned_vector<T> NormFactor;
  ///pre-evaluated factor \f$1/\sqrt{(l+m)\times(l+1-m)}\f$
  aligned_vector<T> FactorLM;
  ///pre-evaluated factor \f$\sqrt{(2l+1)/(4\pi)}\f$
  aligned_vector<T> FactorL;
  ///pre-evaluated factor \f$(2l+1)/(2l-1)\f$
  aligned_vector<T> Factor2L;
  ///composite
  VectorSoaContainer<T, 5> cYlm;
  using xyz_type       = TinyVector<T, 3>;
  using OffloadArray2D = Array<T, 2, OffloadPinnedAllocator<T>>;
  using OffloadArray3D = Array<T, 3, OffloadPinnedAllocator<T>>;
  using OffloadArray4D = Array<T, 4, OffloadPinnedAllocator<T>>;

  explicit SoaSphericalTensor(const int l_max, bool addsign = false);

  SoaSphericalTensor(const SoaSphericalTensor& rhs) = default;

  ///compute Ylm
  void evaluate_bare(T x, T y, T z, T* Ylm) const;
  static void evaluate_bare_impl(T x, T y, T z, T* Ylm, const size_t Lmax_, const T* FacL, const T* FacLM);
  static void evaluateVGL_impl(const T x,
                               const T y,
                               const T z,
                               T* restrict Ylm_vgl,
                               const size_t Lmax_,
                               const T* FactorL_ptr,
                               const T* FactorLM_ptr,
                               const T* Factor2L_ptr,
                               const T* NormFactor_ptr,
                               const size_t offset);

  ///compute Ylm
  inline void evaluateV(T x, T y, T z, T* Ylm) const
  {
    evaluate_bare(x, y, z, Ylm);
    for (int i = 0, nl = cYlm.size(); i < nl; i++)
      Ylm[i] *= NormFactor[i];
  }

  ///compute Ylm
  //inline void evaluateV_batch(T* xyz, T* Ylm, size_t nr, size_t nlm) const
  inline void mw_evaluateV(T* xyz, T* Ylm, size_t nr, size_t nlm) const
  {
    for (size_t ir = 0; ir < nr; ir++)
      evaluate_bare(xyz[0 + 3 * ir], xyz[1 + 3 * ir], xyz[2 + 3 * ir], Ylm + (ir * nlm));
    //evaluate_bare_impl(FacL_ptr, FacLM_ptr, xyz[0 + 3 * ir], xyz[1 + 3 * ir], xyz[2 + 3 * ir], Ylm + (ir * nlm));
    for (size_t ir = 0; ir < nr; ir++)
      for (int i = 0, nl = cYlm.size(); i < nl; i++)
        Ylm[ir * nlm + i] *= NormFactor[i];
  }

  /**
   * @brief evaluate VGL for multiple electrons and multiple pbc images
   * 
   * @param [in] xyz electron positions [Nelec, Npbc, 3(x,y,z)]
   * @param [out] Ylm Spherical tensor elements [Nelec, Npbc, Nlm]
  */
  inline void batched_evaluateV(OffloadArray3D& xyz, OffloadArray3D& Ylm) const
  {
    const size_t nElec = xyz.size(0);
    const size_t Npbc  = xyz.size(1); // number of PBC images
    assert(xyz.size(2) == 3);

    assert(Ylm.size(0) == nElec);
    assert(Ylm.size(1) == Npbc);
    const size_t Nlm = Ylm.size(2);

    size_t nR = nElec * Npbc; // total number of positions to evaluate

    auto* xyz_devptr     = xyz.device_data();
    auto* Ylm_devptr     = Ylm.device_data();
    auto* flm_ptr        = FactorLM.data();
    auto* fl_ptr         = FactorL.data();
    auto* NormFactor_ptr = NormFactor.data();


    PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                    map(to:flm_ptr[:Nlm], fl_ptr[:Lmax+1], NormFactor_ptr[:Nlm]) \
                    is_device_ptr(xyz_devptr, Ylm_devptr)")
    for (size_t ir = 0; ir < nR; ir++)
    {
      evaluate_bare_impl(xyz_devptr[0 + 3 * ir], xyz_devptr[1 + 3 * ir], xyz_devptr[2 + 3 * ir],
                         Ylm_devptr + (ir * Nlm), Lmax, fl_ptr, flm_ptr);
      for (int i = 0; i < Nlm; i++)
        Ylm_devptr[ir * Nlm + i] *= NormFactor_ptr[i];
    }
  }

  /**
   * @brief evaluate VGL for multiple electrons and multiple pbc images
   * 
   * @param [in] xyz electron positions [Nelec, Npbc, 3(x,y,z)]
   * @param [out] Ylm_vgl Spherical tensor elements [5(v, gx, gy, gz, lapl), Nelec, Npbc, Nlm]
  */
  inline void batched_evaluateVGL(OffloadArray3D& xyz, OffloadArray4D& Ylm_vgl) const
  {
    const size_t nElec = xyz.size(0);
    const size_t Npbc  = xyz.size(1); // number of PBC images
    assert(xyz.size(2) == 3);

    assert(Ylm_vgl.size(0) == 5);
    assert(Ylm_vgl.size(1) == nElec);
    assert(Ylm_vgl.size(2) == Npbc);
    const size_t Nlm = Ylm_vgl.size(3);
    assert(NormFactor.size() == Nlm);

    size_t nR     = nElec * Npbc; // total number of positions to evaluate
    size_t offset = Nlm * nR;     // stride for v/gx/gy/gz/l

    auto* xyz_devptr     = xyz.device_data();
    auto* Ylm_vgl_devptr = Ylm_vgl.device_data();
    auto* flm_ptr        = FactorLM.data();
    auto* fl_ptr         = FactorL.data();
    auto* f2l_ptr        = Factor2L.data();
    auto* NormFactor_ptr = NormFactor.data();


    PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                    map(to:flm_ptr[:Nlm], fl_ptr[:Lmax+1], NormFactor_ptr[:Nlm], f2l_ptr[:Lmax+1]) \
                    is_device_ptr(xyz_devptr, Ylm_devptr)")
    for (size_t ir = 0; ir < nR; ir++)
      evaluateVGL_impl(xyz_devptr[0 + 3 * ir], xyz_devptr[1 + 3 * ir], xyz_devptr[2 + 3 * ir],
                       Ylm_vgl_devptr + (ir * Nlm), Lmax, fl_ptr, flm_ptr, f2l_ptr, NormFactor_ptr, offset);
  }

  ///compute Ylm
  inline void evaluateV(T x, T y, T z)
  {
    T* restrict Ylm = cYlm.data(0);
    evaluate_bare(x, y, z, Ylm);
    for (int i = 0, nl = cYlm.size(); i < nl; i++)
      Ylm[i] *= NormFactor[i];
  }

  ///makes a table of \f$ r^l S_l^m \f$ and their gradients up to Lmax.
  void evaluateVGL(T x, T y, T z);

  ///makes a table of \f$ r^l S_l^m \f$ and their gradients up to Lmax.
  void evaluateVGH(T x, T y, T z);

  ///makes a table of \f$ r^l S_l^m \f$ and their gradients up to Lmax.
  void evaluateVGHGH(T x, T y, T z);

  ///returns the index/locator for (\f$l,m\f$) combo, \f$ l(l+1)+m \f$
  static inline int index(int l, int m) { return (l * (l + 1)) + m; }

  /** return the starting address of the component
   *
   * component=0(V), 1(dx), 2(dy), 3(dz), 4(Lap)
   */
  inline const T* operator[](size_t component) const { return cYlm.data(component); }

  inline size_t size() const { return cYlm.size(); }

  inline int lmax() const { return Lmax; }
};

/** constructor
 * @param l_max maximum angular momentum
 * @param addsign flag to determine what convention to use
 *
 * Evaluate all the constants and prefactors.
 * The spherical harmonics is defined as
 * \f[ Y_l^m (\theta,\phi) = \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}} P_l^m(\cos\theta)e^{im\phi}\f]
 * Note that the data member Ylm is a misnomer and should not be confused with "spherical harmonics"
 * \f$Y_l^m\f$.
 - When addsign == true, e.g., Gaussian packages
 \f{eqnarray*}
 S_l^m &=& (-1)^m \sqrt{2}\Re(Y_l^{|m|}), \;\;\;m > 0 \\
 &=& Y_l^0, \;\;\;m = 0 \\
 &=& (-1)^m \sqrt{2}\Im(Y_l^{|m|}),\;\;\;m < 0
 \f}
 - When addsign == false, e.g., SIESTA package,
 \f{eqnarray*}
 S_l^m &=& \sqrt{2}\Re(Y_l^{|m|}), \;\;\;m > 0 \\
 &=& Y_l^0, \;\;\;m = 0 \\
 &=&\sqrt{2}\Im(Y_l^{|m|}),\;\;\;m < 0
 \f}
 */
template<typename T>
inline SoaSphericalTensor<T>::SoaSphericalTensor(const int l_max, bool addsign) : Lmax(l_max)
{
  constexpr T czero(0);
  constexpr T cone(1);
  const int ntot = (Lmax + 1) * (Lmax + 1);
  cYlm.resize(ntot);
  cYlm = czero;
  NormFactor.resize(ntot, cone);
  const T sqrt2 = std::sqrt(2.0);
  if (addsign)
  {
    for (int l = 0; l <= Lmax; l++)
    {
      NormFactor[index(l, 0)] = cone;
      for (int m = 1; m <= l; m++)
      {
        NormFactor[index(l, m)]  = std::pow(-cone, m) * sqrt2;
        NormFactor[index(l, -m)] = std::pow(-cone, -m) * sqrt2;
      }
    }
  }
  else
  {
    for (int l = 0; l <= Lmax; l++)
    {
      for (int m = 1; m <= l; m++)
      {
        NormFactor[index(l, m)]  = sqrt2;
        NormFactor[index(l, -m)] = sqrt2;
      }
    }
  }
  FactorL.resize(Lmax + 1);
  const T omega = 1.0 / std::sqrt(16.0 * std::atan(1.0));
  for (int l = 1; l <= Lmax; l++)
    FactorL[l] = std::sqrt(static_cast<T>(2 * l + 1)) * omega;
  Factor2L.resize(Lmax + 1);
  for (int l = 1; l <= Lmax; l++)
    Factor2L[l] = static_cast<T>(2 * l + 1) / static_cast<T>(2 * l - 1);
  FactorLM.resize(ntot);
  for (int l = 1; l <= Lmax; l++)
    for (int m = 1; m <= l; m++)
    {
      T fac2                 = 1.0 / std::sqrt(static_cast<T>((l + m) * (l + 1 - m)));
      FactorLM[index(l, m)]  = fac2;
      FactorLM[index(l, -m)] = fac2;
    }
}


PRAGMA_OFFLOAD("omp declare target")
template<typename T>
inline void SoaSphericalTensor<T>::evaluate_bare_impl(const T x,
                                                      const T y,
                                                      const T z,
                                                      T* restrict Ylm,
                                                      const size_t Lmax_,
                                                      const T* FactorL_ptr,
                                                      const T* FactorLM_ptr)
{
  constexpr T czero(0);
  constexpr T cone(1);
  const T pi       = 4.0 * std::atan(1.0);
  const T omega    = 1.0 / std::sqrt(4.0 * pi);
  constexpr T eps2 = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();

  /*  Calculate r, cos(theta), sin(theta), cos(phi), sin(phi) from input
      coordinates. Check here the coordinate singularity at cos(theta) = +-1.
      This also takes care of r=0 case. */
  T cphi, sphi, ctheta;
  T r2xy = x * x + y * y;
  T r    = std::sqrt(r2xy + z * z);
  if (r2xy < eps2)
  {
    cphi   = czero;
    sphi   = cone;
    ctheta = (z < czero) ? -cone : cone;
  }
  else
  {
    ctheta = z / r;
    //protect ctheta, when ctheta is slightly >1 or <-1
    if (ctheta > cone)
      ctheta = cone;
    if (ctheta < -cone)
      ctheta = -cone;
    T rxyi = cone / std::sqrt(r2xy);
    cphi   = x * rxyi;
    sphi   = y * rxyi;
  }
  T stheta = std::sqrt(cone - ctheta * ctheta);
  /* Now to calculate the associated legendre functions P_lm from the
     recursion relation from l=0 to Lmax. Conventions of J.D. Jackson,
     Classical Electrodynamics are used. */
  Ylm[0] = cone;
  // calculate P_ll and P_l,l-1
  T fac = cone;
  int j = -1;
  for (int l = 1; l <= Lmax_; l++)
  {
    j += 2;
    fac *= -j * stheta;
    int ll  = index(l, l);
    int l1  = index(l, l - 1);
    int l2  = index(l - 1, l - 1);
    Ylm[ll] = fac;
    Ylm[l1] = j * ctheta * Ylm[l2];
  }
  // Use recurence to get other plm's //
  for (int m = 0; m < Lmax_ - 1; m++)
  {
    int j = 2 * m + 1;
    for (int l = m + 2; l <= Lmax_; l++)
    {
      j += 2;
      int lm  = index(l, m);
      int l1  = index(l - 1, m);
      int l2  = index(l - 2, m);
      Ylm[lm] = (ctheta * j * Ylm[l1] - (l + m - 1) * Ylm[l2]) / (l - m);
    }
  }
  // Now to calculate r^l Y_lm. //
  T sphim, cphim, temp;
  Ylm[0] = omega; //1.0/sqrt(pi4);
  T rpow = 1.0;
  for (int l = 1; l <= Lmax_; l++)
  {
    rpow *= r;
    //fac = rpow*sqrt(static_cast<T>(2*l+1))*omega;//rpow*sqrt((2*l+1)/pi4);
    //FactorL[l] = sqrt(2*l+1)/sqrt(4*pi)
    fac    = rpow * FactorL_ptr[l];
    int l0 = index(l, 0);
    Ylm[l0] *= fac;
    cphim = cone;
    sphim = czero;
    for (int m = 1; m <= l; m++)
    {
      temp   = cphim * cphi - sphim * sphi;
      sphim  = sphim * cphi + cphim * sphi;
      cphim  = temp;
      int lm = index(l, m);
      fac *= FactorLM_ptr[lm];
      temp    = fac * Ylm[lm];
      Ylm[lm] = temp * cphim;
      lm      = index(l, -m);
      Ylm[lm] = temp * sphim;
    }
  }
  //for (int i=0; i<Ylm.size(); i++)
  //  Ylm[i]*= NormFactor[i];
}
PRAGMA_OFFLOAD("omp end declare target")

template<typename T>
inline void SoaSphericalTensor<T>::evaluate_bare(T x, T y, T z, T* restrict Ylm) const
{
  constexpr T czero(0);
  constexpr T cone(1);
  const T pi       = 4.0 * std::atan(1.0);
  const T omega    = 1.0 / std::sqrt(4.0 * pi);
  constexpr T eps2 = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();

  /*  Calculate r, cos(theta), sin(theta), cos(phi), sin(phi) from input
      coordinates. Check here the coordinate singularity at cos(theta) = +-1.
      This also takes care of r=0 case. */
  T cphi, sphi, ctheta;
  T r2xy = x * x + y * y;
  T r    = std::sqrt(r2xy + z * z);
  if (r2xy < eps2)
  {
    cphi   = czero;
    sphi   = cone;
    ctheta = (z < czero) ? -cone : cone;
  }
  else
  {
    ctheta = z / r;
    //protect ctheta, when ctheta is slightly >1 or <-1
    if (ctheta > cone)
      ctheta = cone;
    if (ctheta < -cone)
      ctheta = -cone;
    T rxyi = cone / std::sqrt(r2xy);
    cphi   = x * rxyi;
    sphi   = y * rxyi;
  }
  T stheta = std::sqrt(cone - ctheta * ctheta);
  /* Now to calculate the associated legendre functions P_lm from the
     recursion relation from l=0 to Lmax. Conventions of J.D. Jackson,
     Classical Electrodynamics are used. */
  Ylm[0] = cone;
  // calculate P_ll and P_l,l-1
  T fac = cone;
  int j = -1;
  for (int l = 1; l <= Lmax; l++)
  {
    j += 2;
    fac *= -j * stheta;
    int ll  = index(l, l);
    int l1  = index(l, l - 1);
    int l2  = index(l - 1, l - 1);
    Ylm[ll] = fac;
    Ylm[l1] = j * ctheta * Ylm[l2];
  }
  // Use recurence to get other plm's //
  for (int m = 0; m < Lmax - 1; m++)
  {
    int j = 2 * m + 1;
    for (int l = m + 2; l <= Lmax; l++)
    {
      j += 2;
      int lm  = index(l, m);
      int l1  = index(l - 1, m);
      int l2  = index(l - 2, m);
      Ylm[lm] = (ctheta * j * Ylm[l1] - (l + m - 1) * Ylm[l2]) / (l - m);
    }
  }
  // Now to calculate r^l Y_lm. //
  T sphim, cphim, temp;
  Ylm[0] = omega; //1.0/sqrt(pi4);
  T rpow = 1.0;
  for (int l = 1; l <= Lmax; l++)
  {
    rpow *= r;
    //fac = rpow*sqrt(static_cast<T>(2*l+1))*omega;//rpow*sqrt((2*l+1)/pi4);
    //FactorL[l] = sqrt(2*l+1)/sqrt(4*pi)
    fac    = rpow * FactorL[l];
    int l0 = index(l, 0);
    Ylm[l0] *= fac;
    cphim = cone;
    sphim = czero;
    for (int m = 1; m <= l; m++)
    {
      temp   = cphim * cphi - sphim * sphi;
      sphim  = sphim * cphi + cphim * sphi;
      cphim  = temp;
      int lm = index(l, m);
      fac *= FactorLM[lm];
      temp    = fac * Ylm[lm];
      Ylm[lm] = temp * cphim;
      lm      = index(l, -m);
      Ylm[lm] = temp * sphim;
    }
  }
  //for (int i=0; i<Ylm.size(); i++)
  //  Ylm[i]*= NormFactor[i];
}

PRAGMA_OFFLOAD("omp declare target")
template<typename T>
inline void SoaSphericalTensor<T>::evaluateVGL_impl(const T x,
                                                    const T y,
                                                    const T z,
                                                    T* restrict Ylm_vgl,
                                                    const size_t Lmax_,
                                                    const T* FactorL_ptr,
                                                    const T* FactorLM_ptr,
                                                    const T* Factor2L_ptr,
                                                    const T* NormFactor_ptr,
                                                    const size_t offset)
{
  T* restrict Ylm = Ylm_vgl;
  // T* restrict Ylm = cYlm.data(0);
  evaluate_bare_impl(x, y, z, Ylm, Lmax_, FactorL_ptr, FactorLM_ptr);
  const size_t Nlm = (Lmax_ + 1) * (Lmax_ + 1);

  constexpr T czero(0);
  constexpr T ahalf(0.5);
  T* restrict gYlmX = Ylm_vgl + offset * 1;
  T* restrict gYlmY = Ylm_vgl + offset * 2;
  T* restrict gYlmZ = Ylm_vgl + offset * 3;
  T* restrict lYlm  = Ylm_vgl + offset * 4; // just need to set to zero

  // Calculating Gradient now//
  for (int l = 1; l <= Lmax_; l++)
  {
    //T fac = ((T) (2*l+1))/(2*l-1);
    T fac = Factor2L_ptr[l];
    for (int m = -l; m <= l; m++)
    {
      int lm = index(l - 1, 0);
      T gx, gy, gz, dpr, dpi, dmr, dmi;
      const int ma = std::abs(m);
      const T cp   = std::sqrt(fac * (l - ma - 1) * (l - ma));
      const T cm   = std::sqrt(fac * (l + ma - 1) * (l + ma));
      const T c0   = std::sqrt(fac * (l - ma) * (l + ma));
      gz           = (l > ma) ? c0 * Ylm[lm + m] : czero;
      if (l > ma + 1)
      {
        dpr = cp * Ylm[lm + ma + 1];
        dpi = cp * Ylm[lm - ma - 1];
      }
      else
      {
        dpr = czero;
        dpi = czero;
      }
      if (l > 1)
      {
        switch (ma)
        {
        case 0:
          dmr = -cm * Ylm[lm + 1];
          dmi = cm * Ylm[lm - 1];
          break;
        case 1:
          dmr = cm * Ylm[lm];
          dmi = czero;
          break;
        default:
          dmr = cm * Ylm[lm + ma - 1];
          dmi = cm * Ylm[lm - ma + 1];
        }
      }
      else
      {
        dmr = cm * Ylm[lm];
        dmi = czero;
        //dmr = (l==1) ? cm*Ylm[lm]:0.0;
        //dmi = 0.0;
      }
      if (m < 0)
      {
        gx = ahalf * (dpi - dmi);
        gy = -ahalf * (dpr + dmr);
      }
      else
      {
        gx = ahalf * (dpr - dmr);
        gy = ahalf * (dpi + dmi);
      }
      lm = index(l, m);
      if (ma)
      {
        gYlmX[lm] = NormFactor_ptr[lm] * gx;
        gYlmY[lm] = NormFactor_ptr[lm] * gy;
        gYlmZ[lm] = NormFactor_ptr[lm] * gz;
      }
      else
      {
        gYlmX[lm] = gx;
        gYlmY[lm] = gy;
        gYlmZ[lm] = gz;
      }
    }
  }
  for (int i = 0; i < Nlm; i++)
  {
    Ylm[i] *= NormFactor_ptr[i];
    lYlm[i] = 0;
  }
  //for (int i=0; i<Ylm.size(); i++) gradYlm[i]*= NormFactor[i];
}
PRAGMA_OFFLOAD("omp end declare target")

template<typename T>
inline void SoaSphericalTensor<T>::evaluateVGL(T x, T y, T z)
{
  T* restrict Ylm = cYlm.data(0);
  evaluate_bare(x, y, z, Ylm);

  constexpr T czero(0);
  constexpr T ahalf(0.5);
  T* restrict gYlmX = cYlm.data(1);
  T* restrict gYlmY = cYlm.data(2);
  T* restrict gYlmZ = cYlm.data(3);

  // Calculating Gradient now//
  for (int l = 1; l <= Lmax; l++)
  {
    //T fac = ((T) (2*l+1))/(2*l-1);
    T fac = Factor2L[l];
    for (int m = -l; m <= l; m++)
    {
      int lm = index(l - 1, 0);
      T gx, gy, gz, dpr, dpi, dmr, dmi;
      const int ma = std::abs(m);
      const T cp   = std::sqrt(fac * (l - ma - 1) * (l - ma));
      const T cm   = std::sqrt(fac * (l + ma - 1) * (l + ma));
      const T c0   = std::sqrt(fac * (l - ma) * (l + ma));
      gz           = (l > ma) ? c0 * Ylm[lm + m] : czero;
      if (l > ma + 1)
      {
        dpr = cp * Ylm[lm + ma + 1];
        dpi = cp * Ylm[lm - ma - 1];
      }
      else
      {
        dpr = czero;
        dpi = czero;
      }
      if (l > 1)
      {
        switch (ma)
        {
        case 0:
          dmr = -cm * Ylm[lm + 1];
          dmi = cm * Ylm[lm - 1];
          break;
        case 1:
          dmr = cm * Ylm[lm];
          dmi = czero;
          break;
        default:
          dmr = cm * Ylm[lm + ma - 1];
          dmi = cm * Ylm[lm - ma + 1];
        }
      }
      else
      {
        dmr = cm * Ylm[lm];
        dmi = czero;
        //dmr = (l==1) ? cm*Ylm[lm]:0.0;
        //dmi = 0.0;
      }
      if (m < 0)
      {
        gx = ahalf * (dpi - dmi);
        gy = -ahalf * (dpr + dmr);
      }
      else
      {
        gx = ahalf * (dpr - dmr);
        gy = ahalf * (dpi + dmi);
      }
      lm = index(l, m);
      if (ma)
      {
        gYlmX[lm] = NormFactor[lm] * gx;
        gYlmY[lm] = NormFactor[lm] * gy;
        gYlmZ[lm] = NormFactor[lm] * gz;
      }
      else
      {
        gYlmX[lm] = gx;
        gYlmY[lm] = gy;
        gYlmZ[lm] = gz;
      }
    }
  }
  for (int i = 0; i < cYlm.size(); i++)
    Ylm[i] *= NormFactor[i];
  //for (int i=0; i<Ylm.size(); i++) gradYlm[i]*= NormFactor[i];
}

template<typename T>
inline void SoaSphericalTensor<T>::evaluateVGH(T x, T y, T z)
{
  throw std::runtime_error("SoaSphericalTensor<T>::evaluateVGH(x,y,z):  Not implemented\n");
}

template<typename T>
inline void SoaSphericalTensor<T>::evaluateVGHGH(T x, T y, T z)
{
  throw std::runtime_error("SoaSphericalTensor<T>::evaluateVGHGH(x,y,z):  Not implemented\n");
}

extern template struct SoaSphericalTensor<float>;
extern template struct SoaSphericalTensor<double>;
} // namespace qmcplusplus
#endif
