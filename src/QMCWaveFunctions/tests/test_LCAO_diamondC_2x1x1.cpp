//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Kevin Gasperich, kgasperich@anl.gov, Argonne National Laboratory
//
// File created by: Kevin Gasperich, kgasperich@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"

#include "OhmmsData/Libxml2Doc.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "LCAO/LCAOrbitalBuilder.h"
#include <ResourceCollection.h>
#include "QMCHamiltonians/NLPPJob.h"
#include "DistanceTable.h"

#include <stdio.h>
#include <string>
#include <limits>

using std::string;

namespace qmcplusplus
{

TEST_CASE("LCAO DiamondC_2x1x1", "[wavefunction]")
{
  using VT       = SPOSet::ValueType;
  Communicate* c = OHMMS::Controller;

  ParticleSet::ParticleLayout lattice;
  lattice.R = {6.7463223, 6.7463223, 0.0, 0.0, 3.37316115, 3.37316115, 3.37316115, 0.0, 3.37316115};

  SimulationCell simcell(lattice);
  ParticleSet ions_(simcell);

  ions_.setName("ion0");
  ions_.create({4});
  ions_.R[0]           = {0.0, 0.0, 0.0};
  ions_.R[1]           = {1.686580575, 1.686580575, 1.686580575};
  ions_.R[2]           = {3.37316115, 3.37316115, 0.0};
  ions_.R[3]           = {5.059741726, 5.059741726, 1.686580575};
  SpeciesSet& ispecies = ions_.getSpeciesSet();
  const int Cidx       = ispecies.addSpecies("C");

  ions_.print(app_log());

  ParticleSet elec_(simcell);
  elec_.setName("elec");
  elec_.create({8, 8});
  elec_.R[0] = {0.0, 1.0, 0.0};
  elec_.R[1] = {0.0, 1.1, 0.0};
  elec_.R[2] = {0.0, 1.2, 0.0};
  elec_.R[3] = {0.0, 1.3, 0.0};

  SpeciesSet& tspecies         = elec_.getSpeciesSet();
  const int upIdx              = tspecies.addSpecies("u");
  const int downIdx            = tspecies.addSpecies("d");
  const int chargeIdx          = tspecies.addAttribute("charge");
  tspecies(chargeIdx, upIdx)   = -1;
  tspecies(chargeIdx, downIdx) = -1;
  const int ei_table_index     = elec_.addTable(ions_);

  // diamondC_2x1x1
  // from tests/solids/diamondC_2x1x1-Gaussian_pp/C_Diamond-Twist0.wfj.xml
  const std::string wf_xml_str = R"(
    <sposet_collection type="molecularorbital" name="LCAOBSet" source="ion0" transform="yes" twist="0  0  0" href="C_Diamond.h5" PBCimages="5  5  5">
      <basisset name="LCAOBSet" key="GTO" transform="yes">
        <grid type="log" ri="1.e-6" rf="1.e2" npts="1001"/>
      </basisset>
      <sposet name="spoud" size="8">
        <occupation mode="ground"/>
        <coefficient size="116" spindataset="0"/>
      </sposet>
    </sposet_collection>
  )";
  Libxml2Document doc;
  bool okay = doc.parseFromString(wf_xml_str);
  REQUIRE(okay);

  xmlNodePtr root       = doc.getRoot();
  xmlNodePtr bset_xml   = xmlFirstElementChild(root);
  xmlNodePtr sposet_xml = xmlNextElementSibling(bset_xml);

  LCAOrbitalBuilder lcaoSet(elec_, ions_, c, root);
  auto spo = lcaoSet.createSPOSetFromXML(sposet_xml);
  REQUIRE(spo);
  auto& lcao_spos = dynamic_cast<const LCAOrbitalSet&>(*spo);
  CHECK(!lcao_spos.isIdentity());

  const int norb = spo->getOrbitalSetSize();
  app_log() << "norb: " << norb << std::endl;

  // test batched interfaces
  const size_t nw = 2;

  ParticleSet elec_2(elec_);
  // interchange positions
  elec_2.R[0] = elec_.R[1];
  elec_2.R[1] = elec_.R[0];

  RefVectorWithLeader<ParticleSet> p_list(elec_, {elec_, elec_2});

  std::unique_ptr<SPOSet> spo_2(spo->makeClone());
  RefVectorWithLeader<SPOSet> spo_list(*spo, {*spo, *spo_2});

  ResourceCollection pset_res("test_pset_res");
  ResourceCollection spo_res("test_spo_res");

  elec_.createResource(pset_res);
  spo->createResource(spo_res);

  ResourceCollectionTeamLock<ParticleSet> mw_pset_lock(pset_res, p_list);
  ResourceCollectionTeamLock<SPOSet> mw_sposet_lock(spo_res, spo_list);

  ParticleSet::mw_update(p_list);

  // make VPs
  const size_t nvp_                  = 4;
  const size_t nvp_2                 = 3;
  const std::vector<size_t> nvp_list = {nvp_, nvp_2};
  VirtualParticleSet VP_(elec_, nvp_);
  VirtualParticleSet VP_2(elec_2, nvp_2);

  // move VPs
  std::vector<ParticleSet::SingleParticlePos> newpos_vp_(nvp_);
  std::vector<ParticleSet::SingleParticlePos> newpos_vp_2(nvp_2);
  for (int i = 0; i < nvp_; i++)
  {
    newpos_vp_[i][0] = 0.1 * i;
    newpos_vp_[i][1] = 0.2 * i;
    newpos_vp_[i][2] = 0.3 * i;
  }
  for (int i = 0; i < nvp_2; i++)
  {
    newpos_vp_2[i][0] = 0.2 * i;
    newpos_vp_2[i][1] = 0.3 * i;
    newpos_vp_2[i][2] = 0.4 * i;
  }

  const auto& ei_table_  = elec_.getDistTableAB(ei_table_index);
  const auto& ei_table_2 = elec_2.getDistTableAB(ei_table_index);
  // make virtual move of elec 0, reference ion 1
  NLPPJob<SPOSet::RealType> job_(1, 0, ei_table_.getDistances()[0][1], -ei_table_.getDisplacements()[0][1]);
  // make virtual move of elec 1, reference ion 3
  NLPPJob<SPOSet::RealType> job_2(3, 1, ei_table_2.getDistances()[1][3], -ei_table_2.getDisplacements()[1][3]);
  // VP_.makeMoves(elec_, 0, newpos_vp_);
  // VP_2.makeMoves(elec_2, 0, newpos_vp_2);


  // make VP refvec
  RefVectorWithLeader<VirtualParticleSet> vp_list(VP_, {VP_, VP_2});


  ResourceCollection vp_res("test_vp_res");
  VP_.createResource(vp_res);
  ResourceCollectionTeamLock<VirtualParticleSet> mw_vpset_lock(vp_res, vp_list);
  VirtualParticleSet::mw_makeMoves(vp_list, p_list, {newpos_vp_, newpos_vp_2}, {job_, job_2}, false);

  // fill invrow with dummy data for each walker
  std::vector<SPOSet::ValueType> psiMinv_data_(norb);
  std::vector<SPOSet::ValueType> psiMinv_data_2(norb);
  SPOSet::ValueVector psiMinv_ref_0(norb);
  SPOSet::ValueVector psiMinv_ref_1(norb);
  for (int i = 0; i < norb; i++)
  {
    psiMinv_data_[i]  = 0.1 * i;
    psiMinv_data_2[i] = 0.2 * i;
    psiMinv_ref_0[i]  = psiMinv_data_[i];
    psiMinv_ref_1[i]  = psiMinv_data_2[i];
  }
  std::vector<const SPOSet::ValueType*> invRow_ptr_list{psiMinv_data_.data(), psiMinv_data_2.data()};

  // ratios_list
  std::vector<std::vector<SPOSet::ValueType>> ratios_list(nw);
  for (size_t iw = 0; iw < nw; iw++)
    ratios_list[iw].resize(nvp_list[iw]);

  // just need dummy refvec with correct size
  SPOSet::ValueVector tmp_psi_list(norb);
  spo->mw_evaluateDetRatios(spo_list, RefVectorWithLeader<const VirtualParticleSet>(VP_, {VP_, VP_2}),
                            RefVector<SPOSet::ValueVector>{tmp_psi_list}, invRow_ptr_list, ratios_list);

  std::vector<SPOSet::ValueType> ratios_ref_0(nvp_);
  std::vector<SPOSet::ValueType> ratios_ref_1(nvp_2);
  spo->evaluateDetRatios(VP_, tmp_psi_list, psiMinv_ref_0, ratios_ref_0);
  spo_2->evaluateDetRatios(VP_2, tmp_psi_list, psiMinv_ref_1, ratios_ref_1);
  for (int ivp = 0; ivp < nvp_; ivp++)
    CHECK(std::real(ratios_list[0][ivp]) == Approx(std::real(ratios_ref_0[ivp])));
  for (int ivp = 0; ivp < nvp_2; ivp++)
    CHECK(std::real(ratios_list[1][ivp]) == Approx(std::real(ratios_ref_1[ivp])));
#ifdef QMC_COMPLEX
  for (int ivp = 0; ivp < nvp_; ivp++)
    CHECK(std::imag(ratios_list[0][ivp]) == Approx(std::imag(ratios_ref_0[ivp])));
  for (int ivp = 0; ivp < nvp_2; ivp++)
    CHECK(std::imag(ratios_list[1][ivp]) == Approx(std::imag(ratios_ref_1[ivp])));
#endif
  // for (int ivp = 0; ivp < nvp_; ivp++)
  //   app_log() << "CHECK(std::real(ratios_ref_0[" << ivp << "]) == Approx(" << std::real(ratios_ref_0[ivp]) << "));\n";
  // for (int ivp = 0; ivp < nvp_2; ivp++)
  //   app_log() << "CHECK(std::real(ratios_ref_1[" << ivp << "]) == Approx(" << std::real(ratios_ref_1[ivp]) << "));\n";
  // app_log() << "ratios_list refvalues: \n" << std::setprecision(14);
  // for (int iw = 0; iw < nw; iw++)
  //   for (int ivp = 0; ivp < nvp_list[iw]; ivp++)
  //     app_log() << "CHECK(std::real(ratios_list[" << iw << "][" << ivp << "]) == Approx("
  //               << std::real(ratios_list[iw][ivp]) << "));\n";
  // app_log() << "ratios_list refvalues: \n" << std::setprecision(14);
  // for (int iw = 0; iw < nw; iw++)
  //   for (int ivp = 0; ivp < nvp_list[iw]; ivp++)
  //     app_log() << "CHECK(std::imag(ratios_list[" << iw << "][" << ivp << "]) == Approx("
  //               << std::imag(ratios_list[iw][ivp]) << "));\n";


  CHECK(std::real(ratios_list[0][0]) == Approx(0.0020585459020935));
  CHECK(std::real(ratios_list[0][1]) == Approx(0.0086907202392454));
  CHECK(std::real(ratios_list[0][2]) == Approx(0.013372172641792));
  CHECK(std::real(ratios_list[0][3]) == Approx(0.013426136583449));
  CHECK(std::real(ratios_list[1][0]) == Approx(0.004117091804187));
  CHECK(std::real(ratios_list[1][1]) == Approx(0.026564850132802));
  CHECK(std::real(ratios_list[1][2]) == Approx(0.031626101035837));

  CHECK(std::real(ratios_ref_0[0]) == Approx(0.002058545902));
  CHECK(std::real(ratios_ref_0[1]) == Approx(0.008690720239));
  CHECK(std::real(ratios_ref_0[2]) == Approx(0.01337217264));
  CHECK(std::real(ratios_ref_0[3]) == Approx(0.01342613658));
  CHECK(std::real(ratios_ref_1[0]) == Approx(0.004117091804));
  CHECK(std::real(ratios_ref_1[1]) == Approx(0.02656485013));
  CHECK(std::real(ratios_ref_1[2]) == Approx(0.03162610104));
}
} // namespace qmcplusplus
