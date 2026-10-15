#######################################################################################
# ATLAS card with pileup, <mu> = 0  (control: same detector card, no pileup)
# Derived from Delphes 3.5.1 cards/delphes_card_ATLAS_PileUp.tcl; only three changes:
#   set PileUpFile  -> absolute path of the 100k-event 13 TeV SoftQCD:inelastic library
#                      (results/pileup/MinBias_13TeV.pileup, made by code/generation/pileup/make_minbias_library.sh)
#   set MeanPileUp  50 -> 0   (Poisson mean number of inelastic pp interactions per crossing)
#   "source trackResolutionATLAS.tcl" replaced by the file's content: Delphes resolves
#   'source' relative to the top card's directory (<proc>/Cards under MG5), so it fails there.
# Everything else (Rho, JetPileUpSubtractor, TrackPileUpSubtractor, anti-kT R=0.6 jets,
# JetEnergyScale = 1.0, JetFlavorAssociation, BTagging, TauTagging) is as in the Delphes card.
# generate_samples.py drops the Tower, ParticleFlowCandidate and Vertex branches.
#######################################################################################
#######################################
# Order of execution of various modules
#######################################
# Has pileup, more accurate tracking efficiencys, impact parameter smearing (still cms)
# and finally, eta/phi smearing! Based on ID TDR for ATLAS.
# TODO: add ATLAS based ip smearing
set ExecutionPath {

  PileUpMerger
  ParticlePropagator

  ChargedHadronTrackingEfficiency
  ElectronTrackingEfficiency
  MuonTrackingEfficiency

  ChargedHadronMomentumSmearing
  ElectronMomentumSmearing
  MuonMomentumSmearing

  TrackMerger
  TrackSmearing
  Calorimeter
  ElectronFilter
  TrackPileUpSubtractor
  NeutralTowerMerger
  EFlowMergerAllTracks
  EFlowMerger
  EFlowFilter
  
  NeutrinoFilter
  GenJetFinder
  GenMissingET
  
  Rho
  FastJetFinder
  JetPileUpSubtractor

  JetEnergyScale

  PhotonEfficiency
  PhotonIsolation

  ElectronEfficiency
  ElectronIsolation

  MuonEfficiency
  MuonIsolation
  
  MissingET
  
  JetFlavorAssociation

  BTagging
  TauTagging

  UniqueObjectFinder

  ScalarHT

  TreeWriter
}

###############
# PileUp Merger
###############

module PileUpMerger PileUpMerger {
  set InputArray Delphes/stableParticles

  set ParticleOutputArray stableParticles
  set VertexOutputArray vertices

  # pre-generated minbias input file
  set PileUpFile /Users/TMHONG/Claude-code/ttbar_bblvqq/results/pileup/MinBias_13TeV.pileup

  # average expected pile up
  set MeanPileUp 0

   # maximum spread in the beam direction in m
  set ZVertexSpread 0.25

  # maximum spread in time in s
  set TVertexSpread 800E-12

  # vertex smearing formula f(z,t) (z,t need to be respectively given in m,s)
  set VertexDistributionFormula {exp(-(t^2/160e-12^2/2))*exp(-(z^2/0.053^2/2))}


}

#################################
# Propagate particles in cylinder
#################################

module ParticlePropagator ParticlePropagator {
  set InputArray PileUpMerger/stableParticles

  set OutputArray stableParticles
  set ChargedHadronOutputArray chargedHadrons
  set ElectronOutputArray electrons
  set MuonOutputArray muons

  # radius of the magnetic field coverage, in m
  set Radius 1.15
  # half-length of the magnetic field coverage, in m
  set HalfLength 3.51

  # magnetic field
  set Bz 2.0
}

####################################
# Charged hadron tracking efficiency
####################################

module Efficiency ChargedHadronTrackingEfficiency {
  set InputArray ParticlePropagator/chargedHadrons
  set OutputArray chargedHadrons

  # add EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for charged hadrons
  set EfficiencyFormula { 
                         (pt <= 0.5)   * (0.00) +
                         (abs(eta) <= 1.5) * (pt > 0.5   && pt <= 5.0)   * (0.85) +
                         (abs(eta) <= 1.5) * (pt > 5.0)                  * (0.90) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.5   && pt <= 5.0)   * (0.75) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 5.0)                  * (0.80) +
                         (abs(eta) > 2.5)                                                  * (0.00)}
}

##############################
# Electron tracking efficiency
##############################

module Efficiency ElectronTrackingEfficiency {
  set InputArray ParticlePropagator/electrons
  set OutputArray electrons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for electrons
  set EfficiencyFormula { 
                        (pt <= 0.5)   * (0.00) +
                        (abs(eta) <= 1.5) * (pt > 0.5   && pt <= 5.0)   * (0.85) +
                        (abs(eta) <= 1.5) * (pt > 5.0)                  * (0.90) +
                        (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.5   && pt <= 5.0)   * (0.75) +
                        (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 5.0)                  * (0.80) +
                        (abs(eta) > 2.5)                                                  * (0.00)}
}

##########################
# Muon tracking efficiency
##########################

module Efficiency MuonTrackingEfficiency {
  set InputArray ParticlePropagator/muons
  set OutputArray muons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # tracking efficiency formula for muons
  set EfficiencyFormula {
            (pt <= 0.5)   * (0.00) +
            (abs(eta) <= 1.5) * (pt > 0.5   && pt <= 5.0)   * (0.85) +
            (abs(eta) <= 1.5) * (pt > 5.0)                  * (0.90) +
            (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.5   && pt <= 5.0)   * (0.75) +
            (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 5.0)                  * (0.80) +
            (abs(eta) > 2.5)                                                  * (0.00)}
}

########################################
# Momentum resolution for charged tracks
########################################

module MomentumSmearing ChargedHadronMomentumSmearing {
  set InputArray ChargedHadronTrackingEfficiency/chargedHadrons
  set OutputArray chargedHadrons

  # set ResolutionFormula {resolution formula as a function of eta and pt}

  # resolution formula for charged hadrons
  # based on arXiv:1405.6569
  set ResolutionFormula {                  (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.06^2 + pt^2*1.3e-3^2) +
                         (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.10^2 + pt^2*1.7e-3^2) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.25^2 + pt^2*3.1e-3^2)}
}

###################################
# Momentum resolution for electrons
###################################

module MomentumSmearing ElectronMomentumSmearing {
  set InputArray ElectronTrackingEfficiency/electrons
  set OutputArray electrons

  # set ResolutionFormula {resolution formula as a function of eta and energy}

  # resolution formula for electrons
  # based on arXiv:1405.6569
  set ResolutionFormula {                  (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.03^2 + pt^2*1.3e-3^2) +
                         (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.05^2 + pt^2*1.7e-3^2) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.15^2 + pt^2*3.1e-3^2)}
}

###############################
# Momentum resolution for muons
###############################

module MomentumSmearing MuonMomentumSmearing {
  set InputArray MuonTrackingEfficiency/muons
  set OutputArray muons

  # set ResolutionFormula {resolution formula as a function of eta and pt}

  # resolution formula for muons
  set ResolutionFormula {                  (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.01^2 + pt^2*1.0e-4^2) +
                         (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.015^2 + pt^2*1.5e-4^2) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.025^2 + pt^2*3.5e-4^2)}
}

##############
# Track merger
##############

module Merger TrackMerger {
# add InputArray InputArray
  add InputArray ChargedHadronMomentumSmearing/chargedHadrons
  add InputArray ElectronMomentumSmearing/electrons
  add InputArray MuonMomentumSmearing/muons
  set OutputArray tracks
}
module TrackSmearing TrackSmearing {
  set InputArray TrackMerger/tracks
#  set BeamSpotInputArray BeamSpotFilter/beamSpotParticle
  set OutputArray tracks
#  set ApplyToPileUp true

  # magnetic field, fixed now 2 T now 3.8 T from CMS
  set Bz 2.0

  # --- inlined from Delphes cards/trackResolutionATLAS.tcl (was: source trackResolutionATLAS.tcl) ---
  set PResolutionFormula { 0.0 }
  # ctg theta and phi resolution from atlas ID TDR
  # pT < 5 based on 1 GeV line
  # pT > 15 based on 20 GeV line
  # pT between is linear interpolation with w=0.3 from 1 GeV -> 20 GeV
  # interpolation values chosen based on sqrt(s) = 13 TeV ttbar tracks, which are largely 
  # low pT
  set CtgThetaResolutionFormula { 
      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt <= 5 ) * 0.001753554502369671 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt <= 5 ) * 0.0018483412322274906 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt <= 5 ) * 0.0020853080568720393 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt <= 5 ) * 0.0024644549763033208 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt <= 5 ) * 0.0030331753554502378 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt <= 5 ) * 0.004360189573459721 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt <= 5 ) * 0.006303317535545024 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt <= 5 ) * 0.009099526066350713 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt <= 5 ) * 0.013933649289099528 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt <= 5 ) * 0.019478672985781993 +\
      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt > 5 && pt <= 15 ) * 0.001447867298578201 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt > 5 && pt <= 15 ) * 0.0014957345971564 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt > 5 && pt <= 15 ) * 0.0016402843601895744 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt > 5 && pt <= 15 ) * 0.001890047393364931 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt > 5 && pt <= 15 ) * 0.0022838862559241713 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt > 5 && pt <= 15 ) * 0.003246919431279625 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt > 5 && pt <= 15 ) * 0.004690995260663507 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt > 5 && pt <= 15 ) * 0.006718009478672987 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt > 5 && pt <= 15 ) * 0.010293838862559242 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt > 5 && pt <= 15 ) * 0.014371563981042657+\
      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt > 15 ) * 0.0007345971563981041 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt > 15 ) * 0.0006729857819905215 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt > 15 ) * 0.0006018957345971564 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt > 15 ) * 0.0005497630331753552 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt > 15 ) * 0.0005355450236966823 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt > 15 ) * 0.0006492890995260666 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt > 15 ) * 0.000928909952606635 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt > 15 ) * 0.0011611374407582938 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt > 15 ) * 0.0018009478672985782 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt > 15 ) * 0.002454976303317536 
  }


  set PhiResolutionFormula { 
      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt <= 5 ) * 0.0015096525096525092 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt <= 5 ) * 0.001625482625482625 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt <= 5 ) * 0.0017181467181467181 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt <= 5 ) * 0.0017876447876447872 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt <= 5 ) * 0.0021003861003861 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt <= 5 ) * 0.0022625482625482623 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt <= 5 ) * 0.0025868725868725866 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt <= 5 ) * 0.002934362934362934 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt <= 5 ) * 0.0031428571428571426 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt <= 5 ) * 0.003722007722007722+\
    
      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt > 5 && pt <= 15 ) * 0.001100647431997914 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt > 5 && pt <= 15 ) * 0.001184333014686712 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt > 5 && pt <= 15 ) * 0.0012486191014165291 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt > 5 && pt <= 15 ) * 0.001299004084470322 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt > 5 && pt <= 15 ) * 0.0015234213956721995 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt > 5 && pt <= 15 ) * 0.0016395394107934298 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt > 5 && pt <= 15 ) * 0.0018697497175632221 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt > 5 && pt <= 15 ) * 0.0021135717389415137 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt > 5 && pt <= 15 ) * 0.002260385852090032 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt > 5 && pt <= 15 ) * 0.002673604762318589+\

      ( abs(eta) > 0.0 && abs(eta) <= 0.25 ) * ( pt > 15 ) * 0.00014630225080385849 +\
      ( abs(eta) > 0.25 && abs(eta) <= 0.5 ) * ( pt > 15 ) * 0.000154983922829582 +\
      ( abs(eta) > 0.5 && abs(eta) <= 0.75 ) * ( pt > 15 ) * 0.00015305466237942118 +\
      ( abs(eta) > 0.75 && abs(eta) <= 1.0 ) * ( pt > 15 ) * 0.00015884244372990354 +\
      ( abs(eta) > 1.0 && abs(eta) <= 1.25 ) * ( pt > 15 ) * 0.0001771704180064309 +\
      ( abs(eta) > 1.25 && abs(eta) <= 1.5 ) * ( pt > 15 ) * 0.00018585209003215433 +\
      ( abs(eta) > 1.5 && abs(eta) <= 1.75 ) * ( pt > 15 ) * 0.00019646302250803858 +\
      ( abs(eta) > 1.75 && abs(eta) <= 2.0 ) * ( pt > 15 ) * 0.00019839228295819939 +\
      ( abs(eta) > 2.0 && abs(eta) <= 2.25 ) * ( pt > 15 ) * 0.00020128617363344054 +\
      ( abs(eta) > 2.25 && abs(eta) <= 2.5 ) * ( pt > 15 ) * 0.00022733118971061093

   }
  # taken from arXiv:1405.6569 fig. 15
  set D0ResolutionFormula {
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 0.5 && pt <= 1.0) * 0.08 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 1.0 && pt <= 2.0 ) * 0.045 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 2.0 && pt <= 4.0 ) * 0.028 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 4.0 && pt <= 6.0 ) * 0.020 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 6.0 && pt <= 10.0 ) * 0.016 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 10.0 && pt <= 30.0 ) * 0.014 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 30.0 ) * 0.0115
  }
  set DZResolutionFormula {
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 0.5 && pt <= 1.0) * 0.140 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 1.0 && pt <= 2.0 ) * 0.096 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 2.0 && pt <= 4.0 ) * 0.075 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 4.0 && pt <= 6.0 ) * 0.058 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 6.0 && pt <= 15.0 ) * 0.055 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 15.0 && pt <= 30.0 ) * 0.050 +\
      ( abs(eta) > 0.0 && abs(eta) <= 2.5 ) * ( pt > 30.0 ) * 0.047
  }
  # --- end of trackResolutionATLAS.tcl ---
}

#############
# Calorimeter
#############

module Calorimeter Calorimeter {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray TrackSmearing/tracks

  set TowerOutputArray towers
  set PhotonOutputArray photons

  set EFlowTrackOutputArray eflowTracks
  set EFlowPhotonOutputArray eflowPhotons
  set EFlowNeutralHadronOutputArray eflowNeutralHadrons

  set ECalEnergyMin 0.5
  set HCalEnergyMin 1.0

  set ECalEnergySignificanceMin 1.0
  set HCalEnergySignificanceMin 1.0

  set SmearTowerCenter true

  set pi [expr {acos(-1)}]

  # lists of the edges of each tower in eta and phi
  # each list starts with the lower edge of the first tower
  # the list ends with the higher edged of the last tower

  # 10 degrees towers
  set PhiBins {}
  for {set i -18} {$i <= 18} {incr i} {
    add PhiBins [expr {$i * $pi/18.0}]
  }
  foreach eta {-3.2 -2.5 -2.4 -2.3 -2.2 -2.1 -2 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 3.3} {
    add EtaPhiBins $eta $PhiBins
  }

  # 20 degrees towers
  set PhiBins {}
  for {set i -9} {$i <= 9} {incr i} {
    add PhiBins [expr {$i * $pi/9.0}]
  }
  foreach eta {-4.9 -4.7 -4.5 -4.3 -4.1 -3.9 -3.7 -3.5 -3.3 -3 -2.8 -2.6 2.8 3 3.2 3.5 3.7 3.9 4.1 4.3 4.5 4.7 4.9} {
    add EtaPhiBins $eta $PhiBins
  }

  # default energy fractions {abs(PDG code)} {Fecal Fhcal}
  add EnergyFraction {0} {0.0 1.0}
  # energy fractions for e, gamma and pi0
  add EnergyFraction {11} {1.0 0.0}
  add EnergyFraction {22} {1.0 0.0}
  add EnergyFraction {111} {1.0 0.0}
  # energy fractions for muon, neutrinos and neutralinos
  add EnergyFraction {12} {0.0 0.0}
  add EnergyFraction {13} {0.0 0.0}
  add EnergyFraction {14} {0.0 0.0}
  add EnergyFraction {16} {0.0 0.0}
  add EnergyFraction {1000022} {0.0 0.0}
  add EnergyFraction {1000023} {0.0 0.0}
  add EnergyFraction {1000025} {0.0 0.0}
  add EnergyFraction {1000035} {0.0 0.0}
  add EnergyFraction {1000045} {0.0 0.0}
  # energy fractions for K0short and Lambda
  add EnergyFraction {310} {0.3 0.7}
  add EnergyFraction {3122} {0.3 0.7}

  # set ECalResolutionFormula {resolution formula as a function of eta and energy}
  # http://arxiv.org/pdf/physics/0608012v1 jinst8_08_s08003
  # http://villaolmo.mib.infn.it/ICATPP9th_2005/Calorimetry/Schram.p.pdf
  # http://www.physics.utoronto.ca/~krieger/procs/ComoProceedings.pdf
  set ECalResolutionFormula {                  (abs(eta) <= 3.2) * sqrt(energy^2*0.0017^2 + energy*0.101^2) +
                             (abs(eta) > 3.2 && abs(eta) <= 4.9) * sqrt(energy^2*0.0350^2 + energy*0.285^2)}

  # set HCalResolutionFormula {resolution formula as a function of eta and energy}
  # http://arxiv.org/pdf/hep-ex/0004009v1
  # http://villaolmo.mib.infn.it/ICATPP9th_2005/Calorimetry/Schram.p.pdf
  set HCalResolutionFormula {                  (abs(eta) <= 1.7) * sqrt(energy^2*0.0302^2 + energy*0.5205^2 + 1.59^2) +
                             (abs(eta) > 1.7 && abs(eta) <= 3.2) * sqrt(energy^2*0.0500^2 + energy*0.706^2) +
                             (abs(eta) > 3.2 && abs(eta) <= 4.9) * sqrt(energy^2*0.09420^2 + energy*1.00^2)}
}

#################
# Electron filter
#################

module PdgCodeFilter ElectronFilter {
  set InputArray Calorimeter/eflowTracks
  set OutputArray electrons
  set Invert true
  add PdgCode {11}
  add PdgCode {-11}
}

##########################
# Track pile-up subtractor
##########################

module TrackPileUpSubtractor TrackPileUpSubtractor {
# add InputArray InputArray OutputArray
  add InputArray Calorimeter/eflowTracks eflowTracks
  add InputArray ElectronFilter/electrons electrons
  add InputArray MuonMomentumSmearing/muons muons

  # assume perfect pile-up subtraction for tracks with |z| > fZVertexResolution
  # Z vertex resolution in m
  set ZVertexResolution {0.0001}
}

####################
# Neutral tower merger
####################

module Merger NeutralTowerMerger {
# add InputArray InputArray
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflowTowers
}

##################################
# Energy flow merger (all tracks)
##################################

module Merger EFlowMergerAllTracks {
# add InputArray InputArray
  add InputArray TrackSmearing/tracks
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflow
}

####################
# Energy flow merger
####################

module Merger EFlowMerger {
# add InputArray InputArray
  add InputArray Calorimeter/eflowTracks
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflow
}

######################
# EFlowFilter
######################

module PdgCodeFilter EFlowFilter {
  set InputArray EFlowMerger/eflow
  set OutputArray eflow
  
  add PdgCode {11}
  add PdgCode {-11}
  add PdgCode {13}
  add PdgCode {-13}
}


#############
# Rho pile-up
#############

module FastJetGridMedianEstimator Rho {

  set InputArray Calorimeter/towers
  set RhoOutputArray rho

  # add GridRange rapmin rapmax drap dphi
  # rapmin - the minimum rapidity extent of the grid
  # rapmax - the maximum rapidity extent of the grid
  # drap - the grid spacing in rapidity
  # dphi - the grid spacing in azimuth

  add GridRange -5.0 -2.5 1.0 1.0
  add GridRange -2.5 2.5 0.5 0.5
  add GridRange 2.5 5.0 1.0 1.0

}


#####################
# Neutrino Filter
#####################

module PdgCodeFilter NeutrinoFilter {

  set InputArray Delphes/stableParticles
  set OutputArray filteredParticles

  set PTMin 0.0

  add PdgCode {12}
  add PdgCode {14}
  add PdgCode {16}
  add PdgCode {-12}
  add PdgCode {-14}
  add PdgCode {-16}

}

#####################
# MC truth jet finder
#####################

module FastJetFinder GenJetFinder {
  set InputArray NeutrinoFilter/filteredParticles

  set OutputArray jets

  # algorithm: 1 CDFJetClu, 2 MidPoint, 3 SIScone, 4 kt, 5 Cambridge/Aachen, 6 antikt
  set JetAlgorithm 6
  set ParameterR 0.6

  set JetPTMin 20.0
}

#########################
# Gen Missing ET merger
########################

module Merger GenMissingET {
# add InputArray InputArray
  add InputArray NeutrinoFilter/filteredParticles
  set MomentumOutputArray momentum
}


############
# Jet finder
############

module FastJetFinder FastJetFinder {
  set InputArray Calorimeter/towers

  set OutputArray jets

  # area algorithm: 0 Do not compute area, 1 Active area explicit ghosts, 2 One ghost passive area, 3 Passive area, 4 Voronoi, 5 Active area
  set AreaAlgorithm 5

  # jet algorithm: 1 CDFJetClu, 2 MidPoint, 3 SIScone, 4 kt, 5 Cambridge/Aachen, 6 antikt
  set JetAlgorithm 6
  set ParameterR 0.6

  set JetPTMin 20.0
}

###########################
# Jet Pile-Up Subtraction
###########################

module JetPileUpSubtractor JetPileUpSubtractor {
  set JetInputArray FastJetFinder/jets
  set RhoInputArray Rho/rho

  set OutputArray jets

  set JetPTMin 20.0
}

##################
# Jet Energy Scale
##################

module EnergyScale JetEnergyScale {
  set InputArray JetPileUpSubtractor/jets
  set OutputArray jets

 # scale formula for jets
  set ScaleFormula {1.0}
}

###################
# Photon efficiency
###################

module Efficiency PhotonEfficiency {
  set InputArray Calorimeter/eflowPhotons
  set OutputArray photons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # efficiency formula for photons
  set EfficiencyFormula {                                      (pt <= 10.0) * (0.00) +
                                           (abs(eta) <= 1.5) * (pt > 10.0)  * (0.95) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 10.0)  * (0.85) +
                         (abs(eta) > 2.5)                                   * (0.00)}
}

##################
# Photon isolation
##################

module Isolation PhotonIsolation {
  set CandidateInputArray PhotonEfficiency/photons
  set IsolationInputArray EFlowFilter/eflow
  set RhoInputArray Rho/rho

  set OutputArray photons

  set DeltaRMax 0.5

  set PTMin 0.5

  set PTRatioMax 0.12
}

#####################
# Electron efficiency
#####################

module Efficiency ElectronEfficiency {
  set InputArray TrackPileUpSubtractor/electrons
  set OutputArray electrons

  # set EfficiencyFormula {efficiency formula as a function of eta and pt}

  # efficiency formula for electrons
  set EfficiencyFormula {                                      (pt <= 10.0) * (0.00) +
                                           (abs(eta) <= 1.5) * (pt > 10.0)  * (0.95) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 10.0)  * (0.85) +
                         (abs(eta) > 2.5)                                   * (0.00)}
}

####################
# Electron isolation
####################

module Isolation ElectronIsolation {
  set CandidateInputArray ElectronEfficiency/electrons
  set IsolationInputArray EFlowFilter/eflow
  set RhoInputArray Rho/rho

  set OutputArray electrons

  set DeltaRMax 0.5

  set PTMin 0.5

  set PTRatioMax 0.12
}

#################
# Muon efficiency
#################

module Efficiency MuonEfficiency {
  set InputArray TrackPileUpSubtractor/muons
  set OutputArray muons

  # set EfficiencyFormula {efficiency as a function of eta and pt}

  # efficiency formula for muons
  set EfficiencyFormula {                                      (pt <= 10.0) * (0.00) +
                                           (abs(eta) <= 1.5) * (pt > 10.0)  * (0.95) +
                         (abs(eta) > 1.5 && abs(eta) <= 2.7) * (pt > 10.0)  * (0.85) +
                         (abs(eta) > 2.7)                                   * (0.00)}
}

################
# Muon isolation
################

module Isolation MuonIsolation {
  set CandidateInputArray MuonEfficiency/muons
  set IsolationInputArray EFlowFilter/eflow
  set RhoInputArray Rho/rho

  set OutputArray muons

  set DeltaRMax 0.5

  set PTMin 0.5

  set PTRatioMax 0.25
}

###################
# Missing ET merger
###################

module Merger MissingET {
# add InputArray InputArray
  add InputArray EFlowMergerAllTracks/eflow
  set MomentumOutputArray momentum
}


##################
# Scalar HT merger
##################

module Merger ScalarHT {
# add InputArray InputArray
  add InputArray UniqueObjectFinder/jets
  add InputArray UniqueObjectFinder/electrons
  add InputArray UniqueObjectFinder/photons
  add InputArray UniqueObjectFinder/muons
  set EnergyOutputArray energy
}

########################
# Jet Flavor Association
########################

module JetFlavorAssociation JetFlavorAssociation {
  
  set PartonInputArray Delphes/partons
  set ParticleInputArray Delphes/allParticles
  set ParticleLHEFInputArray Delphes/allParticlesLHEF
  set JetInputArray JetEnergyScale/jets
  
  set DeltaR 0.5
  set PartonPTMin 1.0
  set PartonEtaMax 2.5

}

###########
# b-tagging
###########

module BTagging BTagging {
  set JetInputArray JetEnergyScale/jets

  set BitNumber 0

  # add EfficiencyFormula {abs(PDG code)} {efficiency formula as a function of eta and pt}
  # PDG code = the highest PDG code of a quark or gluon inside DeltaR cone around jet axis
  # gluon's PDG code has the lowest priority

  # based on ATL-PHYS-PUB-2015-022

  # default efficiency formula (misidentification rate)
  add EfficiencyFormula {0} {0.002+7.3e-06*pt}

  # efficiency formula for c-jets (misidentification rate)
  add EfficiencyFormula {4} {0.20*tanh(0.02*pt)*(1/(1+0.0034*pt))}

  # efficiency formula for b-jets
  add EfficiencyFormula {5} {0.80*tanh(0.003*pt)*(30/(1+0.086*pt))}
}

#############
# tau-tagging
#############

module TrackCountingTauTagging TauTagging {
 
  set ParticleInputArray Delphes/allParticles
  set PartonInputArray Delphes/partons
  set TrackInputArray TrackSmearing/tracks
  set JetInputArray JetEnergyScale/jets

  set DeltaR 0.2
  set DeltaRTrack 0.2

  set TrackPTMin 1.0
 
  set TauPTMin 1.0
  set TauEtaMax 2.5

  # instructions: {n-prongs} {eff} 
  
  # 1 - one prong efficiency
  # 2 - two or more efficiency
  # -1 - one prong mistag rate
  # -2 - two or more mistag rate
 
  set BitNumber 0
 
  # taken from ATL-PHYS-PUB-2015-045 (medium working point)
  add EfficiencyFormula {1} {0.70}
  add EfficiencyFormula {2} {0.60}
  add EfficiencyFormula {-1} {0.02}
  add EfficiencyFormula {-2} {0.01}

}

#####################################################
# Find uniquely identified photons/electrons/tau/jets
#####################################################

module UniqueObjectFinder UniqueObjectFinder {
# earlier arrays take precedence over later ones
# add InputArray InputArray OutputArray
  add InputArray PhotonIsolation/photons photons
  add InputArray ElectronIsolation/electrons electrons
  add InputArray MuonIsolation/muons muons
  add InputArray JetEnergyScale/jets jets
}

##################
# ROOT tree writer
##################

# tracks, towers and eflow objects are not stored by default in the output.
# if needed (for jet constituent or other studies), uncomment the relevant
# "add Branch ..." lines.

module TreeWriter TreeWriter {
# add Branch InputArray BranchName BranchClass
  add Branch Delphes/allParticles Particle GenParticle

#  add Branch TrackMerger/tracks Track Track
  add Branch Calorimeter/towers Tower Tower

#  add Branch Calorimeter/eflowTracks EFlowTrack Track
#  add Branch Calorimeter/eflowPhotons EFlowPhoton Tower
#  add Branch Calorimeter/eflowNeutralHadrons EFlowNeutralHadron Tower
  add Branch EFlowMerger/eflow ParticleFlowCandidate ParticleFlowCandidate
  add Branch GenJetFinder/jets GenJet Jet
  add Branch GenMissingET/momentum GenMissingET MissingET

  add Branch UniqueObjectFinder/jets Jet Jet
  add Branch UniqueObjectFinder/electrons Electron Electron
  add Branch UniqueObjectFinder/photons Photon Photon
  add Branch UniqueObjectFinder/muons Muon Muon
  add Branch MissingET/momentum MissingET MissingET
  add Branch ScalarHT/energy ScalarHT ScalarHT
  add Branch Rho/rho Rho Rho
  add Branch PileUpMerger/vertices Vertex Vertex

}
