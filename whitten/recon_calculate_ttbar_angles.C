#ifdef __CLING__
R__ADD_INCLUDE_PATH(/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/MG5_aMC_v3_7_0/Delphes)
R__ADD_INCLUDE_PATH(/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/MG5_aMC_v3_7_0/Delphes/external)
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#endif


// which mass constrains the neutrino ---
enum Hypothesis { kTTbarHypothesis = 0, kHiggsHypothesis = 1 };
const int kHypothesis = kTTbarHypothesis;      //change this if you want to change the hypothesis used

const char* kInputFile = "/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/generatedEvents/full_diHiggs/Events/run_01/tag_1_delphes_events.root"; //change for different input/output files
const char* kOutFile   = "/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/rootFiles/reco_hh.root";
const char* kPlotSub   = "plots/reco_hh";

// Set by calculate_ttbar_angles(); read by drawHist().
std::string gPlotSubDir = "plots/ttbar";

// have to match the values from param_card.dat
//       grep -iE "^[[:space:]]*(6|23|24|25)[[:space:]]" /home/cicero/Pitt/Hong/sim-tools/runs/ttbar_bbWW/Cards/param_card.dat
const double kMW     = 8.041900e+01;
const double kMT     = 1.730000e+02;
const double kMHiggs = 1.250000e+02;
const double kGammaW = 2.047600e+00;

// --- chi2 reference values and resolutions.
//     Run with kCalibrationMode = true to measure your own, then paste here.
double kRefTopLep = 173.703,  kSigTopLep = 28.6801;
double kRefTopHad = 170.944,  kSigTopHad = 23.3526;
double kRefWHad   =  82.5544,  kSigWHad   = 15.3515;
double kRefWLep   =  86.9516,  kSigWLep   = 14.3852;

// --- object selection ---
const double kJetPtMin  = 25.0;
const double kJetEtaMax =  2.5;
const double kLepPtMin  = 25.0;
const double kLepEtaMax =  2.5;
const double kLepIsoMax =  0.15;
const double kMetMin    = 30.0;
const double kOverlapDR =  0.4;

// --- truth matching cones (ATLAS arXiv:1204.0952) ---
const double kMatchDRJet = 0.3;
const double kMatchDRLep = 0.1;

// --- behaviour switches ---
const bool kCalibrationMode  = true; // force the truth assignment and print
                                      // the mass means/RMSs for calibration
const bool kDownTypeUseWFrame = false; // false: softer jet in the TOP frame
                                      // true : jet closest to b in the W frame)
const bool kRescaleMetOnFail = true; // on an imaginary solution: false takes
                                      // the real part, true shrinks MET until discriminant is zero


double_t* cross(double_t A[3], double_t B[3])
{
  double_t* C = new double_t[3];
  C[0] = A[1]*B[2]-A[2]*B[1];
  C[1] = -(A[0]*B[2]-A[2]*B[0]);
  C[2] = A[0]*B[1]-A[1]*B[0];
  return C;
}

double_t dot(double_t A[3], double_t B[3])
{
  double_t C = 0;
  C = A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
  return C;
}

double_t norm(double_t A[3])
{
  double_t magnitude;
  magnitude = sqrt(pow(A[0],2)+pow(A[1],2)+pow(A[2],2));
  return magnitude;
}

void normalize(double_t A[3])
{
  double_t magnitude = norm(A);
  if (magnitude > 0) {
    A[0] /= magnitude;
    A[1] /= magnitude;
    A[2] /= magnitude;
  }
}

double_t getCosth(TLorentzVector* axis, TLorentzVector* tvec1)
{
    double_t d3vec_1[3] = {tvec1->X(), tvec1->Y(), tvec1->Z()};
    double_t axis_3vec[3] = {axis->X(), axis->Y(), axis->Z()};

    return dot(d3vec_1, axis_3vec) / (norm(d3vec_1) * norm(axis_3vec));

}


double_t getPhi(TLorentzVector* x_axis, TLorentzVector* y_axis, TLorentzVector* z_axis, TLorentzVector* tvec)
{
    double_t d3vec[3] = {tvec->X(), tvec->Y(), tvec->Z()};

    double_t x_axis_3vec[3] = {x_axis->X(), x_axis->Y(), x_axis->Z()};
    double_t y_axis_3vec[3] = {y_axis->X(), y_axis->Y(), y_axis->Z()};
    double_t z_axis_3vec[3] = {z_axis->X(), z_axis->Y(), z_axis->Z()};
    
    // Normalize axes to unit vectors
    double_t z_norm = norm(z_axis_3vec);
    double_t x_norm = norm(x_axis_3vec);
    double_t y_norm = norm(y_axis_3vec);
    double_t z_hat[3] = {z_axis_3vec[0]/z_norm, z_axis_3vec[1]/z_norm, z_axis_3vec[2]/z_norm};
    double_t x_hat[3] = {x_axis_3vec[0]/x_norm, x_axis_3vec[1]/x_norm, x_axis_3vec[2]/x_norm};
    double_t y_hat[3] = {y_axis_3vec[0]/y_norm, y_axis_3vec[1]/y_norm, y_axis_3vec[2]/y_norm};
    
    // Project d3vec onto the plane perpendicular to z_axis
    double_t dot_z = dot(d3vec, z_hat);
    double_t d3vec_proj[3] = {d3vec[0] - dot_z * z_hat[0],
                              d3vec[1] - dot_z * z_hat[1],
                              d3vec[2] - dot_z * z_hat[2]};
    
    return atan2(dot(d3vec_proj, y_hat), dot(d3vec_proj, x_hat));
}

double_t* calcAngles(TLorentzVector quark, TLorentzVector boson, TLorentzVector lepton, int lambda)
{
    // Reference frame for theta and phi can be arbitrary
    TLorentzVector x_axis = TLorentzVector(1,0,0,0);
    TLorentzVector y_axis = TLorentzVector(0,1,0,0);
    TLorentzVector z_axis = TLorentzVector(0,0,1,0);

    boson.Boost(-quark.BoostVector());
    lepton.Boost(-quark.BoostVector());

    double_t theta = getCosth(&z_axis, &boson);
    double_t phi = getPhi(&x_axis, &y_axis, &z_axis, &boson);

    TVector3 beam = TVector3(0,0,1);

    // Thetastar and phistar use a prime fram where Z' is in the direction of boson momentum and Y' lies in the original xy-plane
    TVector3 zprime = lambda * boson.Vect().Unit();
    double_t crossMag = beam.Cross(zprime).Mag();
    if (crossMag < 1e-6) {
        std::cout << "WARNING: W parallel to beam, event " << std::endl;
        return new double_t[4]{-10,-10,-10,10};  // skip this event
    }
    TVector3 yprime = beam.Cross(zprime).Unit();
    TVector3 xprime = yprime.Cross(zprime).Unit();

    //We boost the prime frame such that it is in the boson's reference frame, then calculate angles in that frame
    z_axis = TLorentzVector(zprime.X(), zprime.Y(), zprime.Z(), 0);
    y_axis = TLorentzVector(yprime.X(), yprime.Y(), yprime.Z(), 0);
    x_axis = TLorentzVector(xprime.X(), xprime.Y(), xprime.Z(), 0);
    z_axis.Boost(-boson.BoostVector());
    y_axis.Boost(-boson.BoostVector());
    x_axis.Boost(-boson.BoostVector());


    lepton.Boost(-boson.BoostVector());

    double_t thetastar = getCosth(&z_axis, &lepton);
    double_t phistar = getPhi(&x_axis, &y_axis, &z_axis, &lepton);

    double_t*angles = new double_t[4];
    angles[0] = theta;
    angles[1] = phi;
    angles[2] = thetastar;
    angles[3] = phistar;
    return angles;
}

void drawHist(std::string title, TH1F* hist, std::string xaxis)
{
    TCanvas* m1 = new TCanvas(title.c_str(), title.c_str());
    // Normalize histogram
    if (hist->Integral() > 0) {
        hist->Scale(1.0 / hist->Integral(), "WIDTH");
    }
    hist->Draw("HIST");
    hist->GetXaxis()->SetTitle(xaxis.c_str());
    hist->GetYaxis()->SetTitle("Probability");
    hist->SetTitle(title.c_str());
    std::string fileName = gPlotSubDir + "/" + title + ".pdf";
    m1->SaveAs(fileName.c_str());
}


// =====================
//  ADDED METHODS
// =====================

struct NuSolSet {
    std::vector<TLorentzVector> sols;
    std::vector<double> chi2;
    bool   real;
    double disc;
    double mT;
};

NuSolSet solveNeutrinoRaw(const TLorentzVector& vis, const TLorentzVector& met,
                          double targetMass)
{
    NuSolSet out;
    out.real = false; out.disc = 0.0; out.mT = 0.0;

    const double pxn = met.Px(), pyn = met.Py();
    const double ptn2 = pxn*pxn + pyn*pyn;
    const double ptn  = sqrt(ptn2);

    const double mv2 = vis.M2();
    const double A   = mv2 + vis.Pt()*vis.Pt();
    if (A < 1e-9) return out;

    const double Lambda = 0.5*(targetMass*targetMass - mv2) + vis.Px()*pxn + vis.Py()*pyn;
    const double disc   = Lambda*Lambda - A*ptn2;
    out.disc = disc;
    out.real = (disc >= 0.0);

    const double mTV = sqrt(A);
    out.mT = sqrt(std::max(0.0, mv2 + 2.0*(mTV*ptn - (vis.Px()*pxn + vis.Py()*pyn))));

    const double dUse = (disc < 0.0) ? 0.0 : disc;
    const double root = vis.E()*sqrt(dUse);

    const double pz1 = (Lambda*vis.Pz() + root)/A;
    TLorentzVector nu1; nu1.SetPxPyPzE(pxn, pyn, pz1, sqrt(ptn2 + pz1*pz1));
    out.sols.push_back(nu1);

    if (out.real && root > 0.0) {
        const double pz2 = (Lambda*vis.Pz() - root)/A;
        TLorentzVector nu2; nu2.SetPxPyPzE(pxn, pyn, pz2, sqrt(ptn2 + pz2*pz2));
        out.sols.push_back(nu2);
    }
    return out;
}

NuSolSet solveNeutrino(const TLorentzVector& vis, const TLorentzVector& met,
                       double targetMass)
{
    NuSolSet first = solveNeutrinoRaw(vis, met, targetMass);
    if (first.real || !kRescaleMetOnFail) return first;

    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 40; ++it) {
        double mid = 0.5*(lo + hi);
        TLorentzVector sc; sc.SetPxPyPzE(met.Px()*mid, met.Py()*mid, 0.0, met.Pt()*mid);
        if (solveNeutrinoRaw(vis, sc, targetMass).real) lo = mid; else hi = mid;
    }
    TLorentzVector sc; sc.SetPxPyPzE(met.Px()*lo, met.Py()*lo, 0.0, met.Pt()*lo);
    NuSolSet out = solveNeutrinoRaw(vis, sc, targetMass);
    out.real = false;
    return out;
}

NuSolSet solveNeutrino_chi2(const TLorentzVector& lep, const std::vector<TLorentzVector>& bJets, const TLorentzVector& met)
{
    NuSolSet out;

    const double metX = met.Px(); const double metY = met.Py();
    const double sW = kSigWLep; const double sT = kSigTopLep; // sigmas for chi2 terms

    for (const TLorentzVector& bLep : bJets) {
        auto chi2 = [&](double pz) {
            TLorentzVector nu;
            nu.SetXYZM(metX, metY, pz, 0.);
            double cW = ((lep + nu).M()        - kRefWLep) / sW;
            double cT = ((lep + nu + bLep).M() - kRefTopLep) / sT;
            return cW*cW + cT*cT;
        };

        ROOT::Math::Functor1D func(chi2);
        ROOT::Math::BrentMinimizer1D bm;
        bm.SetFunction(func, -3000., 3000.);
        bm.SetNpx(1000);
        bm.Minimize(200, 1e-4, 1e-6);

        TLorentzVector nu;
        nu.SetXYZM(metX, metY, bm.XMinimum(), 0.);
        out.sols.push_back(nu);
        out.chi2.push_back(bm.FValMinimum());
    }
    return out;
}

// --- chi2 terms ------------------------------------------------------------
double gaussTerm(double x, double ref, double sigma)
{
    const double d = (x - ref)/sigma;
    return d*d;
}

double bwPenalty(double m, double m0, double gamma)
{
    const double num = m*m - m0*m0;
    return 2.0*log(num*num + m0*m0*gamma*gamma);
}

// --- down-type jet ---------------------------------------------------------
//  Returns 0 or 1: index into {j1, j2} of the down-type candidate.
int pickDownTypeIndex(const TLorentzVector& j1, const TLorentzVector& j2,
                      const TLorentzVector& wHad, const TLorentzVector& bHad)
{
    if (!kDownTypeUseWFrame) {
        TLorentzVector topHad = wHad + bHad;
        TLorentzVector a = j1, b = j2;
        a.Boost(-topHad.BoostVector());
        b.Boost(-topHad.BoostVector());
        return (a.E() < b.E()) ? 0 : 1;
    } else {
        TLorentzVector a = j1, b = j2, bb = bHad;
        a.Boost(-wHad.BoostVector());
        b.Boost(-wHad.BoostVector());
        bb.Boost(-wHad.BoostVector());
        return (a.Angle(bb.Vect()) < b.Angle(bb.Vect())) ? 0 : 1;
    }
}

// --- truth matching --------------------------------------------------------
int closestJet(const TLorentzVector& parton,
               const std::vector<TLorentzVector>& jets, double maxDR)
{
    int best = -1;
    double bestDR = maxDR;
    for (size_t i = 0; i < jets.size(); ++i) {
        double dr = parton.DeltaR(jets[i]);
        if (dr < bestDR) { bestDR = dr; best = (int)i; }
    }
    return best;
}

bool matchTwo(const TLorentzVector& p0, const TLorentzVector& p1,
              const std::vector<TLorentzVector>& jets, double maxDR,
              int& i0, int& i1)
{
    i0 = closestJet(p0, jets, maxDR);
    i1 = closestJet(p1, jets, maxDR);
    return (i0 >= 0 && i1 >= 0 && i0 != i1);
}


// ==================
//  MAIN
// ==================

long cf_all = 0, cf_lep = 0, cf_met = 0, cf_b1 = 0, cf_b2 = 0, cf_lj = 0;

void recon_calculate_ttbar_angles()
{
    bool debug = false;

    const bool isHiggs = (kHypothesis == kHiggsHypothesis);
    const char* inFile  = kInputFile;
    const char* outFile = kOutFile;
    gPlotSubDir = kPlotSub;
    const double kTargetMass = isHiggs ? kMHiggs : kMW;

    std::cout << "Hypothesis: " << (isHiggs ? "HIGGS  (m(l nu j j) = mH)"
                                            : "TTBAR  (m(l nu) = mW)") << std::endl;
    std::cout << "Input : " << inFile  << std::endl;
    std::cout << "Output: " << outFile << std::endl;

    gSystem->mkdir("rootFiles", kTRUE);
    gSystem->mkdir(gPlotSubDir.c_str(), kTRUE);
    gSystem->mkdir("/home/cicero/Pitt/Hong/sim-tools/analysis/rootFiles", kTRUE);

    gSystem->Load("libDelphes");

    TChain chain("Delphes");
    chain.Add(inFile);

    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchJet      = treeReader->UseBranch("Jet");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchMuon     = treeReader->UseBranch("Muon");
    TClonesArray *branchMET      = treeReader->UseBranch("MissingET");

    int nEvents = treeReader->GetEntries();

    int progressStep = nEvents / 100;
    if (progressStep < 1) progressStep = 1;   // guard for nEvents < 100
    TStopwatch progressTimer;
    progressTimer.Start();

    int events_total = 0;
    int events_jets = 0;
    int events_leptons = 0;
    int events_btags = 0;
    int events_selected = 0;

    TH1F *wplus_mass = new TH1F("test", "test", 50, 0, 200);
    TH1F *wminus_mass = new TH1F("test", "test", 50, 0, 200);
    TH1F *t_theta_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *t_phi_hist = new TH1F("test", "test", 50, -3.14, 3.14);
    TH1F *t_thetastar_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *t_phistar_hist = new TH1F("test", "test", 50, -3.14, 3.14);
    TH1F *tbar_theta_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *tbar_phi_hist = new TH1F("test", "test", 50, -3.14, 3.14);
    TH1F *tbar_thetastar_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *tbar_phistar_hist = new TH1F("test", "test", 50, -3.14, 3.14);
    TH1F *rapidity_hist = new TH1F("test", "test", 50, -5,5);
    TH1F *cosprod_hist = new TH1F("test", "test", 50, -1,1);

    TH1F *cal_topLep = new TH1F("cal_topLep", "m(l#nu b) truth perm", 60, 80, 320);
    TH1F *cal_topHad = new TH1F("cal_topHad", "m(jjb) truth perm",    60, 80, 320);
    TH1F *cal_wHad   = new TH1F("cal_wHad",   "m(jj) truth perm",     60,  0, 200);
    TH1F *cal_wLep   = new TH1F("cal_wLep",   "m(l#nu) truth perm",   60,  0, 200);
    TH1F *cal_nuRes  = new TH1F("cal_nuRes",  "pz(#nu) reco-truth",   60,-300,300);

    TFile f(outFile, "recreate");
    TTree tt_tree("ttbar", "ttbar");

    float_t rapidity, cosprod, tl_thetastar, tl_phistar, tl_theta, tl_phi, th_thetastar, th_phistar, th_theta, th_phi;
    tt_tree.Branch("rapidity", &rapidity, "rapidity/F");
    tt_tree.Branch("cosprodangle", &cosprod, "cosprod/F");
    tt_tree.Branch("t_thetaprime", &tl_thetastar, "tl_thetastar/F");
    tt_tree.Branch("t_phistar", &tl_phistar, "tl_phistar/F");
    tt_tree.Branch("t_theta", &tl_theta, "tl_theta/F");
    tt_tree.Branch("t_phi", &tl_phi, "tl_phi/F");
    tt_tree.Branch("t~_thetastar", &th_thetastar, "th_thetastar/F");
    tt_tree.Branch("t~_phistar", &th_phistar, "th_phistar/F");
    tt_tree.Branch("t~_theta", &th_theta, "th_theta/F");
    tt_tree.Branch("t~_phi", &th_phi, "th_phi/F");

    Int_t v_truth_matched, v_bassign_correct, v_downtype_correct, v_nu_branch_correct;
    Int_t v_nu_real, v_lepton_is_positive;
    Float_t v_chi2, v_m_top_lep, v_m_top_had, v_m_w_had, v_m_w_lep, v_m_bb;
    Float_t v_m_vis, v_mT, v_nu_pz_reco, v_nu_pz_truth;
    tt_tree.Branch("truth_matched",      &v_truth_matched,      "truth_matched/I");
    tt_tree.Branch("bassign_correct",    &v_bassign_correct,    "bassign_correct/I");
    tt_tree.Branch("downtype_correct",   &v_downtype_correct,   "downtype_correct/I");
    tt_tree.Branch("nu_branch_correct",  &v_nu_branch_correct,  "nu_branch_correct/I");
    tt_tree.Branch("nu_real",            &v_nu_real,            "nu_real/I");
    tt_tree.Branch("lepton_is_positive", &v_lepton_is_positive, "lepton_is_positive/I");
    tt_tree.Branch("chi2",               &v_chi2,               "chi2/F");
    tt_tree.Branch("m_top_lep",          &v_m_top_lep,          "m_top_lep/F");
    tt_tree.Branch("m_top_had",          &v_m_top_had,          "m_top_had/F");
    tt_tree.Branch("m_w_had",            &v_m_w_had,            "m_w_had/F");
    tt_tree.Branch("m_w_lep",            &v_m_w_lep,            "m_w_lep/F");
    tt_tree.Branch("m_bb",               &v_m_bb,               "m_bb/F");
    tt_tree.Branch("m_vis",              &v_m_vis,              "m_vis/F");
    tt_tree.Branch("mT_vis_met",         &v_mT,                 "mT_vis_met/F");
    tt_tree.Branch("nu_pz_reco",         &v_nu_pz_reco,         "nu_pz_reco/F");
    tt_tree.Branch("nu_pz_truth",        &v_nu_pz_truth,        "nu_pz_truth/F");

    std::cout << "Processing Events..." << std::endl;

    std::vector<int> rejectedPIDS;

    int antiCount = 0;

    // NEW: counters for the validation printout.
    long nPassReco = 0, nNuNotReal = 0, nTruthFound = 0, nTruthMatched = 0;
    long nBTried = 0, nBOK = 0, nDTried = 0, nDOK = 0, nNuTried = 0, nNuOK = 0;
    long nVisTooHeavy = 0;

    for (int n = 0; n < nEvents; n++)
    {
        if (n == 0) {
            std::cout << "\n===== FIRST EVENT PARTICLES (Status 23) =====" << std::endl;
            treeReader->ReadEntry(0);
            int nParticles = branchParticle->GetEntries();
            for (int i = 0; i < nParticles; i++) {
                GenParticle *gen = (GenParticle*) branchParticle->At(i);
                if (gen->Status == 23) {
                    std::cout << "PID: " << gen->PID << " (E,Px,Py,Pz) = (" 
                              << gen->E << "," << gen->Px << "," << gen->Py << "," << gen->Pz << ")" << std::endl;
                }
            }

            std::cout << "--- status codes present in event 0 ---" << std::endl;
            std::unordered_map<int,int> scount;
            for (int i = 0; i < nParticles; i++) {
                GenParticle *gen = (GenParticle*) branchParticle->At(i);
                scount[gen->Status]++;
            }
            for (auto& kv : scount)
                std::cout << "  status " << kv.first << " : " << kv.second << std::endl;
            std::cout << "=========================================\n" << std::endl;
        }
        
        treeReader->ReadEntry(n);
        int nParticles = branchParticle->GetEntries();
        int nJets = branchJet->GetEntries();

        events_total++;

        if (n % progressStep == 0)
        {
            int pct = (int)(100.0 * n / nEvents);
            double elapsed = progressTimer.RealTime();
            progressTimer.Continue();
            double eta = (n > 0) ? elapsed * (nEvents - n) / n : 0.0;
            std::cout << "\r  " << pct << "%  (" << n << " / " << nEvents
                      << " events, " << events_selected << " selected, "
                      << (int)elapsed << "s elapsed, ~" << (int)eta << "s left)      "
                      << std::flush;
        }

        std::vector<TLorentzVector> bottoms, bottombars, leptons, neutrinos, quarks, antiquarks;

        bool anti = false; 

        // ===================================================================
        // Truth Data
        // ===================================================================
        std::vector<TLorentzVector> tr_bottoms, tr_bottombars, tr_leptons,
                                    tr_neutrinos, tr_downtype, tr_uptype;
        int tr_leptonPID = 0;

        for (int i = 0; i < nParticles; i++)
        {
            GenParticle *genParticle = (GenParticle*) branchParticle->At(i);
            if (genParticle->Status == 23) //We only care about outgoing particles from the main process
            {
                if (genParticle->PID == 5) //b
                {
                    tr_bottoms.push_back(genParticle->P4());
                }
                if (genParticle->PID == -5)//b~
                {
                    tr_bottombars.push_back(genParticle->P4());
                }

                if (genParticle->PID == 11 || genParticle->PID == -11 ||
                    genParticle->PID == 13 || genParticle->PID == -13 ||
                    genParticle->PID == 15 || genParticle->PID == -15) //lepton
                {
                    tr_leptonPID = genParticle->PID;
                    tr_leptons.push_back(genParticle->P4());
                }

                if (genParticle->PID == 12 || genParticle->PID == -12 ||
                    genParticle->PID == 14 || genParticle->PID == -14 ||
                    genParticle->PID == 16 || genParticle->PID == -16) // neutrino
                {
                    tr_neutrinos.push_back(genParticle->P4());
                }

                // Down-type vs up-type from PDG parity: |PID| in {1,3} = d,s
                // is down-type, {2,4} = u,c is up-type. This covers both W
                // charges at once, since W+ -> u dbar and W- -> d ubar both
                // put odd |PID| on the down-type fermion.
                int apid = abs(genParticle->PID);
                if (apid == 1 || apid == 3) { tr_downtype.push_back(genParticle->P4()); }
                if (apid == 2 || apid == 4) { tr_uptype.push_back(genParticle->P4()); }
            }
            else {
                rejectedPIDS.push_back(genParticle->Status);
            }
        }

        // ========================
        // RECO OBJECT SELECTION
        // ========================
        std::vector<TLorentzVector> selLeptons;
        std::vector<int>            selCharges;

        for (int i = 0; i < branchElectron->GetEntries(); i++) {
            Electron *el = (Electron*) branchElectron->At(i);
            if (el->PT < kLepPtMin)            continue;
            if (fabs(el->Eta) > kLepEtaMax)    continue;
            if (el->IsolationVar > kLepIsoMax) continue;
            selLeptons.push_back(el->P4());
            selCharges.push_back(el->Charge);
        }
        for (int i = 0; i < branchMuon->GetEntries(); i++) {
            Muon *mu = (Muon*) branchMuon->At(i);
            if (mu->PT < kLepPtMin)            continue;
            if (fabs(mu->Eta) > kLepEtaMax)    continue;
            if (mu->IsolationVar > kLepIsoMax) continue;
            selLeptons.push_back(mu->P4());
            selCharges.push_back(mu->Charge);
        }

        std::vector<TLorentzVector> bJets, lightJets;
        for (int i = 0; i < branchJet->GetEntries(); i++) {
            Jet *jet = (Jet*) branchJet->At(i);
            if (jet->PT < kJetPtMin)         continue;
            if (fabs(jet->Eta) > kJetEtaMax) continue;

            TLorentzVector p = jet->P4();   // uses PT, Eta, Phi AND Mass

            bool overlaps = false;
            for (size_t k = 0; k < selLeptons.size(); k++)
                if (p.DeltaR(selLeptons[k]) < kOverlapDR) overlaps = true;
            if (overlaps) continue;


            if (jet->BTag & 1) bJets.push_back(p);
            else               lightJets.push_back(p);
        }

        TLorentzVector met;
        double metMag = 0.0;
        if (branchMET->GetEntries() > 0) {
            MissingET *m = (MissingET*) branchMET->At(0);
            metMag = m->MET;
            met.SetPxPyPzE(m->MET*cos(m->Phi), m->MET*sin(m->Phi), 0.0, m->MET);
        }


        int    asg_bLepIdx = -1, asg_downIdx = -1, asg_nuIdx = -1;
        bool   asg_ok = false;
        double asg_chi2 = -1.0;
        NuSolSet nu;

        cf_all++;
        if (selLeptons.size() == 1) { cf_lep++;
            if (metMag >= kMetMin)    { cf_met++;
                if (bJets.size() >= 1)    cf_b1++;
                if (bJets.size() == 2)  { cf_b2++;
                    if (lightJets.size() >= 2) cf_lj++; } } }
        if (selLeptons.size() == 1 && bJets.size() == 2 &&
            lightJets.size() >= 2 && metMag >= kMetMin)
        {
            const TLorentzVector& lep = selLeptons[0];
            TLorentzVector wHad = lightJets[0] + lightJets[1];

            TLorentzVector vis = isHiggs ? (lep + wHad) : lep;
            v_m_vis = vis.M();
            if (isHiggs && vis.M() > kMHiggs) nVisTooHeavy++;
            // nu = solveNeutrino(vis, met, kTargetMass);
            NuSolSet nu = solveNeutrino_chi2(lep, bJets, met);
            
            // double top_diff1 = (lep + nus[0] + bJets[0]).M() - kMT;
            // double top_diff2 = (lep + nus[1] + bJets[1]).M() - kMT;

            // top_diff1 < top_diff2 ? (const TLorentzVector& bLep = bJets[0], const TLorentzVector& bHad = bJets[1], nu = nus[0]) 
            //                     : (const TLorentzVector& bLep = bJets[1], const TLorentzVector& bHad = bJets[0], nu = nus[1]);

            // if (!nu.real) nNuNotReal++;

            if (!nu.sols.empty()) {
                // chi2 over 2 b-assignments x (1 or 2) neutrino solutions.
                // double bestChi2 = 1e18;
                // int bestIB = -1, bestK = -1;
                // for (int ib = 0; ib < 2; ib++) {
                //     const TLorentzVector& bLep = bJets[ib];
                //     const TLorentzVector& bHad = bJets[1-ib];
                //     TLorentzVector topHad = wHad + bHad;
                //     for (size_t k = 0; k < nu.sols.size(); k++) {
                //         TLorentzVector wLep   = lep + nu.sols[k];
                //         TLorentzVector topLep = wLep + bLep;
                //         double chi2 = gaussTerm(topLep.M(), kRefTopLep, kSigTopLep)
                //                     + gaussTerm(topHad.M(), kRefTopHad, kSigTopHad)
                //                     + gaussTerm(wHad.M(),   kRefWHad,   kSigWHad);
                //         if (chi2 < bestChi2) { bestChi2 = chi2; bestIB = ib; bestK = (int)k; }
                //     }
                // }

                // full chi2 per b-assignment: leptonic terms from the fit + hadronic terms
                double bestChi2 = 1e18;
                int bestIB = -1, bestK = -1;
                for (int ib = 0; ib < 2; ib++) {
                    TLorentzVector topHad = wHad + bJets[1-ib];
                    double chi2 = nu.chi2[ib]
                                + gaussTerm(topHad.M(), kRefTopHad, kSigTopHad)
                                + gaussTerm(wHad.M(),   kRefWHad,   kSigWHad);
                    if (chi2 < bestChi2) { bestChi2 = chi2; bestIB = ib; bestK = ib; }
                }

                TLorentzVector wLep = lep + nu.sols[bestK];
                TLorentzVector topLep = wLep + bJets[bestIB];

                if (bestIB >= 0) {
                    nPassReco++;

                    int useIB = bestIB, useK = useIB;
                    int trB = -1, trBbar = -1, trDown = -1, trUp = -1, trBestK = -1;
                    bool truthFound =
                        (tr_bottoms.size() == 1 && tr_bottombars.size() == 1 &&
                         tr_leptons.size() == 1 && tr_neutrinos.size() == 1 &&
                         tr_downtype.size() == 1 && tr_uptype.size() == 1 &&
                         abs(tr_leptonPID) != 15);
                    bool truthMatched = false;
                    if (truthFound) {
                        nTruthFound++;
                        bool okB = matchTwo(tr_bottoms[0], tr_bottombars[0], bJets,
                                            kMatchDRJet, trB, trBbar);
                        std::vector<TLorentzVector> lj2;
                        lj2.push_back(lightJets[0]); lj2.push_back(lightJets[1]);
                        bool okL = matchTwo(tr_downtype[0], tr_uptype[0], lj2,
                                            kMatchDRJet, trDown, trUp);
                        bool okLep = (tr_leptons[0].DeltaR(lep) < kMatchDRLep);
                        truthMatched = okB && okL && okLep;
                        if (truthMatched) nTruthMatched++;

                        double bestd = 1e18;
                        for (size_t k = 0; k < nu.sols.size(); k++) {
                            double d = fabs(nu.sols[k].Pz() - tr_neutrinos[0].Pz());
                            if (d < bestd) { bestd = d; trBestK = (int)k; }
                        }
                    }

                    if (kCalibrationMode) {
                        if (!truthMatched || trBestK < 0) { /* leave vectors empty */ }
                        else {
                            bool anti_tmp = (selCharges[0] > 0);
                            useIB = anti_tmp ? trB : trBbar;
                            useK  = trBestK;
                        }
                    }

                    if (!kCalibrationMode || (truthMatched && trBestK >= 0)) {
                        asg_ok      = true;
                        asg_bLepIdx = useIB;
                        asg_nuIdx   = useK;
                        asg_chi2    = bestChi2;

                        const TLorentzVector bLep = bJets[useIB];
                        const TLorentzVector bHad = bJets[1-useIB];
                        TLorentzVector wLep   = lep + nu.sols[useK];
                        TLorentzVector topLep = wLep + bLep;
                        TLorentzVector topHad = wHad + bHad;

                        // down-type candidate among the two light jets
                        asg_downIdx = pickDownTypeIndex(lightJets[0], lightJets[1],
                                                        wHad, bHad);
                        TLorentzVector downJet = lightJets[0];
                        TLorentzVector upJet   = lightJets[1];


                        anti = (selCharges[0] > 0);

                        leptons.push_back(lep);
                        neutrinos.push_back(nu.sols[useK]);


                        //top    = wplus  + bottoms[0]
                        //topbar = wminus + bottombars[0]
                        if (anti) {                 // leptonic W is the W+
                            bottoms.push_back(bLep);
                            bottombars.push_back(bHad);
                        } else {                    // leptonic W is the W-
                            bottoms.push_back(bHad);
                            bottombars.push_back(bLep);
                        }

                        if (anti) {
                            quarks.push_back(downJet);
                            antiquarks.push_back(upJet);
                        } else {
                            antiquarks.push_back(downJet);
                            quarks.push_back(upJet);
                        }

                        v_chi2      = bestChi2;
                        v_m_top_lep = topLep.M();
                        v_m_top_had = topHad.M();
                        v_m_w_had   = wHad.M();
                        v_m_w_lep   = wLep.M();
                        v_m_bb      = (bJets[0] + bJets[1]).M();
                        v_mT        = nu.mT;
                        v_nu_real   = nu.real ? 1 : 0;
                        v_nu_pz_reco  = nu.sols[useK].Pz();
                        v_nu_pz_truth = truthFound ? tr_neutrinos[0].Pz() : -9999.0;
                        v_lepton_is_positive = anti ? 1 : 0;

                        if (kCalibrationMode) {
                            cal_topLep->Fill(v_m_top_lep);
                            cal_topHad->Fill(v_m_top_had);
                            cal_wHad->Fill(v_m_w_had);
                            cal_wLep->Fill(v_m_w_lep);
                            if (truthFound) cal_nuRes->Fill(v_nu_pz_reco - v_nu_pz_truth);
                        }

                        // =============================================
                        // Validation of assignments
                        // =============================================

                        v_truth_matched     = truthMatched ? 1 : 0;
                        v_bassign_correct   = -1;
                        v_downtype_correct  = -1;
                        v_nu_branch_correct = -1;

                        if (truthMatched && !kCalibrationMode) {
                            if (!isHiggs) {
                                int truthBLepIdx = anti ? trB : trBbar;
                                nBTried++;
                                v_bassign_correct = (useIB == truthBLepIdx) ? 1 : 0;
                                if (v_bassign_correct) nBOK++;
                            }
                            nDTried++;
                            v_downtype_correct = (asg_downIdx == trDown) ? 1 : 0;
                            if (v_downtype_correct) nDOK++;
                            // neutrino branch
                            if (trBestK >= 0 && nu.sols.size() > 1) {
                                nNuTried++;
                                v_nu_branch_correct = (useK == trBestK) ? 1 : 0;
                                if (v_nu_branch_correct) nNuOK++;
                            }
                        }
                    }
                }
            }
        }

        if (bottoms.size() != 1 || bottombars.size() != 1 || leptons.size() != 1 || neutrinos.size() != 1 || quarks.size() != 1 || antiquarks.size() != 1)
        {
            continue;
        }

        TLorentzVector wplus;
        TLorentzVector wminus;

        if (anti) {
            wplus = leptons[0] + neutrinos[0];
            wminus = quarks[0] + antiquarks[0];
            antiCount++;
        } else {
            wminus = leptons[0] + neutrinos[0];
            wplus = quarks[0] + antiquarks[0];
        }

        TLorentzVector top = wplus + bottoms[0];
        TLorentzVector topbar = wminus + bottombars[0];
        TLorentzVector ttbar = top + topbar;

        rapidity = ttbar.Rapidity();

        TLorentzVector prodAngleTVec = TLorentzVector(top);
        prodAngleTVec.Boost(-ttbar.BoostVector());
        double_t zeroMomentumT3Vec[3] = {prodAngleTVec.X(), prodAngleTVec.Y(), prodAngleTVec.Z()};
        double_t Z_axis[3] = {0,0,1};
        cosprod = dot(zeroMomentumT3Vec, Z_axis)/(norm(zeroMomentumT3Vec) * norm(Z_axis));

        double_t* t_angles;
        double_t* tbar_angles;

        TVector3 ttb = ttbar.BoostVector();
        top.Boost(-ttb); topbar.Boost(-ttb); wplus.Boost(-ttb); wminus.Boost(-ttb); 
        leptons[0].Boost(-ttb); neutrinos[0].Boost(-ttb); quarks[0].Boost(-ttb); antiquarks[0].Boost(-ttb);
        if (anti)
        {
            t_angles = calcAngles(top, wplus, leptons[0], 1);
            tbar_angles = calcAngles(topbar, wminus, quarks[0], 1);
        }
        else {
            t_angles = calcAngles(top, wplus, antiquarks[0], 1);
            tbar_angles = calcAngles(topbar, wminus, leptons[0], 1);
        }

        if (t_angles[0] == -10 || tbar_angles[0] == -10)
        {
            continue;
        }

        if(anti)
        {
            tl_theta = t_angles[0];
            tl_phi = t_angles[1];
            tl_thetastar = t_angles[2];
            tl_phistar = t_angles[3];
            th_theta = tbar_angles[0];
            th_phi = tbar_angles[1];
            th_thetastar = tbar_angles[2];
            th_phistar = tbar_angles[3];
        }
        else {
            tl_theta = tbar_angles[0];
            tl_phi = tbar_angles[1];
            tl_thetastar = tbar_angles[2];
            tl_phistar = tbar_angles[3];
            th_theta = t_angles[0];
            th_phi = t_angles[1];
            th_thetastar = t_angles[2];
            th_phistar = t_angles[3];
        }

        rapidity_hist->Fill(rapidity);
        cosprod_hist->Fill(cosprod);
        t_theta_hist->Fill(tl_theta);
        t_phi_hist->Fill(tl_phi);
        t_thetastar_hist->Fill(tl_thetastar);
        t_phistar_hist->Fill(tl_phistar);
        tbar_theta_hist->Fill(th_theta);
        tbar_phi_hist->Fill(th_phi);
        tbar_thetastar_hist->Fill(th_thetastar);
        tbar_phistar_hist->Fill(th_phistar);

        tt_tree.Fill();
        events_selected++;
        }

    progressTimer.Stop();
    std::cout << "\r  100%  (" << nEvents << " / " << nEvents << " events, "
              << events_selected << " selected, "
              << (int)progressTimer.RealTime() << "s elapsed)                      "
              << std::endl;

    tt_tree.Write();

    std::cout << "Completed Angle Calculations" << std::endl;
    
    std::unordered_map<Int_t, int> pid_counts;

    for (int pid : rejectedPIDS)
    {
        pid_counts[pid]++;
    }

    for (auto pair : pid_counts)
    {
        std::cout << "Rejected PID " << pair.first << ". Count " << pair.second << std::endl;
    }


    std::cout << "Selection efficiency:      " << (100.0 * events_selected / events_total) << "%" << std::endl;
    std::cout << "Number of anti events " << antiCount << std::endl;

    // =======================
    // Validation summary
    // =======================
    std::cout << "\n============ RECONSTRUCTION VALIDATION ============" << std::endl;
    std::cout << "hypothesis                    : " << (isHiggs ? "HIGGS" : "TTBAR") << std::endl;
    std::cout << "events read                   : " << events_total << std::endl;
    std::cout << "reconstruction succeeded      : " << nPassReco << std::endl;
    std::cout << "written to tree               : " << events_selected << std::endl;
    if (nPassReco > 0) {
        std::cout << "no real nu solution           : " << nNuNotReal
                  << "  (" << 100.0*nNuNotReal/nPassReco << "%)"
                  << "   == fraction with mT > m_target" << std::endl;
        if (isHiggs)
            std::cout << "m(l+jj) > mH  (unsolvable)    : " << nVisTooHeavy
                      << "  (" << 100.0*nVisTooHeavy/nPassReco << "%)" << std::endl;
        std::cout << "truth partons found           : " << nTruthFound
                  << "  (" << 100.0*nTruthFound/nPassReco << "%)" << std::endl;
    }
    if (nTruthFound > 0)
        std::cout << "truth matched to reco         : " << nTruthMatched
                  << "  (" << 100.0*nTruthMatched/nTruthFound << "%)" << std::endl;
    std::cout << "---- assignment accuracy (truth-matched events only) ----" << std::endl;
    if (nBTried > 0)
        std::cout << "b-jet assignment correct      : " << 100.0*nBOK/nBTried
                  << "%  (" << nBOK << "/" << nBTried << ")"
                  << "   [ref 62/74% ATLAS 1204.0952, ~80% 1207.0823]" << std::endl;
    else
        std::cout << "b-jet assignment correct      : N/A (no truth definition)" << std::endl;
    if (nDTried > 0)
        std::cout << "down-type jet correct         : " << 100.0*nDOK/nDTried
                  << "%  (" << nDOK << "/" << nDTried << ")"
                  << "   [ref 61% 1401.3021, 60% 1012.3093]" << std::endl;
    if (nNuTried > 0)
        std::cout << "neutrino branch correct       : " << 100.0*nNuOK/nNuTried
                  << "%  (" << nNuOK << "/" << nNuTried << ")" << std::endl;
    std::cout << "===================================================\n" << std::endl;

    if (kCalibrationMode) {
        std::cout << "---- CALIBRATION (truth assignment) ----" << std::endl;
        std::cout << "m(l nu b) mean " << cal_topLep->GetMean()
                  << "  RMS " << cal_topLep->GetRMS() << "  -> kRefTopLep/kSigTopLep" << std::endl;
        std::cout << "m(j j b)  mean " << cal_topHad->GetMean()
                  << "  RMS " << cal_topHad->GetRMS() << "  -> kRefTopHad/kSigTopHad" << std::endl;
        std::cout << "m(j j)    mean " << cal_wHad->GetMean()
                  << "  RMS " << cal_wHad->GetRMS() << "  -> kRefWHad/kSigWHad" << std::endl;
        std::cout << "m(nu l)    mean " << cal_wLep->GetMean()
                  << "  RMS " << cal_wLep->GetRMS() << "  -> kRefWLep/kSigWLep" << std::endl;
        std::cout << "pz(nu) reco-truth mean " << cal_nuRes->GetMean()
                  << "  RMS " << cal_nuRes->GetRMS() << " GeV" << std::endl;
        std::cout << "(RMS overestimates the core width if there are tails;"
                     " fit a Gaussian over the peak for a better sigma)\n" << std::endl;
    }

    drawHist("leptonic t #theta", t_theta_hist, "cos(#theta)");
    drawHist("leptonic t #phi", t_phi_hist, "#phi");
    drawHist("leptonic t #theta'", t_thetastar_hist, "cos(#theta*)");
    drawHist("leptonic t #phi'", t_phistar_hist, "#phi*");
    drawHist("hadronic t #theta", tbar_theta_hist, "cos(#theta)");
    drawHist("hadronic t #phi", tbar_phi_hist, "#phi");
    drawHist("hadronic #theta '", tbar_thetastar_hist, "cos(#theta*)");
    drawHist("hadronic #phi' '", tbar_phistar_hist, "#phi*");
    drawHist("rapidity", rapidity_hist, "rapidity");
    drawHist("production angle", cosprod_hist, "cos(prod)");

    std::cout << cf_all << " events read, " << cf_lep << " with 1 lepton, "
              << cf_met << " with MET > " << kMetMin << ", "
              << cf_b1 << " with >= 1 b-jet, "
              << cf_b2 << " with 2 b-jets, "
              << cf_lj << " with >= 2 light jets" << std::endl;

    std::cout << "Histogramms Created. Done!" << std::endl;


}
