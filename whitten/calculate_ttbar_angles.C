#ifdef __CLING__
R__ADD_INCLUDE_PATH(/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/MG5_aMC_v3_7_0/Delphes)
R__ADD_INCLUDE_PATH(/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/MG5_aMC_v3_7_0/Delphes/external)
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include <cmath>
#endif

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
    std::string fileName = "plots/ttTObbWW/" + title + ".pdf";
    m1->SaveAs(fileName.c_str());
}

TLorentzVector fitNeutrino_Chi2(const TLorentzVector& lep, const TLorentzVector& bLep,
                           double metx, double mety)
{
    const double mW = kMassW,  sW = 2.1;
    const double mT = kMassT, sT = 15.0;

    auto chi2 = [&](double pz) {
        TLorentzVector nu;
        nu.SetXYZM(metx, mety, pz, 0.);
        double cW = ((lep + nu).M()        - mW) / sW;
        double cT = ((lep + nu + bLep).M() - mT) / sT;
        return cW*cW + cT*cT;
    };

    ROOT::Math::Functor1D func(chi2);
    ROOT::Math::BrentMinimizer1D bm;
    bm.SetFunction(func, -3000., 3000.);
    bm.SetNpx(1000);              // coarse scan first to find the global minimum
    bm.Minimize(200, 1e-4, 1e-6);

    TLorentzVector nu;
    nu.SetXYZM(metx, mety, bm.XMinimum(), 0.);
    return nu;
}

struct NuSolSet {
    std::vector<TLorentzVector> sols;
    bool   real;
    double disc;
    double mT;
};

NuSolSet solveNeutrinoRaw(const TLorentzVector& lep, const TLorentzVector& met,
                          double targetMass)
{
    NuSolSet out;
    out.real = false; out.disc = 0.0; out.mT = 0.0;

    const double pxn = met.Px(), pyn = met.Py();
    const double ptn2 = pxn*pxn + pyn*pyn;
    const double ptn  = sqrt(ptn2);

    const double mv2 = lep.M2();
    const double A   = mv2 + lep.Pt()*lep.Pt();
    if (A < 1e-9) return out;

    const double Lambda = 0.5*(targetMass*targetMass - mv2) + lep.Px()*pxn + lep.Py()*pyn;
    const double disc   = Lambda*Lambda - A*ptn2;
    out.disc = disc;
    out.real = (disc >= 0.0);

    const double mTV = sqrt(A);
    out.mT = sqrt(std::max(0.0, mv2 + 2.0*(mTV*ptn - (lep.Px()*pxn + lep.Py()*pyn))));

    const double dUse = (disc < 0.0) ? 0.0 : disc;
    const double root = lep.E()*sqrt(dUse);

    const double pz1 = (Lambda*lep.Pz() + root)/A;
    TLorentzVector nu1; nu1.SetPxPyPzE(pxn, pyn, pz1, sqrt(ptn2 + pz1*pz1));
    out.sols.push_back(nu1);

    if (out.real && root > 0.0) {
        const double pz2 = (Lambda*lep.Pz() - root)/A;
        TLorentzVector nu2; nu2.SetPxPyPzE(pxn, pyn, pz2, sqrt(ptn2 + pz2*pz2));
        out.sols.push_back(nu2);
    }
    return out;
}

NuSolSet solveNeutrino(const TLorentzVector& lep, const TLorentzVector& met,
                       double targetMass)
{
    NuSolSet first = solveNeutrinoRaw(lep, met, targetMass);
    if (first.real || false) return first;

    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 40; ++it) {
        double mid = 0.5*(lo + hi);
        TLorentzVector sc; sc.SetPxPyPzE(met.Px()*mid, met.Py()*mid, 0.0, met.Pt()*mid);
        if (solveNeutrinoRaw(lep, sc, targetMass).real) lo = mid; else hi = mid;
    }
    TLorentzVector sc; sc.SetPxPyPzE(met.Px()*lo, met.Py()*lo, 0.0, met.Pt()*lo);
    NuSolSet out = solveNeutrinoRaw(lep, sc, targetMass);
    out.real = false;
    return out;
}


bool passObj(const TLorentzVector& p, double ptMin, double etaMax)
{
    return p.Pt() >= ptMin && std::fabs(p.Eta()) <= etaMax;
}

void calculate_ttbar_angles()
{
    bool debug = false;

    gSystem->Load("libDelphes");

    TChain chain("Delphes");
    chain.Add("/Users/alexwhitten/Documents/HongResearch/10-angle-analysis/generatedEvents/full_diHiggs/Events/run_01/tag_1_delphes_events.root");

    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
    TClonesArray *branchGenMET = treeReader->UseBranch("GenMissingET");

    int nEvents = treeReader->GetEntries();

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
    TH1F *t_phistar_hist = new TH1F("test", "test", 50, -TMath::Pi(), TMath::Pi());
    TH1F *tbar_theta_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *tbar_phi_hist = new TH1F("test", "test", 50, -3.14, 3.14);
    TH1F *tbar_thetastar_hist = new TH1F("test", "test", 50, -1, 1);
    TH1F *tbar_phistar_hist = new TH1F("test", "test", 50, -TMath::Pi(), TMath::Pi());
    TH1F *rapidity_hist = new TH1F("test", "test", 50, -5,5);
    TH1F *cosprod_hist = new TH1F("test", "test", 50, -1,1);
    TH1F *met_diff = new TH1F("test", "test", 50, -200,200);



    TFile f("./rootFiles/full_hh_angles.root", "recreate");
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

    std::cout << "Processing Events..." << std::endl;

    std::vector<int> rejectedPIDS;

    int antiCount = 0;

    for (int n = 0; n < nEvents; n++)
    {
        if (n == 0) {
            // Print first event's particles for debugging
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
            std::cout << "=========================================\n" << std::endl;
        }
        
        treeReader->ReadEntry(n);
        int nParticles = branchParticle->GetEntries();
        int nJets = branchGenJet->GetEntries();

        events_total++;

        //Lorentz Vectors for all of the particles we use.
        std::vector<TLorentzVector> bottoms, bottombars, leptons, neutrinos, quarks, antiquarks;

        bool anti = false; 
        //Get Truth Particles
        for (int i = 0; i < nParticles; i++)
        {
            GenParticle *genParticle = (GenParticle*) branchParticle->At(i);
            if (genParticle->Status == 23) //We only care about outgoing particles from the main process
            {
                if (genParticle->PID == 5) //b
                {
                    bottoms.push_back(genParticle->P4());
                }
                if (genParticle->PID == -5)//b~
                {
                    bottombars.push_back(genParticle->P4());
                }

                if (genParticle->PID == 11 || genParticle->PID == -11 ||
                    genParticle->PID == 13 || genParticle->PID == -13 ) //lepton
                {
                    //std::cout << "Lepton " << genParticle->PID << std::endl;
                    anti = genParticle->PID < 0;
                    leptons.push_back(genParticle->P4());
                }

                if (genParticle->PID == 12 || genParticle->PID == -12 ||
                    genParticle->PID == 14 || genParticle->PID == -14 ||
                    genParticle->PID == 16 || genParticle->PID == -16) // neutrino
                {
                    //std::cout << "Neutrino " << genParticle->PID << std::endl;
                    neutrinos.push_back(genParticle->P4());
                }

                if (genParticle->PID == 1 || genParticle->PID == 2 || genParticle->PID == 3 || genParticle->PID == 4)
                {
                    quarks.push_back(genParticle->P4());
                }
                if (genParticle->PID == -1 || genParticle->PID == -2 || genParticle->PID == -3 || genParticle->PID == -4)
                {
                    antiquarks.push_back(genParticle->P4());
                }
                
            }
            else {
                //rejectedPIDS.push_back(genParticle->Status);
            }
        }
        if (bottoms.size() != 1 || bottombars.size() != 1 || leptons.size() != 1 || neutrinos.size() != 1 || quarks.size() != 1 || antiquarks.size() != 1)
        {
            continue;
        }

        MissingET* met = (MissingET*) branchGenMET->At(0);
        double metx = met->MET*cos(met->Phi);
        double mety = met->MET*sin(met->Phi);

        TLorentzVector wplus;
        TLorentzVector wminus;

        TLorentzVector lepB = anti ? bottoms[0] : bottombars[0];
        TLorentzVector nu   = fitNeutrino_Chi2(leptons[0], lepB, metx, mety);
        // NuSolSet nu_out   = solveNeutrino(leptons[0], TLorentzVector(metx, mety, 0, met->MET), kMassW);
        // TLorentzVector others = leptons[0] + (anti ? bottoms[0] : bottombars[0]);
        // TLorentzVector nu = (fabs((nu_out.sols[0] + others).M() - kMassT) < fabs((nu_out.sols[1] + others).M() - kMassT)) ? nu_out.sols[0] : nu_out.sols[1];
        met_diff->Fill(nu.Pz() - neutrinos[0].Pz());
        neutrinos[0] = nu; // Replace the neutrino with the fitted one

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
        t_theta_hist->Fill(tl_theta);
        t_phi_hist->Fill(tl_phi);
        t_thetastar_hist->Fill(tl_thetastar);
        t_phistar_hist->Fill(tl_phistar);
        tbar_theta_hist->Fill(th_theta);
        tbar_phi_hist->Fill(th_phi);
        tbar_thetastar_hist->Fill(th_thetastar);
        tbar_phistar_hist->Fill(th_phistar);

        rapidity_hist->Fill(rapidity);
        cosprod_hist->Fill(cosprod);

        tt_tree.Fill();
        events_selected++;
        }
    
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

    drawHist("leptonic t #theta", t_theta_hist, "cos(#theta)");
    drawHist("leptonic t #phi", t_phi_hist, "#phi");
    drawHist("leptonic t #theta'", t_thetastar_hist, "cos(#theta*)");
    drawHist("leptonic t #phi'", t_phistar_hist, "#phi*");
    drawHist("hadronic t #theta", tbar_theta_hist, "cos(#theta)");
    drawHist("hadronic t #phi", tbar_phi_hist, "#phi");
    drawHist("hadronic #theta '", tbar_thetastar_hist, "cos(#theta*)");
    drawHist("hadronic #phi' ", tbar_phistar_hist, "#phi*");
    drawHist("rapidity", rapidity_hist, "rapidity");
    drawHist("production angle", cosprod_hist, "cos(prod)");
    drawHist("Pz difference", met_diff, "Pz difference (GeV)");
    std::cout << "Pz RMS: " << met_diff->GetRMS() << std::endl;

    std::cout << "Histogramms Created. Done!" << std::endl;


}

