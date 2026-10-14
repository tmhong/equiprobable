// ============================================================================
//  compare_paired.C
//
//  Reads the TWO trees written by chi2_truth_matched_down_recon_angles.C and
//  produces ONE multi-page pdf:
//
//    page 1  ALL truth  vs  selected reco
//            truth_all tree against the ttbar tree. DIFFERENT event sets, so
//            the difference is acceptance + selection + smearing combined.
//
//    page 2  selected reco  vs  its OWN truth
//            both from the ttbar tree, cut on truth_angles_valid==1. SAME
//            events, so selection drops out and this is smearing alone.
//
//    page 3  per-event angular resolution, reco - truth, same events.
//
//    page 4  neutrino pz: reco-input fit vs truth-input fit vs the true pz,
//            plus the reco-vs-true migration.
//
//  Pages 1 and 2 answer different questions and the pair is the point: page 1
//  is what an unfolding study needs, page 2 is what "how well do I measure
//  this angle" needs. A single overlay cannot do both.
//
//  Run:
//    cd /home/cicero/Pitt/Hong/sim-tools/analysis
//    root -l 'compare_paired.C("rootFiles/ttbar_angles_truthdown.root","ttbar")'
// ============================================================================

const int kN = 10;

// Branch names. The ttbar tree holds reco under plain names and truth under a
// truth_ prefix; the truth_all tree holds truth under the PLAIN names.
const char *kRecoBr[kN] = {
    "t_theta",  "t_phi",  "t_thetaprime", "t_phistar",
    "t~_theta", "t~_phi", "t~_thetastar", "t~_phistar",
    "rapidity", "cosprodangle" };
const char *kTruthBr[kN] = {
    "truth_t_theta",  "truth_t_phi",  "truth_t_thetaprime", "truth_t_phistar",
    "truth_t~_theta", "truth_t~_phi", "truth_t~_thetastar", "truth_t~_phistar",
    "truth_rapidity", "truth_cosprodangle" };
const char *kShort[kN] = {
    "lep_costheta", "lep_phi", "lep_costhetastar", "lep_phistar",
    "had_costheta", "had_phi", "had_costhetastar", "had_phistar",
    "rapidity",     "cosprod" };
const char *kXlab[kN] = {
    "cos#theta (lep)",  "#phi (lep)",
    "cos#theta* (lep)", "#phi* (lep)",
    "cos#theta (had)",  "#phi (had)",
    "cos#theta* (had)", "#phi* (had)",
    "y(t#bar{t})",      "cos#Theta_{prod}" };
// The four theta-type branches store cos(theta), not theta, so they live on
// [-1,1]. calcAngles returns getCosth(), i.e. a cosine.
const double kLo[kN] = {-1, -3.1416, -1, -3.1416, -1, -3.1416, -1, -3.1416, -5, -1};
const double kHi[kN] = { 1,  3.1416,  1,  3.1416,  1,  3.1416,  1,  3.1416,  5,  1};
// 1 => the variable is an azimuth, so differences must be wrapped to (-pi,pi]
const int    kIsPhi[kN] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 0 };
// half-width of the (reco - truth) axis
const double kResHalf[kN] = { 1, 3.1416, 1, 3.1416, 1, 3.1416, 1, 3.1416, 2, 1 };


// Lay out a 4x3 grid of overlay + ratio pairs on one canvas.
static void gridOverlay(TCanvas *c, TH1F **hA, TH1F **hB,
                        const char *labA, const char *labB,
                        const char *note1, const char *note2,
                        const char *note3, double *tv)
{
    const int NCOL = 4, NROW = 3;
    c->cd(0);
    for (int i = 0; i < kN; ++i) {
        if (hA[i]->Integral() > 0) hA[i]->Scale(1.0 / hA[i]->Integral());
        if (hB[i]->Integral() > 0) hB[i]->Scale(1.0 / hB[i]->Integral());

        tv[i] = 0.0;
        for (int k = 1; k <= hA[i]->GetNbinsX(); ++k)
            tv[i] += fabs(hA[i]->GetBinContent(k) - hB[i]->GetBinContent(k));
        tv[i] *= 0.5;

        const int col = i % NCOL, row = i / NCOL;
        const double x1 = col/(double)NCOL, x2 = (col+1)/(double)NCOL;
        const double yT = 1.0 - row/(double)NROW, yB = 1.0 - (row+1)/(double)NROW;
        const double yS = yB + 0.32*(yT - yB);

        c->cd(0);
        TPad *pu = new TPad(Form("%s_pu%d", c->GetName(), i), "", x1, yS, x2, yT);
        TPad *pd = new TPad(Form("%s_pd%d", c->GetName(), i), "", x1, yB, x2, yS);
        pu->SetTopMargin(0.04); pu->SetBottomMargin(0.02);
        pu->SetLeftMargin(0.17); pu->SetRightMargin(0.04);
        pd->SetTopMargin(0.03); pd->SetBottomMargin(0.38);
        pd->SetLeftMargin(0.17); pd->SetRightMargin(0.04);
        pu->Draw(); pd->Draw();

        pu->cd();
        hA[i]->SetLineColor(kBlack); hA[i]->SetLineWidth(2);
        hB[i]->SetLineColor(kBlue);  hB[i]->SetLineWidth(2);
        hA[i]->SetMinimum(0.0);
        hA[i]->SetMaximum(1.35*TMath::Max(hA[i]->GetMaximum(), hB[i]->GetMaximum()));
        hA[i]->GetXaxis()->SetLabelSize(0);
        hA[i]->GetYaxis()->SetTitle("fraction");
        hA[i]->GetYaxis()->SetTitleSize(0.075);
        hA[i]->GetYaxis()->SetTitleOffset(1.05);
        hA[i]->GetYaxis()->SetLabelSize(0.060);
        hA[i]->GetYaxis()->SetNdivisions(505);
        hA[i]->Draw("HIST"); hB[i]->Draw("HIST SAME");
        TLatex *tl = new TLatex(); tl->SetNDC(); tl->SetTextSize(0.075);
        tl->DrawLatex(0.22, 0.86, Form("TV = %.3f", tv[i]));

        pd->cd();
        TH1F *r = (TH1F*) hB[i]->Clone(Form("%s_r%d", c->GetName(), i));
        r->Divide(hA[i]);                  // empty denominator bins give 0
        r->SetLineColor(kBlue); r->SetLineWidth(2);
        r->SetMinimum(0.0); r->SetMaximum(2.0);
        r->GetYaxis()->SetTitle("B / A");
        r->GetYaxis()->SetTitleSize(0.135);
        r->GetYaxis()->SetTitleOffset(0.58);
        r->GetYaxis()->SetLabelSize(0.115);
        r->GetYaxis()->SetNdivisions(503);
        r->GetXaxis()->SetTitle(kXlab[i]);
        r->GetXaxis()->SetTitleSize(0.155);
        r->GetXaxis()->SetTitleOffset(1.00);
        r->GetXaxis()->SetLabelSize(0.125);
        r->GetXaxis()->SetNdivisions(505);
        r->Draw("HIST");
        TLine *one = new TLine(kLo[i], 1.0, kHi[i], 1.0);
        one->SetLineStyle(2); one->SetLineColor(kGray+2); one->Draw();
    }

    // twelfth cell: legend and notes
    const int col = kN % NCOL, row = kN / NCOL;
    const double x1 = col/(double)NCOL, x2 = (col+1)/(double)NCOL;
    const double yT = 1.0 - row/(double)NROW, yB = 1.0 - (row+1)/(double)NROW;
    c->cd(0);
    TPad *pl = new TPad(Form("%s_pl", c->GetName()), "", x1, yB, x2, yT);
    pl->Draw(); pl->cd();
    TLegend *lg = new TLegend(0.08, 0.66, 0.94, 0.90);
    lg->AddEntry(hA[0], labA, "l");
    lg->AddEntry(hB[0], labB, "l");
    lg->SetTextSize(0.070); lg->SetBorderSize(0); lg->Draw();
    TLatex *tx = new TLatex(); tx->SetNDC(); tx->SetTextSize(0.055);
    tx->DrawLatex(0.08, 0.56, note1);
    tx->DrawLatex(0.08, 0.48, note2);
    tx->SetTextSize(0.046);
    tx->DrawLatex(0.08, 0.36, note3);
    tx->DrawLatex(0.08, 0.22, "ratio panel = B / A");
    tx->DrawLatex(0.08, 0.14, "TV = total variation");
}


void compare_paired(const char *filename = "rootFiles/hh_angles_truth_recon.root",
                    const char *tag      = "dihiggs")
{
    TFile *f = TFile::Open(filename);
    if (!f || f->IsZombie()) { printf("cannot open %s\n", filename); return; }
    TTree *tp = (TTree*) f->Get("ttbar");       // paired reco + truth
    TTree *ta = (TTree*) f->Get("truth_all");   // truth, reco selection not applied
    if (!tp) { printf("no tree 'ttbar' in %s\n", filename); return; }
    if (!ta) printf("WARNING: no tree 'truth_all' -- page 1 will be skipped\n");
    if (!tp->GetBranch("truth_t_thetaprime")) {
        printf("ERROR: no truth_* branches. Rerun the macro with Block E.\n"); return;
    }

    gSystem->mkdir("plots", kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    const Long64_t nP = tp->GetEntries();
    const Long64_t nA = ta ? ta->GetEntries() : 0;
    const Long64_t nV = (Long64_t) tp->GetEntries("truth_angles_valid==1");
    printf("\n%s:  ttbar tree %lld   truth_all tree %lld   paired-valid %lld\n\n",
           tag, nP, nA, nV);

    TString out = Form("plots/compare_paired_%s.pdf", tag);
    double tv1[kN], tv2[kN];

    // ================= page 1: all truth vs selected reco =================
    if (ta) {
        TH1F *hA[kN], *hB[kN];
        for (int i = 0; i < kN; ++i) {
            hA[i] = new TH1F(Form("p1a%d", i), "", 50, kLo[i], kHi[i]);
            hB[i] = new TH1F(Form("p1b%d", i), "", 50, kLo[i], kHi[i]);
            // truth_all uses the PLAIN names; ttbar's plain names are the reco ones
            ta->Draw(Form("%s>>p1a%d", kRecoBr[i], i), "", "goff");
            tp->Draw(Form("%s>>p1b%d", kRecoBr[i], i), "", "goff");
        }
        TCanvas *c1 = new TCanvas("c1", "all truth vs selected reco", 1800, 1500);
        gridOverlay(c1, hA, hB, "A: all truth", "B: selected reco",
                    Form("sample: %s", tag),
                    Form("A %lld   B %lld entries", nA, nP),
                    "DIFFERENT events: acceptance", tv1);
        c1->Print((out + "(").Data());
    }

    // ============ page 2: selected reco vs its own truth (same events) ============
    {
        TH1F *hA[kN], *hB[kN];
        for (int i = 0; i < kN; ++i) {
            hA[i] = new TH1F(Form("p2a%d", i), "", 50, kLo[i], kHi[i]);
            hB[i] = new TH1F(Form("p2b%d", i), "", 50, kLo[i], kHi[i]);
            tp->Draw(Form("%s>>p2a%d", kTruthBr[i], i), "truth_angles_valid==1", "goff");
            tp->Draw(Form("%s>>p2b%d", kRecoBr[i],  i), "truth_angles_valid==1", "goff");
        }
        TCanvas *c2 = new TCanvas("c2", "paired truth vs reco", 1800, 1500);
        gridOverlay(c2, hA, hB, "A: truth (paired)", "B: reco",
                    Form("sample: %s", tag),
                    Form("%lld events, both sides", nV),
                    "SAME events: smearing only", tv2);
        if (ta) c2->Print(out.Data()); else c2->Print((out + "(").Data());
    }

    // ================= page 3: per-event angular resolution =================
    // SetBranchAddress rather than Draw expressions, because the "t~_..."
    // branch names contain a tilde and arithmetic on them inside a TTreeFormula
    // is fragile.
    {
        Float_t vr[kN], vt[kN];
        Int_t valid = 0;
        for (int i = 0; i < kN; ++i) {
            tp->SetBranchAddress(kRecoBr[i],  &vr[i]);
            tp->SetBranchAddress(kTruthBr[i], &vt[i]);
        }
        tp->SetBranchAddress("truth_angles_valid", &valid);

        TH1F *hD[kN];
        for (int i = 0; i < kN; ++i)
            hD[i] = new TH1F(Form("p3d%d", i), "", 80, -kResHalf[i], kResHalf[i]);

        for (Long64_t e = 0; e < nP; ++e) {
            tp->GetEntry(e);
            if (valid != 1) continue;
            for (int i = 0; i < kN; ++i) {
                double d = vr[i] - vt[i];
                // azimuths are periodic: without wrapping, a 0.05 rad shift
                // across the +/-pi boundary would be recorded as ~6.2 rad
                if (kIsPhi[i]) d = TVector2::Phi_mpi_pi(d);
                hD[i]->Fill(d);
            }
        }

        TCanvas *c3 = new TCanvas("c3", "per-event resolution", 1800, 1200);
        c3->Divide(4, 3);
        printf("%-18s  %9s %9s   %9s %9s   %7s %7s\n",
               "variable", "TV(all)", "TV(same)", "res mean", "res RMS", "", "");
        for (int i = 0; i < kN; ++i) {
            c3->cd(i+1);
            hD[i]->SetLineColor(kRed); hD[i]->SetLineWidth(2);
            hD[i]->GetXaxis()->SetTitle(Form("#Delta %s", kXlab[i]));
            hD[i]->GetYaxis()->SetTitle("events");
            hD[i]->Draw("HIST");
            TLine *z = new TLine(0, 0, 0, 1.05*hD[i]->GetMaximum());
            z->SetLineStyle(2); z->SetLineColor(kGray+2); z->Draw();
            TLatex *t = new TLatex(); t->SetNDC(); t->SetTextSize(0.055);
            t->DrawLatex(0.15, 0.85, Form("mean %+.3f", hD[i]->GetMean()));
            t->DrawLatex(0.15, 0.78, Form("RMS  %.3f",  hD[i]->GetRMS()));
            printf("%-18s  %9.3f %9.3f   %9.4f %9.4f\n",
                   kShort[i], tv1[i], tv2[i], hD[i]->GetMean(), hD[i]->GetRMS());
        }
        c3->cd(kN+1);
        TLatex *tx = new TLatex(); tx->SetNDC(); tx->SetTextSize(0.075);
        tx->DrawLatex(0.08, 0.80, "reco - truth, per event");
        tx->DrawLatex(0.08, 0.68, Form("%lld events", nV));
        tx->SetTextSize(0.060);
        tx->DrawLatex(0.08, 0.50, "azimuths wrapped to");
        tx->DrawLatex(0.08, 0.42, "(-#pi, #pi] before");
        tx->DrawLatex(0.08, 0.34, "differencing");
        c3->Print(out.Data());

        tp->ResetBranchAddresses();   // or the stale pointers bite the next Draw
    }

    // ================= page 4: neutrino pz =================
    {
        TCanvas *c4 = new TCanvas("c4", "neutrino pz", 1600, 1100);
        c4->Divide(2, 2);

        // sentinel: nu_pz_truth and truth_nu_pz are -9999 when truth is absent
        const char *cut = "nu_pz_truth>-9000 && truth_nu_pz>-9000";

        // (1) the quantity asked for: reco-input fit vs truth-input fit
        c4->cd(1);
        TH1F *d1 = new TH1F("d1", "", 100, -400, 400);
        tp->Draw("nu_pz_reco - truth_nu_pz>>d1", cut, "goff");
        d1->SetLineColor(kBlue); d1->SetLineWidth(2);
        d1->GetXaxis()->SetTitle("p_{z}^{reco fit} - p_{z}^{truth fit} [GeV]");
        d1->GetYaxis()->SetTitle("events");
        d1->Draw("HIST");
        TLatex l1; l1.SetNDC(); l1.SetTextSize(0.040);
        l1.DrawLatex(0.14, 0.86, "reco-input fit - truth-input fit");
        l1.DrawLatex(0.14, 0.80, Form("mean %+.2f  RMS %.2f GeV", d1->GetMean(), d1->GetRMS()));

        // (2) each against the ACTUAL true pz
        c4->cd(2);
        TH1F *d2 = new TH1F("d2", "", 100, -400, 400);   // reco fit  - true
        TH1F *d3 = new TH1F("d3", "", 100, -400, 400);   // truth fit - true
        tp->Draw("nu_pz_reco  - nu_pz_truth>>d2", cut, "goff");
        tp->Draw("truth_nu_pz - nu_pz_truth>>d3", cut, "goff");
        if (d2->Integral() > 0) d2->Scale(1.0/d2->Integral());
        if (d3->Integral() > 0) d3->Scale(1.0/d3->Integral());
        d2->SetLineColor(kBlue);  d2->SetLineWidth(2);
        d3->SetLineColor(kRed);   d3->SetLineWidth(2);
        d3->SetMaximum(1.3*TMath::Max(d2->GetMaximum(), d3->GetMaximum()));
        d3->GetXaxis()->SetTitle("p_{z}^{fit} - p_{z}^{true} [GeV]");
        d3->GetYaxis()->SetTitle("fraction");
        d3->Draw("HIST"); d2->Draw("HIST SAME");
        TLegend lg2(0.56, 0.72, 0.88, 0.88);
        lg2.AddEntry(d2, "reco inputs", "l");
        lg2.AddEntry(d3, "truth inputs", "l");
        lg2.SetBorderSize(0); lg2.Draw();

        // (3) migration: does the reco pz track the true pz at all?
        c4->cd(3);
        TH2F *m = new TH2F("m", "", 60, -400, 400, 60, -400, 400);
        tp->Draw("nu_pz_reco:nu_pz_truth>>m", cut, "goff");
        m->GetXaxis()->SetTitle("p_{z}^{true} [GeV]");
        m->GetYaxis()->SetTitle("p_{z}^{reco fit} [GeV]");
        m->Draw("COLZ");
        TLine diag(-400, -400, 400, 400);
        diag.SetLineStyle(2); diag.SetLineColor(kGray+2); diag.Draw();

        c4->cd(4);
        TLatex t4; t4.SetNDC(); t4.SetTextSize(0.050);
        t4.DrawLatex(0.06, 0.88, "neutrino p_{z}");
        t4.SetTextSize(0.040);
        t4.DrawLatex(0.06, 0.78, Form("reco fit  - truth fit : RMS %.1f GeV", d1->GetRMS()));
        t4.DrawLatex(0.06, 0.70, Form("reco fit  - true      : RMS %.1f GeV",
                                      d2->GetRMS()));
        t4.DrawLatex(0.06, 0.64, Form("truth fit - true      : RMS %.1f GeV",
                                      d3->GetRMS()));
        t4.DrawLatex(0.06, 0.58, Form("correlation(reco, true) = %.3f",
                                      m->GetCorrelationFactor()));
        t4.SetTextSize(0.034);
        t4.DrawLatex(0.06, 0.44, "A low correlation means the reco p_{z}");
        t4.DrawLatex(0.06, 0.38, "carries little information about the");
        t4.DrawLatex(0.06, 0.32, "true p_{z}, which would cap how well");
        t4.DrawLatex(0.06, 0.26, "any leptonic #theta*/#phi* can do.");
        printf("\nneutrino pz:  reco-fit minus truth-fit  mean %+.2f  RMS %.2f GeV\n",
               d1->GetMean(), d1->GetRMS());
        printf("              correlation(reco fit, true pz) = %.3f\n",
               m->GetCorrelationFactor());

        c4->Print((out + ")").Data());
    }

    printf("\nwrote %s\n\n", out.Data());
}
