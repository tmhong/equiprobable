// ============================================================================
//  overlay_truth_reco.C      (Option A: truth and reco in SEPARATE files)
//
//  Produces ONE pdf containing all ten distributions in a 4x3 grid. Each cell
//  is an overlay of truth (black) vs reconstructed (blue), normalised to unit
//  area, with a reco/truth ratio panel underneath. The twelfth cell holds the
//  legend and event counts.
//
//  Both files must contain a tree called "ttbar" with the same ten branch
//  names, which is the case since truth_calculate_ttbar_angles.C and
//  recon_calculate_ttbar_angles.C write identical branch names.
//
//  INTERPRETATION NOTE
//  The two files do NOT contain the same events. The truth macro selects on
//  exactly one each of b, bbar, lepton, neutrino, quark, antiquark at
//  Status==23 plus >=4 GenJets; the reco macro selects one lepton, two b-tags,
//  >=2 light jets and MET>30. Any difference below is therefore the COMBINED
//  effect of (a) the two selections keeping different events and (b)
//  reconstruction smearing, and cannot separate the two. For that you need
//  per-event truth and reco in one tree (Option B).
//
//  The TV number printed in each cell is the total variation between truth and
//  reco for that variable, i.e. a measure of DISTORTION. This is a different
//  quantity from the ttbar-vs-HH TV values, which measured SEPARATION. Here
//  large is bad.
//
//  Run:
//    cd /home/cicero/Pitt/Hong/sim-tools/analysis
//    root -l 'overlay_truth_reco.C("rootFiles/ttbar_angles_truth.root","rootFiles/ttbar_angles_ttbar.root","ttbar")'
//    root -l 'overlay_truth_reco.C("rootFiles/hh_angles_truth.root","rootFiles/hh_angles_ttbar.root","hh")'
// ============================================================================

void overlay_truth_reco(const char *truthFile = "rootFiles/alex_ttbar_angles_truth.root",
                        const char *recoFile  = "rootFiles/ttbar_angles_truthdown.root",
                        const char *tag       = "ttbar")
{
    TFile *ftr = TFile::Open(truthFile);
    TFile *frc = TFile::Open(recoFile);
    if (!ftr || ftr->IsZombie()) { printf("cannot open %s\n", truthFile); return; }
    if (!frc || frc->IsZombie()) { printf("cannot open %s\n", recoFile);  return; }

    TTree *ttr = (TTree*) ftr->Get("ttbar");
    TTree *trc = (TTree*) frc->Get("ttbar");
    if (!ttr || !trc) { printf("missing tree 'ttbar' in one of the files\n"); return; }

    const Long64_t nTruth = ttr->GetEntries();
    const Long64_t nReco  = trc->GetEntries();
    printf("\n%s:  truth %lld events   reco %lld events\n\n", tag, nTruth, nReco);
    if (nTruth == 0 || nReco == 0) { printf("one of the trees is empty\n"); return; }

    gSystem->mkdir("plots", kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);          // the x-axis title names each variable

    const int N = 10;
    const char *var[N] = {
        "t_theta",  "t_phi",  "t_thetaprime", "t_phistar",
        "t~_theta", "t~_phi", "t~_thetastar", "t~_phistar",
        "rapidity", "cosprodangle" };
    const char *shortName[N] = {
        "lep_costheta", "lep_phi", "lep_costhetastar", "lep_phistar",
        "had_costheta", "had_phi", "had_costhetastar", "had_phistar",
        "rapidity",     "cosprod" };
    const char *xlab[N] = {
        "cos#theta (lep)",  "#phi (lep)",
        "cos#theta* (lep)", "#phi* (lep)",
        "cos#theta (had)",  "#phi (had)",
        "cos#theta* (had)", "#phi* (had)",
        "y(t#bar{t})",      "cos#Theta_{prod}" };
    const double lo[N] = {-1, -3.1416, -1, -3.1416, -1, -3.1416, -1, -3.1416, -5, -1};
    const double hi[N] = { 1,  3.1416,  1,  3.1416,  1,  3.1416,  1,  3.1416,  5,  1};

    // 4 columns x 3 rows: ten variables plus a legend/info cell.
    const int NCOL = 4, NROW = 3;
    TCanvas *c = new TCanvas("cgrid", Form("truth vs reco (%s)", tag), 1800, 1500);

    // Everything is heap-allocated so it survives until SaveAs at the end.
    TH1F *ht[N], *hr[N];
    double tv[N];

    printf("%-18s  %9s %9s   %9s %9s   %6s\n",
           "variable", "mean_tru", "mean_rec", "rms_tru", "rms_rec", "TV");

    for (int i = 0; i < N; ++i) {
        ht[i] = new TH1F(Form("ht_%d", i), "", 50, lo[i], hi[i]);
        hr[i] = new TH1F(Form("hr_%d", i), "", 50, lo[i], hi[i]);

        // "goff" fills the histogram without opening a graphics window.
        ttr->Draw(Form("%s>>ht_%d", var[i], i), "", "goff");
        trc->Draw(Form("%s>>hr_%d", var[i], i), "", "goff");

        const double mt = ht[i]->GetMean(), mr = hr[i]->GetMean();
        const double st = ht[i]->GetRMS(),  sr = hr[i]->GetRMS();

        // unit area, so the different event counts do not matter
        if (ht[i]->Integral() > 0) ht[i]->Scale(1.0 / ht[i]->Integral());
        if (hr[i]->Integral() > 0) hr[i]->Scale(1.0 / hr[i]->Integral());

        tv[i] = 0.0;
        for (int k = 1; k <= ht[i]->GetNbinsX(); ++k)
            tv[i] += fabs(ht[i]->GetBinContent(k) - hr[i]->GetBinContent(k));
        tv[i] *= 0.5;

        printf("%-18s  %9.4f %9.4f   %9.4f %9.4f   %6.3f\n",
               shortName[i], mt, mr, st, sr, tv[i]);

        // ---- geometry of this cell, built by hand so each cell can hold a
        //      stacked pair of pads (Divide() would give only one pad each) ----
        const int col = i % NCOL;
        const int row = i / NCOL;
        const double x1 =  col      / (double) NCOL;
        const double x2 = (col + 1) / (double) NCOL;
        const double yTop = 1.0 -  row      / (double) NROW;
        const double yBot = 1.0 - (row + 1) / (double) NROW;
        const double ySplit = yBot + 0.32 * (yTop - yBot);

        c->cd(0);                                  // back to the mother canvas
        TPad *pu = new TPad(Form("pu_%d", i), "", x1, ySplit, x2, yTop);
        TPad *pd = new TPad(Form("pd_%d", i), "", x1, yBot,   x2, ySplit);
        pu->SetTopMargin(0.04);  pu->SetBottomMargin(0.02);
        pu->SetLeftMargin(0.17); pu->SetRightMargin(0.04);
        pd->SetTopMargin(0.03);  pd->SetBottomMargin(0.38);
        pd->SetLeftMargin(0.17); pd->SetRightMargin(0.04);
        pu->Draw(); pd->Draw();

        // ---- upper pad: the overlay ----
        pu->cd();
        ht[i]->SetLineColor(kBlack); ht[i]->SetLineWidth(2);
        hr[i]->SetLineColor(kBlue);  hr[i]->SetLineWidth(2);
        ht[i]->SetMinimum(0.0);
        ht[i]->SetMaximum(1.35 * TMath::Max(ht[i]->GetMaximum(), hr[i]->GetMaximum()));
        ht[i]->GetXaxis()->SetLabelSize(0);        // labels go on the lower pad
        ht[i]->GetYaxis()->SetTitle("fraction");
        ht[i]->GetYaxis()->SetTitleSize(0.075);
        ht[i]->GetYaxis()->SetTitleOffset(1.05);
        ht[i]->GetYaxis()->SetLabelSize(0.060);
        ht[i]->GetYaxis()->SetNdivisions(505);
        ht[i]->Draw("HIST");
        hr[i]->Draw("HIST SAME");

        TLatex *tl = new TLatex();
        tl->SetNDC(); tl->SetTextSize(0.075);
        tl->DrawLatex(0.22, 0.86, Form("TV = %.3f", tv[i]));

        // ---- lower pad: reco / truth ----
        pd->cd();
        TH1F *ratio = (TH1F*) hr[i]->Clone(Form("ratio_%d", i));
        ratio->Divide(ht[i]);            // empty truth bins give 0, not inf
        ratio->SetLineColor(kBlue);
        ratio->SetLineWidth(2);
        ratio->SetMinimum(0.0);
        ratio->SetMaximum(2.0);
        ratio->GetYaxis()->SetTitle("reco/truth");
        ratio->GetYaxis()->SetTitleSize(0.135);
        ratio->GetYaxis()->SetTitleOffset(0.58);
        ratio->GetYaxis()->SetLabelSize(0.115);
        ratio->GetYaxis()->SetNdivisions(503);
        ratio->GetXaxis()->SetTitle(xlab[i]);
        ratio->GetXaxis()->SetTitleSize(0.155);
        ratio->GetXaxis()->SetTitleOffset(1.00);
        ratio->GetXaxis()->SetLabelSize(0.125);
        ratio->GetXaxis()->SetNdivisions(505);
        ratio->Draw("HIST");

        TLine *one = new TLine(lo[i], 1.0, hi[i], 1.0);
        one->SetLineStyle(2);
        one->SetLineColor(kGray + 2);
        one->Draw();
    }

    // ---- cell 11: legend, event counts, and the interpretation warning ----
    {
        const int col = N % NCOL, row = N / NCOL;
        const double x1 =  col      / (double) NCOL;
        const double x2 = (col + 1) / (double) NCOL;
        const double yTop = 1.0 -  row      / (double) NROW;
        const double yBot = 1.0 - (row + 1) / (double) NROW;

        c->cd(0);
        TPad *pl = new TPad("pl", "", x1, yBot, x2, yTop);
        pl->Draw(); pl->cd();

        TLegend *leg = new TLegend(0.10, 0.62, 0.92, 0.88);
        leg->AddEntry(ht[0], "truth level",   "l");
        leg->AddEntry(hr[0], "reconstructed", "l");
        leg->SetTextSize(0.075);
        leg->SetBorderSize(0);
        leg->Draw();

        TLatex *tx = new TLatex();
        tx->SetNDC(); tx->SetTextSize(0.062);
        tx->DrawLatex(0.10, 0.52, Form("sample: %s", tag));
        tx->DrawLatex(0.10, 0.44, Form("truth entries: %lld", nTruth));
        tx->DrawLatex(0.10, 0.36, Form("reco  entries: %lld", nReco));
        tx->SetTextSize(0.048);
        tx->DrawLatex(0.10, 0.24, "TV = total variation,");
        tx->DrawLatex(0.10, 0.18, "truth vs reco (distortion).");
        tx->DrawLatex(0.10, 0.10, "Different event sets:");
        tx->DrawLatex(0.10, 0.04, "selection + smearing combined.");
    }

    // One pdf with everything. The png is for convenience -- raster survives
    // copy/paste and upload better than vector pdf.
    c->SaveAs(Form("plots/truth_vs_reco_%s.pdf", tag));
    c->SaveAs(Form("plots/truth_vs_reco_%s.png", tag));
    printf("\nwrote plots/truth_vs_reco_%s.pdf (and .png)\n\n", tag);
}
