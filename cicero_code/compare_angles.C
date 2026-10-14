void compare_angles()
{
    TFile *ft = TFile::Open("rootFiles/ttbar_angles_truth_recon.root");
    TFile *fh = TFile::Open("rootFiles/hh_angles_truth_recon.root");
    if (!ft || !fh || ft->IsZombie() || fh->IsZombie()) {
        printf("could not open one of the input files\n"); return;
    }
    TTree *tt = (TTree*) ft->Get("ttbar");
    TTree *th = (TTree*) fh->Get("ttbar");

    gSystem->mkdir("plots/compare", kTRUE);
    gStyle->SetOptStat(0);

    const int N = 12;
    const char *var[N] = {
        "t_theta",      "t_phi",      "t_thetaprime", "t_phistar",
        "t~_theta",     "t~_phi",     "t~_thetastar", "t~_phistar",
        "rapidity",     "cosprodangle", "chi2",       "m_bb"
    };
    const char *name[N] = {
        "lep_costheta", "lep_phi",   "lep_costhetastar", "lep_phistar",
        "had_costheta", "had_phi",   "had_costhetastar", "had_phistar",
        "rapidity",     "cosprod",   "chi2",             "m_bb"
    };
    const char *xlab[N] = {
        "cos#theta (leptonic)", "#phi (leptonic)",
        "cos#theta* (leptonic)", "#phi* (leptonic)",
        "cos#theta (hadronic)", "#phi (hadronic)",
        "cos#theta* (hadronic)", "#phi* (hadronic)",
        "y(t#bar{t})", "cos#Theta_{prod}", "#chi^{2}", "m(bb) [GeV]"
    };
    const double lo[N] = {-1, -3.1416, -1, -3.1416, -1, -3.1416, -1, -3.1416, -5, -1,  0,   0};
    const double hi[N] = { 1,  3.1416,  1,  3.1416,  1,  3.1416,  1,  3.1416,  5,  1, 60, 400};

    TCanvas *grid = new TCanvas("grid", "ttbar vs HH", 1800, 1200);
    grid->Divide(4, 3);

    for (int i = 0; i < N; ++i) {
        TH1F *a = new TH1F(Form("a_%d", i), "", 50, lo[i], hi[i]);
        TH1F *b = new TH1F(Form("b_%d", i), "", 50, lo[i], hi[i]);

        tt->Draw(Form("%s>>a_%d", var[i], i), "", "goff");
        th->Draw(Form("%s>>b_%d", var[i], i), "", "goff");

        if (a->Integral() > 0) a->Scale(1.0/a->Integral());
        if (b->Integral() > 0) b->Scale(1.0/b->Integral());

        a->SetLineColor(kBlue);  a->SetLineWidth(2);
        b->SetLineColor(kRed);   b->SetLineWidth(2);
        a->SetTitle(xlab[i]);
        a->GetXaxis()->SetTitle(xlab[i]);
        a->GetYaxis()->SetTitle("fraction of events");
        a->SetMaximum(1.25 * TMath::Max(a->GetMaximum(), b->GetMaximum()));

        // separation metric: 0.5 * sum |p_tt - p_hh|, the total variation distance
        double tv = 0.0;
        for (int k = 1; k <= a->GetNbinsX(); ++k)
            tv += fabs(a->GetBinContent(k) - b->GetBinContent(k));
        tv *= 0.5;
        printf("%-16s  TV = %5.3f   mean: tt %8.4f  hh %8.4f\n",
               name[i], tv, a->GetMean(), b->GetMean());

        // one-off canvas per variable
        TCanvas c(Form("c_%d", i), xlab[i], 800, 600);
        a->Draw("HIST"); b->Draw("HIST SAME");
        TLegend leg(0.68, 0.78, 0.88, 0.88);
        leg.AddEntry(a, "t#bar{t}", "l");
        leg.AddEntry(b, "HH", "l");
        leg.SetBorderSize(0);
        leg.Draw();
        TLatex tl; tl.SetNDC(); tl.SetTextSize(0.035);
        tl.DrawLatex(0.15, 0.84, Form("TV = %.3f", tv));
        c.SaveAs(Form("plots/compare/%s.pdf", name[i]));

        // same thing into the grid
        grid->cd(i + 1);
        a->DrawCopy("HIST"); b->DrawCopy("HIST SAME");
        leg.DrawClone();
    }

    grid->SaveAs("plots/compare/all_angles.pdf");
    grid->SaveAs("plots/compare/all_angles.png");
    printf("\nwrote plots/compare/\n");
}
