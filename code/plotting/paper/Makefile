# Makefile for the NPAD paper: see what changed between two versions with latexdiff.
# It lives in the code repository (code/paper_figs/, run make there); the paper is the Overleaf clone at $(PAPER).
#
#   make                        $(BUILD)/main_diff.pdf  (tag v1 -> main.tex; also REV=v2, v3 or any commit)
#   make OLD=a.tex NEW=b.tex    compare any two files
#   make REV=HEAD               compare main.tex with a git commit, e.g. what is on Overleaf
#   make words                  quick word-level diff in the terminal (same options)
#   make clean                  delete $(BUILD)/
#
# All output goes to $(BUILD) = results/paper_figs/build/, so nothing new appears next to main.tex.
# latexdiff: the installed one is used if there is one; otherwise the stand-alone
# version (a single Perl script) is downloaded from CTAN into $(BUILD)/ on first use.

OLD   ?=
# the old version: main.tex at this git tag or commit (v1 = the first Overleaf version)
PAPER ?= ../../../paper
NEW   ?= $(PAPER)/main.tex
REV   ?= v1
BUILD ?= ../../results/paper_figs/build
NAME  ?= main_diff

# the figures are found in $(PAPER) through TEXINPUTS
LATEX ?= TEXINPUTS=$(abspath $(PAPER))//: pdflatex -interaction=nonstopmode -halt-on-error

LATEXDIFF_URL ?= https://mirrors.ctan.org/support/latexdiff/latexdiff-so
ifneq ($(shell command -v latexdiff 2>/dev/null),)
LATEXDIFF ?= latexdiff
else
LATEXDIFF ?= perl $(BUILD)/latexdiff-so
LATEXDIFF_DEP := $(BUILD)/latexdiff-so
endif

# Mark up the text but treat the reference list as one block. Otherwise latexdiff
# interleaves the old and new \bibitem entries and the diff does not compile.
# --math-markup=whole: a changed equation is marked as one block (markup inside align breaks LaTeX)
DIFFOPTS ?= --config="PICTUREENV=(?:picture|DIFnomarkup|thebibliography)[\w\d*@]*" --math-markup=whole --exclude-textcmd="section,subsection"

# With REV set, the old version is $(NEW) as it was in that git commit.
ifeq ($(strip $(REV)),)
OLDFILE := $(OLD)
else
OLDFILE := $(BUILD)/old.tex
endif

.PHONY: diff words prep clean
.DEFAULT_GOAL := diff

diff: prep $(LATEXDIFF_DEP)
	$(LATEXDIFF) $(DIFFOPTS) $(OLDFILE) $(NEW) > $(BUILD)/$(NAME).tex
	@# start from a fresh .aux: one left by an interrupted run breaks the next pass
	@rm -f $(BUILD)/$(NAME).aux
	@for pass in 1 2; do \
	  echo "pdflatex pass $$pass of 2"; \
	  $(LATEX) -output-directory=$(BUILD) $(BUILD)/$(NAME).tex > /dev/null \
	    || { grep -m1 -A3 '^!' $(BUILD)/$(NAME).log; exit 1; }; \
	done
	@echo "Wrote $(BUILD)/$(NAME).pdf"

words: prep
	@git diff --no-index --color-words $(OLDFILE) $(NEW); [ $$? -le 1 ]

prep:
	@mkdir -p $(BUILD)
	@for f in $(if $(strip $(REV)),,$(OLD)) $(NEW); do \
	  [ -f "$$f" ] || { echo "$$f not found. Choose the files with: make OLD=old.tex NEW=new.tex"; exit 1; }; \
	done
ifneq ($(strip $(REV)),)
	git -C $(PAPER) show $(REV):main.tex > $(OLDFILE)
endif

$(BUILD)/latexdiff-so:
	@mkdir -p $(BUILD)
	@echo "latexdiff is not installed; downloading it from CTAN"
	curl -fsSL -o $@.part $(LATEXDIFF_URL) && mv $@.part $@

clean:
	rm -rf $(BUILD)
