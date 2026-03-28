# Makefile for LaTeX report compilation

MAIN = main
TEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Compile PDF
$(MAIN).pdf: $(MAIN).tex figures/*
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Quick compile (single pass)
quick:
	$(TEX) $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).lof $(MAIN).lot $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz

# Clean all generated files
cleanall: clean
	rm -f $(MAIN).pdf

# View PDF (macOS)
view: $(MAIN).pdf
	open $(MAIN).pdf

# Continuous compilation with latexmk
watch:
	latexmk -pdf -pvc $(MAIN).tex

.PHONY: all quick clean cleanall view watch
