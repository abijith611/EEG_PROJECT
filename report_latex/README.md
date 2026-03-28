# LaTeX Report

This folder contains the LaTeX source for the reproduction study report.

## Files

- `main.tex` - Main LaTeX document
- `figures/` - All figures used in the report
  - `Figure1_author.png` - Original Figure 1 from the paper
  - `Figure1_ours.png` - Our reproduced Figure 1
  - `Figure2_author.png` - Original Figure 2 from the paper
  - `Figure2_lda_ours.png` - Our reproduced Figure 2 (LDA)
  - `Figure3_author.png` - Original Figure 3 from the paper
  - `Figure3_lda_ours.png` - Our reproduced Figure 3 (LDA)

## Compilation

### Using pdflatex (recommended)

```bash
# Compile twice for proper cross-references
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (automatic)

```bash
latexmk -pdf main.tex
```

### Using the Makefile

```bash
make        # Compile PDF
make clean  # Remove auxiliary files
make cleanall  # Remove all generated files including PDF
```

## Requirements

The following LaTeX packages are required:
- geometry
- graphicx
- booktabs
- longtable
- listings
- xcolor
- hyperref
- natbib
- fancyhdr
- caption
- csquotes
- microtype
- amsmath, amssymb
- float
- subcaption
- array, multirow, tabularx
- parskip
- setspace

Most of these are included in standard LaTeX distributions (TeX Live, MiKTeX).

## Output

After compilation, you will get:
- `main.pdf` - The complete report in PDF format
