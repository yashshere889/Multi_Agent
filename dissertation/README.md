# Dissertation

No LaTeX toolchain on the authoring machine. Compile on Overleaf (upload this
folder and set `main.tex` as the root document), or locally with:

    latexmk -pdf main.tex

`IEEEtranN.bst` is not bundled — on Overleaf it resolves automatically. If it
does not, switch `\bibliographystyle{IEEEtranN}` in `main.tex` to `plainnat`
(Harvard-style author–year, also acceptable per the brief) and remove the
`numbers` option from the `natbib` package line.

Page budget is recorded at the top of `main.tex` and in each chapter file.
