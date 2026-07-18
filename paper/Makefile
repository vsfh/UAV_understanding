.PHONY: all clean package

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C

package:
	zip -r UTS-UAV_overleaf.zip . -x '*.aux' '*.bbl' '*.blg' '*.fdb_latexmk' '*.fls' '*.log' '*.out' '*.synctex.gz' '*.zip'
