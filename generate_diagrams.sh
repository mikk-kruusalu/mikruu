#!/bin/bash

cd tikz/catenary-in-wind
podmanlatex latexmk -auxdir=build -pdf cable.tex
pdftoppm cable.pdf cable
magick cable-1.ppm cable.png
cp cable.png ../../static/images/
cd ../../
