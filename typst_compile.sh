#!/usr/bin/env bash
typst compile assets/kruusalu_cv.typ assets/kruusalu_cv.pdf
typst compile projects/catenary-in-wind/cable.typ projects/catenary-in-wind/cable.svg
echo "Typst compiled successfully."
