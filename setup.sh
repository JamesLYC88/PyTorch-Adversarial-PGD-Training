#! /usr/bin/env bash

gdown --id 1QsXu3FpU5N4pXN4Vs5r2QR6Y-Y79hdpK -O wrn16.pth
gdown --id 1F0Jye2aOHAtSoknMV-ElPVC7UtqMTKAs -O benign.zip
gdown --id 1Y-3PPHZuOcATU-SSFCdBn1uyRLfiV9AD -O adv.zip

unzip benign.zip &> /dev/null
unzip adv.zip &> /dev/null

rm benign.zip adv.zip
