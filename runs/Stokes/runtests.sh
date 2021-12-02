#!/bin/bash
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")") # get script location
scriptDir=$scriptDir/..
cd "$scriptDir"
echo
echo "Test Potential Curl"
Stokes/PotentialCurl
echo
echo "Test Potential Nabla"
Stokes/PotentialNabla
echo
echo "Test LocalComplex"
Stokes/LocalComplex
echo
echo "Test LocalComplexKernel"
Stokes/LocalComplexKernel
echo
echo "Test EquivNorm"
Stokes/EquivNorm
echo
echo "Test Consistency"
Stokes/Consistency
echo
echo "Skipping test AdjConsistenc"
# Stokes/AdjConsistency # Slow and memory hungry
