# Library: stokescore

FILE(GLOB source_list stokescore.* xcurlstokes.* xnabla.* xvl.* tensorbasis.*)
ADD_LIBRARY(stokescore ${source_list})
TARGET_LINK_LIBRARIES(stokescore ddrcore basis mesh quadrature)
