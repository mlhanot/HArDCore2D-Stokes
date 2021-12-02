ADD_EXECUTABLE(PotentialNabla PotentialNabla.cpp) 
TARGET_LINK_LIBRARIES(PotentialNabla mesh quadrature stokescore plot)

ADD_EXECUTABLE(PotentialCurl PotentialCurl.cpp) 
TARGET_LINK_LIBRARIES(PotentialCurl mesh quadrature stokescore plot)

ADD_EXECUTABLE(EquivNorm EquivNorm.cpp) 
TARGET_LINK_LIBRARIES(EquivNorm mesh quadrature stokescore plot)

ADD_EXECUTABLE(Consistency Consistency.cpp) 
TARGET_LINK_LIBRARIES(Consistency mesh quadrature stokescore plot)

ADD_EXECUTABLE(AdjConsistency AdjConsistency.cpp) 
TARGET_LINK_LIBRARIES(AdjConsistency mesh quadrature stokescore plot)

ADD_EXECUTABLE(LocalComplex LocalComplex.cpp) 
TARGET_LINK_LIBRARIES(LocalComplex mesh quadrature stokescore plot)

ADD_EXECUTABLE(LocalComplexKernel LocalComplexKernel.cpp) 
TARGET_LINK_LIBRARIES(LocalComplexKernel mesh quadrature stokescore plot)
