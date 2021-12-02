// Core data structures and methods required to implement the discrete Stokes sequence in 2D
//
// Provides:
//  - Full and partial polynomial spaces on the element, faces, and edges
//  - Generic routines to create quadrature nodes over cells and faces of the mesh
//
// Author: Marien Lorenzo Hanot (marien-lorenzo.hanot@umontpellier.fr), 
// based on code written by Daniele Di Pietro (daniele.di-pietro@umontpellier.fr), Jerome Droniou (jerome.droniou@monash.edu)
//

/*
 *
 *      This library was developed around HHO methods, although some parts of it have a more
 * general purpose. If you use this code or part of it in a scientific publication, 
 * please mention the following book as a reference for the underlying principles
 * of HHO schemes:
 *
 * The Hybrid High-Order Method for Polytopal Meshes: Design, Analysis, and Applications. 
 *  D. A. Di Pietro and J. Droniou. Modeling, Simulation and Applications, vol. 19. 
 *  Springer International Publishing, 2020, xxxi + 525p. doi: 10.1007/978-3-030-37203-3. 
 *  url: https://hal.archives-ouvertes.fr/hal-02151813.
 *
 */

/*
 * The Stokes sequence has been designed in
 *
 *
 * If you use this code in a scientific publication, please mention the above article.
 *
 */
 

#ifndef STOKESCORE_HPP
#define STOKESCORE_HPP

#include <memory>
#include <iostream>

#include <basis.hpp>
#include <polynomialspacedimension.hpp>

#include "tensorbasis.hpp"

/*!	
 * @defgroup DDRCore 
 * @brief Classes providing tools to the Discrete Stokes sequence
 */


namespace HArDCore2D
{

  /*!
   *	\addtogroup StokesCore
   * @{
   */


  //------------------------------------------------------------------------------

  /// Construct all polynomial spaces for the Stokes sequence
  class StokesCore
  {
  public:
    // Types for element bases
    typedef Family<MonomialScalarBasisCell> PolyBasisCellType;
    typedef TensorizedVectorFamilywrot<RestrictedBasis<PolyBasisCellType>, 2> Poly2BasisCellType;
    typedef Family<GradientBasis<ShiftedBasis<MonomialScalarBasisCell> > > GolyBasisCellType;
    typedef Family<GolyComplBasiswCurlCell> GolyComplBasisCellType;
    typedef Family<CurlBasis<ShiftedBasis<MonomialScalarBasisCell> > > RolyBasisCellType;
    typedef Family<RolyComplBasisCell> RolyComplBasisCellType;
    typedef trFamily<RolybComplBasisCell> RolybComplBasisCellType;
    typedef MatrixVectorFamily<RolyComplBasisCellType, 2> Roly2ComplBasisCellType;
    typedef trFamily<RolybBasisCell> RolybBasisCellType;
    typedef MatrixVectorFamily<Family<CurlBasiswDiv<ShiftedBasis<MonomialScalarBasisCell> > >, 2> Roly2BasisCellType;
    typedef SumFamily<RolybComplBasisCellType,SumFamily<RolybBasisCellType,Roly2BasisCellType> > RTbBasisCellType;


    // Type for edge basis
    typedef Family<MonomialScalarBasisEdge> PolyEdgeBasisType;

    /// Structure to store element bases
    /** 'Poly': basis of polynomial space; 'Goly': gradient basis; 'Roly': curl basis.\n
        'k', 'kmo' (k-1) and 'kpo' (k+1) determines the degree.\n
        'Compl' for the complement of the corresponding 'Goly' or 'Roly' in the 'Poly' space */
    struct CellBases
    {
      /// Geometric support
      typedef Cell GeometricSupport;

      std::unique_ptr<PolyBasisCellType> Polykpo;
      std::unique_ptr<RestrictedBasis<PolyBasisCellType> > Polyk;
      std::unique_ptr<RestrictedBasis<PolyBasisCellType> > Polykmo;
      std::unique_ptr<Poly2BasisCellType> Polyk2po;
      std::unique_ptr<Poly2BasisCellType> Polyk2;
      std::unique_ptr<GolyComplBasisCellType> GolyComplkp2;
      std::unique_ptr<GolyComplBasisCellType> GolyComplk;
      std::unique_ptr<GolyBasisCellType> Golykmo;
      std::unique_ptr<Roly2ComplBasisCellType> RolyComplk2p2;
      std::unique_ptr<RolybComplBasisCellType> RolybComplkpo;
      std::unique_ptr<RolybBasisCellType> Rolybk;
      std::unique_ptr<RTbBasisCellType> RTbkpo;
    };

    /// Structure to store edge bases
    /** See CellBases for details */
    struct EdgeBases
    {
      /// Geometric support
      typedef Edge GeometricSupport;

      std::unique_ptr<PolyEdgeBasisType> Polykp2;
      std::unique_ptr<RestrictedBasis<PolyEdgeBasisType> > Polykpo;
      std::unique_ptr<RestrictedBasis<PolyEdgeBasisType> > Polyk;
      std::unique_ptr<RestrictedBasis<PolyEdgeBasisType> > Polykmo;
      std::unique_ptr<TensorizedVectorFamily<PolyEdgeBasisType, 2>> Polyk2p2;
      std::unique_ptr<TensorizedVectorFamily<RestrictedBasis<PolyEdgeBasisType>, 2>> Polyk2po;
    };    
    
    /// Constructor
    StokesCore(const Mesh & mesh, size_t K, bool use_threads = true, std::ostream & output = std::cout);
    
    /// Return a const reference to the mesh
    const Mesh & mesh() const
    {
      return m_mesh;
    }

    /// Return the polynomial degree
    const size_t & degree() const
    {
      return m_K;
    }
    
    /// Return cell bases for element iT
    inline const CellBases & cellBases(size_t iT) const
    {
      // Make sure that the basis has been created
      assert( m_cell_bases[iT] );
      return *m_cell_bases[iT].get();
    }

    /// Return edge bases for edge iE
    inline const EdgeBases & edgeBases(size_t iE) const
    {
      // Make sure that the basis has been created
      assert( m_edge_bases[iE] );
      return *m_edge_bases[iE].get();
    }

  private:
    /// Compute the bases on an element T
    CellBases _construct_cell_bases(size_t iT);

    /// Compute the bases on an edge E
    EdgeBases _construct_edge_bases(size_t iE);
    
    // Pointer to the mesh
    const Mesh & m_mesh;
    // Degree
    const size_t m_K;
    // Output stream
    std::ostream & m_output;
    
    // Cell bases
    std::vector<std::unique_ptr<CellBases> > m_cell_bases;
    // Edge bases
    std::vector<std::unique_ptr<EdgeBases> > m_edge_bases;
        
  };
  
} // end of namespace HArDCore2D

#endif // STOKESCORE_HPP
