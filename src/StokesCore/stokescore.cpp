#include <cassert>

#include <parallel_for.hpp>

#include "stokescore.hpp"

using namespace HArDCore2D;

//------------------------------------------------------------------------------

StokesCore::StokesCore(const Mesh & mesh, size_t K, bool use_threads, std::ostream & output)
  : m_mesh(mesh),
    m_K(K),
    m_output(output),
    m_cell_bases(mesh.n_cells()),
    m_edge_bases(mesh.n_edges())
{
  m_output << "[StokesCore] Initializing" << std::endl;
  
  // Construct element bases
  std::function<void(size_t, size_t)> construct_all_cell_bases
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iT = start; iT < end; iT++) {
	        this->m_cell_bases[iT].reset( new CellBases(this->_construct_cell_bases(iT)) );
	      } // for iT
      };

  m_output << "[StokesCore] Constructing element bases" << std::endl;
  parallel_for(mesh.n_cells(), construct_all_cell_bases, use_threads);
  
  // Construct edge bases
  std::function<void(size_t, size_t)> construct_all_edge_bases   
    = [this](size_t start, size_t end)->void
      {
	      for (size_t iE = start; iE < end; iE++) {
	        this->m_edge_bases[iE].reset( new EdgeBases(_construct_edge_bases(iE)) );
	      } // for iF
      };
  
  m_output << "[StokesCore] Constructing edge bases" << std::endl;
  parallel_for(mesh.n_edges(), construct_all_edge_bases, use_threads);
}

//------------------------------------------------------------------------------

StokesCore::CellBases StokesCore::_construct_cell_bases(size_t iT)
{
  const Cell & T = *m_mesh.cell(iT);

  CellBases bases_T;
  
  //------------------------------------------------------------------------------
  // Basis for Pk+1(T)
  //------------------------------------------------------------------------------
  
  MonomialScalarBasisCell basis_Pkpo_T(T, m_K + 1);
  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (m_K + 1));
  boost::multi_array<double, 2> on_basis_Pkpo_T_quad = evaluate_quad<Function>::compute(basis_Pkpo_T, quad_2kpo_T);
  // Orthonormalize and store
  bases_T.Polykpo.reset( new PolyBasisCellType(l2_orthonormalize(basis_Pkpo_T, quad_2kpo_T, on_basis_Pkpo_T_quad)) );   
  // Check that we got the dimension right
  assert( bases_T.Polykpo->dimension() == PolynomialSpaceDimension<Cell>::Poly(m_K + 1) );

  //------------------------------------------------------------------------------
  // Basis for Pk(T), Pk-1(T), Pk+1(T)^2 and Pk(T)^2
  //------------------------------------------------------------------------------

  // Given that the basis for Pk+1(T) is hierarchical, bases for Pk(T) and
  // Pk-1(T) can be obtained by restricting the former
  bases_T.Polyk.reset( new RestrictedBasis<PolyBasisCellType>(*bases_T.Polykpo, PolynomialSpaceDimension<Cell>::Poly(m_K)) );  
  bases_T.Polyk2.reset( new Poly2BasisCellType(*bases_T.Polyk) );
  bases_T.Polyk2po.reset( new Poly2BasisCellType(RestrictedBasis<PolyBasisCellType>(*bases_T.Polykpo, PolynomialSpaceDimension<Cell>::Poly(m_K+1))) );
  bases_T.Polykmo.reset( new RestrictedBasis<PolyBasisCellType>(*bases_T.Polykpo, PolynomialSpaceDimension<Cell>::Poly(m_K - 1)) );
  // Check dimension Pk(T)^2
  assert( bases_T.Polyk2->dimension() == 2 * PolynomialSpaceDimension<Cell>::Poly(m_K) );
  assert( bases_T.Polyk2po->dimension() == 2 * PolynomialSpaceDimension<Cell>::Poly(m_K + 1) );
  
  //------------------------------------------------------------------------------
  // Basis for Gck+2(T)
  //------------------------------------------------------------------------------
  QuadratureRule quad_2kp2_T = generate_quadrature_rule(T, 2 * (m_K + 2));
  // Non-orthonormalised basis of Gck+2(T)
  GolyComplBasiswCurlCell basis_Gckp2(T, m_K + 2);
  auto basis_Gckp2_T_quad = evaluate_quad<Function>::compute(basis_Gckp2, quad_2kp2_T);
  // Orthonormalise, store and check dimension
  bases_T.GolyComplkp2.reset(new GolyComplBasisCellType(l2_orthonormalize(basis_Gckp2, quad_2kp2_T, basis_Gckp2_T_quad)));
  assert( bases_T.GolyComplkp2->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K + 2));

  //------------------------------------------------------------------------------
  // Basis for Gck(T)
  //------------------------------------------------------------------------------
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * m_K);
  // Non-orthonormalised basis of Gck(T)
  GolyComplBasiswCurlCell basis_Gck(T, m_K );
  // Orthonormalise, store and check dimension
  if (m_K > 0) {
  auto basis_Gck_T_quad = evaluate_quad<Function>::compute(basis_Gck, quad_2k_T);
    bases_T.GolyComplk.reset(new GolyComplBasisCellType(l2_orthonormalize(basis_Gck, quad_2k_T, basis_Gck_T_quad)));
  } else {
    Eigen::Matrix<double,0,0> ZM;
    bases_T.GolyComplk.reset(new GolyComplBasisCellType(basis_Gck,ZM));
  }
  assert( bases_T.GolyComplk->dimension() == PolynomialSpaceDimension<Cell>::GolyCompl(m_K) );

  //------------------------------------------------------------------------------
  // Basis for Gkmo(T)
  //------------------------------------------------------------------------------
  // Non-orthonormalised basis of Rk-1(T). 
  MonomialScalarBasisCell basis_Pk_T(T, m_K);
  ShiftedBasis<MonomialScalarBasisCell> basis_Pk0_T(basis_Pk_T,1);
  GradientBasis<ShiftedBasis<MonomialScalarBasisCell>> basis_Gkmo_T(basis_Pk0_T);
  // Orthonormalise, store and check dimension
  if (m_K > 0) {
    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(basis_Gkmo_T, quad_2k_T);
    bases_T.Golykmo.reset( new GolyBasisCellType(l2_orthonormalize(basis_Gkmo_T, quad_2k_T, basis_Gkmo_T_quad)) );
  } else {
    Eigen::Matrix<double,0,0> ZM;
    bases_T.Golykmo.reset( new GolyBasisCellType(basis_Gkmo_T,ZM));
  }
  assert( bases_T.Golykmo->dimension() == PolynomialSpaceDimension<Cell>::Goly(m_K - 1) );

  //------------------------------------------------------------------------------
  // Basis for Rck+2(T)^2
  //------------------------------------------------------------------------------
  // Non-orthonormalised
  RolyComplBasisCell basis_Rckp2_T(T,m_K + 2);
  auto basis_Rckp2_T_quad = evaluate_quad<Function>::compute(basis_Rckp2_T, quad_2kp2_T);
  // Orthonormalise, tensorize, store and check dimension
  bases_T.RolyComplk2p2.reset( new Roly2ComplBasisCellType(l2_orthonormalize(basis_Rckp2_T, quad_2kp2_T, basis_Rckp2_T_quad)));
  assert( bases_T.RolyComplk2p2->dimension() == 2*PolynomialSpaceDimension<Cell>::RolyCompl(m_K + 2) );

  //------------------------------------------------------------------------------
  // Basis for Rbck+1(T)
  //------------------------------------------------------------------------------
  // Non-orthonormalised basis of Rbck+1(T)
  RolybComplBasisCell basis_Rbckpo(T, m_K + 1);
  // Orthonormalise, store and check dimension
  if (m_K > 0) {
    auto basis_Rbckpo_T_quad = evaluate_quad<Function>::compute(basis_Rbckpo, quad_2kpo_T);
    bases_T.RolybComplkpo.reset(new RolybComplBasisCellType(l2_orthonormalize(basis_Rbckpo, quad_2kpo_T, basis_Rbckpo_T_quad)));
  } else {
    Eigen::Matrix<double,0,0> ZM;
    bases_T.RolybComplkpo.reset(new RolybComplBasisCellType(basis_Rbckpo, ZM));
  }
  assert( bases_T.RolybComplkpo->dimension() == PolynomialSpaceDimensionRolybCompl(m_K + 1));

  //------------------------------------------------------------------------------
  // Basis for Rbk(T)
  //------------------------------------------------------------------------------
  // Non-orthonormalised basis of Rck(T)
  RolybBasisCell basis_Rbk(T, m_K );
  // Orthonormalise, store and check dimension
  if (m_K > 0) {
    auto basis_Rbk_T_quad = evaluate_quad<Function>::compute(basis_Rbk, quad_2k_T);
    bases_T.Rolybk.reset(new RolybBasisCellType(l2_orthonormalize(basis_Rbk, quad_2k_T, basis_Rbk_T_quad)));
  } else {
    Eigen::Matrix<double,0,0> ZM;
    bases_T.Rolybk.reset(new RolybBasisCellType(basis_Rbk,ZM));
  }
  assert( bases_T.Rolybk->dimension() == PolynomialSpaceDimensionRRolyb(m_K));

  //------------------------------------------------------------------------------
  // Basis for RTbkpo(T)
  //------------------------------------------------------------------------------
  // Non-orthonormalised basis of Rk(T)^2
  ShiftedBasis<MonomialScalarBasisCell> basis_Pkpo0_T(basis_Pkpo_T,1);
  CurlBasiswDiv<ShiftedBasis<MonomialScalarBasisCell>> basis_Rk_T(basis_Pkpo0_T);
  // Orthonormalise, tensorize, store and check dimension
  auto basis_Rk_T_quad = evaluate_quad<Function>::compute(basis_Rk_T, quad_2k_T);
  Roly2BasisCellType basis_Rk2_T(l2_orthonormalize(basis_Rk_T, quad_2k_T, basis_Rk_T_quad));
  assert( basis_Rk2_T.dimension() == 2*PolynomialSpaceDimension<Cell>::Roly(m_K));
  // Take the direct sum
  bases_T.RTbkpo.reset(new RTbBasisCellType(*bases_T.RolybComplkpo, SumFamily<RolybBasisCellType,Roly2BasisCellType>(*bases_T.Rolybk, basis_Rk2_T)));
  assert( bases_T.RTbkpo->dimension() == PolynomialSpaceDimensionRTb(m_K + 1));
  
  return bases_T;
}


//------------------------------------------------------------------------------

StokesCore::EdgeBases StokesCore::_construct_edge_bases(size_t iE)
{
  const Edge & E = *m_mesh.edge(iE);

  EdgeBases bases_E;

  // Basis for Pk+2(E)
  MonomialScalarBasisEdge basis_Pkp2_E(E, m_K + 2);
  QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * (m_K + 2));
  auto basis_Pkp2_E_quad = evaluate_quad<Function>::compute(basis_Pkp2_E, quad_2kp2_E);
  bases_E.Polykp2.reset( new PolyEdgeBasisType(l2_orthonormalize(basis_Pkp2_E, quad_2kp2_E, basis_Pkp2_E_quad)) );

  // Basis for Pk+1(E)
  bases_E.Polykpo.reset( new RestrictedBasis<PolyEdgeBasisType>(*bases_E.Polykp2, PolynomialSpaceDimension<Edge>::Poly(m_K + 1)));

  // Basis for Pk(E)
  bases_E.Polyk.reset( new RestrictedBasis<PolyEdgeBasisType>(*bases_E.Polykp2, PolynomialSpaceDimension<Edge>::Poly(m_K)) );
  
  // Basis for Pk-1(E)
  // Given that the basis for Pk+1(E) is hierarchical, a basis for Pk-1(E)
  // can be obtained by restricting the former
    bases_E.Polykmo.reset( new RestrictedBasis<PolyEdgeBasisType>(*bases_E.Polykp2, PolynomialSpaceDimension<Edge>::Poly(m_K - 1)) );

  // Basis for Pk+2(E)^2
  bases_E.Polyk2p2.reset( new TensorizedVectorFamily<PolyEdgeBasisType, 2>(*bases_E.Polykp2));
  assert( bases_E.Polyk2p2->dimension() == 2*PolynomialSpaceDimension<Edge>::Poly(m_K + 2));

  // Basis for Pk+1(E)^2
  bases_E.Polyk2po.reset( new TensorizedVectorFamily<RestrictedBasis<PolyEdgeBasisType>, 2>(*bases_E.Polykpo));
  assert( bases_E.Polyk2po->dimension() == 2*PolynomialSpaceDimension<Edge>::Poly(m_K + 1));

  return bases_E;
}
