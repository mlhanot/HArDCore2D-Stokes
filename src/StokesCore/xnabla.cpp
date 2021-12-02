#include <basis.hpp>
#include <parallel_for.hpp>

#include "xnabla.hpp"

using namespace HArDCore2D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XNabla::XNabla(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : DDRSpace(stokes_core.mesh(),
	     2,  
	     PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree())*2,
	     PolynomialSpaceDimension<Cell>::Goly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Cell>::GolyCompl(stokes_core.degree())
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_edge_operators(stokes_core.mesh().n_edges()),
    m_cell_operators(stokes_core.mesh().n_cells())
{
  m_output << "[XNabla] Initializing" << std::endl;
  if (use_threads) {
    m_output << "[XNabla] Parallel execution" << std::endl;
  } else {
    m_output << "[XNabla] Sequential execution" << std::endl;
  }
  
  // Construct edge grad and potentials
  std::function<void(size_t, size_t)> construct_all_edge_grads_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          m_edge_operators[iE].reset( new LocalOperators(_compute_edge_grad_potential(iE)) );
        } // for iE
      };

  m_output << "[XNabla] Constructing edge grads and potentials" << std::endl;
  parallel_for(mesh().n_edges(), construct_all_edge_grads_potentials, use_threads);

  // Construct cell grad and potentials
  std::function<void(size_t, size_t)> construct_all_cell_grads_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          m_cell_operators[iT].reset( new LocalOperators(_compute_cell_grad_potential(iT)) );
        } // for iT
      };

  m_output << "[XNabla] Constructing cell grads and potentials" << std::endl;
  parallel_for(mesh().n_cells(), construct_all_cell_grads_potentials, use_threads);
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XNabla::interpolate(const FunctionType & v, const int deg_quad) const
{
  Eigen::VectorXd vh = Eigen::VectorXd::Zero(dimension()); // dimension is the global dimension of dofs

  // Degree of quadrature rules
  size_t dqr = (deg_quad >= 0 ? deg_quad : 2 * degree() + 3);

  // Interpolate at vertices
  std::function<void(size_t, size_t)> interpolate_vertices
  = [this, &vh, v](size_t start, size_t end)->void
    {
      for (size_t iV = start; iV < end; iV++) {
        const Vertex & V = *mesh().vertex(iV);
        vh.segment(globalOffset(V),2) = v(mesh().vertex(iV)->coords());
      } // for iV
    };
  parallel_for(mesh().n_vertices(), interpolate_vertices, m_use_threads);

  
  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &vh, v,  &dqr](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          const Edge & E = *mesh().edge(iE);
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr);
          auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_dqr_E);

          std::function<double (const Eigen::Vector2d &)> vnE // function that take the tangential component
              = [this, v, &E](const Eigen::Vector2d & coord)->double {
                  return E.tangent().dot(v(coord));
              };
          vh.segment(globalOffset(E),PolynomialSpaceDimension<Edge>::Poly(degree())) 
            = l2_projection(vnE, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);

          std::function<double (const Eigen::Vector2d &)> vnEp // function that take the normal (to the face) component
              = [this, v, &E](const Eigen::Vector2d & coord)->double {
                  return E.normal().dot(v(coord));
              };
          vh.segment(globalOffset(E)+PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree())) 
            = l2_projection(vnEp, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);
        } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &vh, v, &dqr](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);
          QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr);
          // Gkmo
          auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_dqr_T);
          vh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::Goly(degree() - 1)) 
            = l2_projection(v, *cellBases(iT).Golykmo, quad_dqr_T, basis_Gkmo_T_quad);
          // Gck
          auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_dqr_T);
          vh.segment(globalOffset(T) + PolynomialSpaceDimension<Cell>::Goly(degree() - 1),PolynomialSpaceDimension<Cell>::GolyCompl(degree())) 
            = l2_projection(v, *cellBases(iT).GolyComplk, quad_dqr_T, basis_Gck_T_quad);
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);

  return vh;
}

//------------------------------------------------------------------------------
// Grad and potential reconstructions
//------------------------------------------------------------------------------

XNabla::LocalOperators XNabla::_compute_edge_grad_potential(size_t iE)
{
  const Edge & E = *mesh().edge(iE);
  
  //------------------------------------------------------------------------------
  // Grad
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree() + 1));
  auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykpo, quad_2kpo_E);
  auto MGE = compute_gram_matrix(basis_Pkpo_E_quad, basis_Pkpo_E_quad, quad_2kpo_E, "sym");

  Eigen::MatrixXd MGEv = Eigen::MatrixXd::Zero(2*edgeBases(iE).Polykpo->dimension(),2*edgeBases(iE).Polykpo->dimension());
  MGEv.topLeftCorner(edgeBases(iE).Polykpo->dimension(),edgeBases(iE).Polykpo->dimension()) = MGE;
  MGEv.bottomRightCorner(edgeBases(iE).Polykpo->dimension(),edgeBases(iE).Polykpo->dimension()) = MGE;


  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGE 
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polykpo->dimension(), 2 + edgeBases(iE).Polyk->dimension()); //
  for (size_t i = 0; i < edgeBases(iE).Polykpo->dimension(); i++) {
    BGE(i, 0) = -edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(0)->coords());
    BGE(i, 1) = edgeBases(iE).Polykpo->function(i, mesh().edge(iE)->vertex(1)->coords());
  } // for i

  QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2 * (degree() ));
  
  // if (degree() > 0) {    
    auto grad_Pkpo_tE_E_quad = scalar_product(
                                            evaluate_quad<Gradient>::compute(*edgeBases(iE).Polykpo, quad_2k_E),
                                            E.tangent()
                                            );
    auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_2k_E);
    BGE.rightCols(PolynomialSpaceDimension<Edge>::Poly(degree()))
      = -compute_gram_matrix(grad_Pkpo_tE_E_quad, basis_Pk_E_quad, quad_2k_E);
  //}

  Eigen::MatrixXd ProjnE = Eigen::MatrixXd::Zero(2*(2 + edgeBases(iE).Polyk->dimension()),2*(2 + edgeBases(iE).Polyk->dimension()));
  Eigen::MatrixXd BGEv = Eigen::MatrixXd::Zero(2*edgeBases(iE).Polykpo->dimension(),2*(2 + edgeBases(iE).Polyk->dimension()));

  ProjnE.block(0,0,1,2) = E.tangent().transpose();
  ProjnE.block(1,2,1,2) = E.tangent().transpose();
  ProjnE.block(2 + edgeBases(iE).Polyk->dimension(),0,1,2) = E.normal().transpose();
  ProjnE.block(2 + edgeBases(iE).Polyk->dimension() + 1,2,1,2) = E.normal().transpose();
  ProjnE.block(2,4, edgeBases(iE).Polyk->dimension(), edgeBases(iE).Polyk->dimension()) = Eigen::MatrixXd::Identity(edgeBases(iE).Polyk->dimension(),edgeBases(iE).Polyk->dimension());
  ProjnE.block(4+edgeBases(iE).Polyk->dimension(),4+edgeBases(iE).Polyk->dimension(), edgeBases(iE).Polyk->dimension(), edgeBases(iE).Polyk->dimension()) 
        = Eigen::MatrixXd::Identity(edgeBases(iE).Polyk->dimension(),edgeBases(iE).Polyk->dimension());
    
  BGEv.topLeftCorner(edgeBases(iE).Polykpo->dimension(),2 + edgeBases(iE).Polyk->dimension()) = BGE;
  BGEv.bottomRightCorner(edgeBases(iE).Polykpo->dimension(),2 + edgeBases(iE).Polyk->dimension()) = BGE;
  BGEv *= ProjnE;

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BPE
    = Eigen::MatrixXd::Zero(2*PolynomialSpaceDimension<Edge>::Poly(degree() + 2),4 + 2*edgeBases(iE).Polyk->dimension());

  // Enforce the gradient of the potential reconstruction
  BPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree()+1)) = BGEv.topRows(PolynomialSpaceDimension<Edge>::Poly(degree()+1));
  BPE.middleRows(PolynomialSpaceDimension<Edge>::Poly(degree()+1)+1,PolynomialSpaceDimension<Edge>::Poly(degree()+1)) = BGEv.bottomRows(PolynomialSpaceDimension<Edge>::Poly(degree()+1));


  // Enforce the average value of the potential reconstruction
    QuadratureRule quad_k_E = generate_quadrature_rule(E, degree());
    auto basis_Pka_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_k_E);

    // We set the average value of the potential equal to the average of the edge unknown
    for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree()); i++) {
      for (size_t iqn = 0; iqn < quad_k_E.size(); iqn++) {
        // 4 + i because the first four column contains the vertices value.
        BPE.row(PolynomialSpaceDimension<Edge>::Poly(degree()+1))(0, 4 + i) += quad_k_E[iqn].w * basis_Pka_E_quad[i][iqn];
        BPE.bottomRows(1)(0, 4 + PolynomialSpaceDimension<Edge>::Poly(degree()) + i) += quad_k_E[iqn].w * basis_Pka_E_quad[i][iqn];
      } // for iqn
    } // for i
  
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  Eigen::MatrixXd MPE
    = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree() + 2), PolynomialSpaceDimension<Edge>::Poly(degree() + 2));
  auto grad_Pkp2_tE_E_quad = scalar_product(
					    evaluate_quad<Gradient>::compute(*edgeBases(iE).Polykp2, quad_2kpo_E),
					    E.tangent()
					    );
  MPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree()+1))
    = compute_gram_matrix(basis_Pkpo_E_quad, grad_Pkp2_tE_E_quad, quad_2kpo_E);
  
  QuadratureRule quad_kp2_E = generate_quadrature_rule(E, degree() + 2);  
  auto basis_Pkp2_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykp2, quad_kp2_E);  
  for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree() + 2); i++) {
    for (size_t iqn = 0; iqn < quad_kp2_E.size(); iqn++) {
      MPE.bottomRows(1)(0, i) += quad_kp2_E[iqn].w * basis_Pkp2_E_quad[i][iqn];
    } // for iqn
  } // for i
  
  
  Eigen::MatrixXd MPEv
    = Eigen::MatrixXd::Zero(2*PolynomialSpaceDimension<Edge>::Poly(degree() + 2), 2*PolynomialSpaceDimension<Edge>::Poly(degree() + 2));

  MPEv.topLeftCorner(PolynomialSpaceDimension<Edge>::Poly(degree() + 2),PolynomialSpaceDimension<Edge>::Poly(degree() + 2)) = MPE;
  MPEv.bottomRightCorner(PolynomialSpaceDimension<Edge>::Poly(degree() + 2),PolynomialSpaceDimension<Edge>::Poly(degree() + 2)) = MPE;
   
  //------------------------------------------------------------------------------
  // Divergence
  Eigen::MatrixXd MDE; // empty, there no edge in L2


  //------------------------------------------------------------------------------
  // Convert from PolykpoxPolykpo -> Polyk2po

  
  QuadratureRule quad_kpopts_E = generate_quadrature_rule(E,2*(PolynomialSpaceDimension<Edge>::Poly(degree()+1)) - 1); // k + 1 points
  auto basis_Pkpo_E_quad_kpo = evaluate_quad<Function>::compute(*edgeBases(iE).Polykpo, quad_kpopts_E); // basis[i][iqn] = {Polykpo[i](iqn)}
  auto basis_Pk2po_E_quad_kpo = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk2po, quad_kpopts_E); // basis[i][iqn] = {Polyk2po[i](iqn)[x],Polyk2po[i](iqn)[y]}
  assert( basis_Pkpo_E_quad_kpo.shape()[0] == basis_Pkpo_E_quad_kpo.shape()[1]);
  Eigen::MatrixXd ValNEkpo = Eigen::MatrixXd::Zero(2*edgeBases(iE).Polykpo->dimension(),2*edgeBases(iE).Polykpo->dimension());
  Eigen::MatrixXd ValEXYkpo = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk2po->dimension(),edgeBases(iE).Polyk2po->dimension());
  // ValNE*vh (dofs nE + nEp) = ValNEXY *vh (dofs X,Y) such that the row iqn = v(iqn).nE for the first half, = v(iqn).nEp for the second
  for (size_t i = 0;i < PolynomialSpaceDimension<Edge>::Poly(degree() + 1); i++) {
    for (size_t j = 0;j < PolynomialSpaceDimension<Edge>::Poly(degree() + 1); j++) {
      double tmp = basis_Pkpo_E_quad_kpo[j][i];
      ValNEkpo(i,j) = tmp;
      ValNEkpo(i+edgeBases(iE).Polykpo->dimension(),j+edgeBases(iE).Polykpo->dimension()) = tmp;
    }
    for (size_t j = 0;j < 2*PolynomialSpaceDimension<Edge>::Poly(degree() + 1); j++) {
      ValEXYkpo(i,j) = E.tangent().dot(basis_Pk2po_E_quad_kpo[j][i]);
      ValEXYkpo(i+edgeBases(iE).Polykpo->dimension(),j) = E.normal().dot(basis_Pk2po_E_quad_kpo[j][i]);
    }
  }
  Eigen::MatrixXd NE2XYkpo = ValEXYkpo.partialPivLu().solve(ValNEkpo); // Convert dofs of Polykpo+Polykpo to dofs of Polyk2po
  
  QuadratureRule quad_kp2pts_E = generate_quadrature_rule(E,2*(PolynomialSpaceDimension<Edge>::Poly(degree()+2)) - 1); // k + 2 points
  auto basis_Pkp2_E_quad_kp2 = evaluate_quad<Function>::compute(*edgeBases(iE).Polykp2, quad_kp2pts_E); // basis[i][iqn] = {Polykpo[i](iqn)}
  auto basis_Pk2p2_E_quad_kp2 = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk2p2, quad_kp2pts_E); // basis[i][iqn] = {Polyk2po[i](iqn)[x],Polyk2po[i](iqn)[y]}
  assert( basis_Pkp2_E_quad_kp2.shape()[0] == basis_Pkp2_E_quad_kp2.shape()[1]);
  Eigen::MatrixXd ValNEkp2 = Eigen::MatrixXd::Zero(2*edgeBases(iE).Polykp2->dimension(),2*edgeBases(iE).Polykp2->dimension());
  Eigen::MatrixXd ValEXYkp2 = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk2p2->dimension(),edgeBases(iE).Polyk2p2->dimension());
  // ValNE*vh (dofs nE + nEp) = ValNEXY *vh (dofs X,Y) such that the row iqn = v(iqn).nE for the first half, = v(iqn).nEp for the second
  for (size_t i = 0;i < PolynomialSpaceDimension<Edge>::Poly(degree() + 2); i++) {
    for (size_t j = 0;j < PolynomialSpaceDimension<Edge>::Poly(degree() + 2); j++) {
      double tmp = basis_Pkp2_E_quad_kp2[j][i];
      ValNEkp2(i,j) = tmp;
      ValNEkp2(i+edgeBases(iE).Polykp2->dimension(),j+edgeBases(iE).Polykp2->dimension()) = tmp;
    }
    for (size_t j = 0;j < 2*PolynomialSpaceDimension<Edge>::Poly(degree() + 2); j++) {
      ValEXYkp2(i,j) = E.tangent().dot(basis_Pk2p2_E_quad_kp2[j][i]);
      ValEXYkp2(i+edgeBases(iE).Polykp2->dimension(),j) = E.normal().dot(basis_Pk2p2_E_quad_kp2[j][i]);
    }
  }
  Eigen::MatrixXd NE2XYkp2 = ValEXYkp2.partialPivLu().solve(ValNEkp2); // Convert dofs of Polykp2+Polykp2 to dofs of Polyk2p2

  return LocalOperators(NE2XYkpo*MGEv.ldlt().solve(BGEv), MDE, NE2XYkp2*MPEv.partialPivLu().solve(BPE));
}

//------------------------------------------------------------------------------

XNabla::LocalOperators XNabla::_compute_cell_grad_potential(size_t iT)
{
  const Cell & T = *mesh().cell(iT);

  //------------------------------------------------------------------------------
  // Gradient
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix

  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree() + 1));
  auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kpo_T);
  auto MGT = compute_gram_matrix(basis_RTbkpo_T_quad, basis_RTbkpo_T_quad, quad_2kpo_T, "sym");

  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGT
    = Eigen::MatrixXd::Zero(cellBases(iT).RTbkpo->dimension(), dimensionCell(iT));

  // Boundary contribution
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    
    QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * (degree() + 2));
    auto basis_RTbkpo_nTE_E_quad
      = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp2_E), T.edge_normal(iE));
    auto basis_Pk2p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk2p2, quad_2kp2_E);
    Eigen::MatrixXd BGT_E
      = compute_gram_matrix(basis_RTbkpo_nTE_E_quad, basis_Pk2p2_E_quad, quad_2kp2_E) * edgeOperators(E).potential; // the potential reconstruct the continuous polynomial and insert the missing dofs 

    // Assemble local contribution
    BGT.middleCols(localOffset(T, *E.vertex(0)),2) += BGT_E.middleCols(0,2); // vx(x0),vy(x1)
    BGT.middleCols(localOffset(T, *E.vertex(1)),2) += BGT_E.middleCols(2,2); // vx(x0),vy(x1)
    BGT.block(0, localOffset(T, E), cellBases(iT).RTbkpo->dimension(), dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)))
        = BGT_E.rightCols(dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)));
  } // for iE

  // Cell contribution
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * (degree()));
  auto divergence_Rbck_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).RolybComplkpo, quad_2k_T);
  auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_2k_T);

  auto divergence_Rbkmo_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).Rolybk, quad_2k_T);
  auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_2k_T);

  Eigen::MatrixXd BGTpart = Eigen::MatrixXd::Zero(cellBases(iT).RTbkpo->dimension(),cellBases(iT).Golykmo->dimension() + cellBases(iT).GolyComplk->dimension());
  BGTpart.topRightCorner(cellBases(iT).RolybComplkpo->dimension(),cellBases(iT).GolyComplk->dimension()) 
      = -compute_gram_matrix(divergence_Rbck_T_quad, basis_Gck_T_quad, quad_2k_T);
  BGTpart.block(cellBases(iT).RolybComplkpo->dimension(),0,cellBases(iT).Rolybk->dimension(),cellBases(iT).Golykmo->dimension()) 
      = -compute_gram_matrix(divergence_Rbkmo_T_quad, basis_Gkmo_T_quad, quad_2k_T);

  BGT.rightCols(cellBases(iT).GolyComplk->dimension() + cellBases(iT).Golykmo->dimension()) = BGTpart;

  Eigen::MatrixXd GT = MGT.ldlt().solve(BGT);

  //------------------------------------------------------------------------------
  // Divergence  
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk, quad_2kpo_T);
  auto MDT = compute_gram_matrix(basis_Pk_T_quad, basis_Pk_T_quad, quad_2kpo_T, "sym");
  
  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  #if __cplusplus > 201703L // std>c++17
  auto basis_trace_RTbkpo = evaluate_quad<Trace>::compute(*cellBases(iT).RTbkpo, quad_2kpo_T);
  #else
  auto basis_trace_RTbkpo = evaluate_quad_trace_compute(*cellBases(iT).RTbkpo, quad_2kpo_T);
  #endif // std>c++17
  Eigen::MatrixXd BDT = compute_gram_matrix(basis_Pk_T_quad,basis_trace_RTbkpo, quad_2kpo_T) * GT;
  // decltype(BDT)::foo= 1;

  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  QuadratureRule quad_2kp2_T = generate_quadrature_rule(T, 2 * (degree() + 2));
  auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk2po, quad_2kp2_T);
  auto divergence_Rck2p2_T_quad = evaluate_quad<Divergence>::compute(*cellBases(iT).RolyComplk2p2, quad_2kp2_T); // Roly to have the isomorphism
  Eigen::MatrixXd MPT = compute_gram_matrix(divergence_Rck2p2_T_quad, basis_Pk2po_T_quad, quad_2kp2_T);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Cell contribution
  Eigen::MatrixXd BPT
    = -compute_gram_matrix(
			   evaluate_quad<Function>::compute(*cellBases(iT).RolyComplk2p2, quad_2kp2_T),
			   evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp2_T),
			   quad_2kp2_T
			   ) * GT;
  
  // Boundary contribution
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    
    QuadratureRule quad_2kp3_E = generate_quadrature_rule(E, 2 * (degree() + 3));
    auto basis_Rck2p2_nTE_E_quad
      = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).RolyComplk2p2, quad_2kp3_E), T.edge_normal(iE));
    auto basis_Pk2p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk2p2, quad_2kp3_E);
    Eigen::MatrixXd BPT_E
      = compute_gram_matrix(basis_Rck2p2_nTE_E_quad, basis_Pk2p2_E_quad, quad_2kp3_E) * edgeOperators(E).potential; // the potential reconstruct the continuous polynomial and insert the missing dofs 

    // Assemble local contribution
    BPT.middleCols(localOffset(T, *E.vertex(0)),2) += BPT_E.middleCols(0,2); // vx(x0),vy(x1)
    BPT.middleCols(localOffset(T, *E.vertex(1)),2) += BPT_E.middleCols(2,2); // vx(x0),vy(x1)
    BPT.block(0, localOffset(T, E), 2* PolynomialSpaceDimension<Cell>::RolyCompl(degree() + 2), dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1))) 
        += BPT_E.rightCols(dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)));
  } // for iE

  //------------------------------------------------------------------------------
  // uNa
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  size_t xvl_E_ndofs = 2*PolynomialSpaceDimension<Edge>::Poly(degree()+1);
  size_t xvl_T_ndofs = cellBases(iT).RTbkpo->dimension();

  Eigen::MatrixXd uNa = Eigen::MatrixXd::Zero(T.n_edges()*xvl_E_ndofs+xvl_T_ndofs,dimensionCell(T));
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    Edge & E = *T.edge(iE);
    uNa.middleRows(iE*xvl_E_ndofs,xvl_E_ndofs) = 
              extendOperator(T,E,edgeOperators(E.global_index()).gradient);
  }
  uNa.bottomRows(xvl_T_ndofs) = GT;
  
  return LocalOperators(GT, MDT.ldlt().solve(BDT), MPT.partialPivLu().solve(BPT), uNa);
}


//-----------------------------------------------------------------------------
// local L2 inner product
//-----------------------------------------------------------------------------
Eigen::MatrixXd XNabla::computeL2Product(
                                        const size_t iT,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_Pk2po_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 
  
  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pk2po_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pk2po_T.rows()==1){
      // We have to compute the mass matrix
      QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree()+1));
      w_mass_Pk2po_T = weight.value(T, T.center_mass()) * compute_gram_matrix(evaluate_quad<Function>::compute(*cellBases(iT).Polyk2po, quad_2kpo_T), quad_2kpo_T);
    }else{
      w_mass_Pk2po_T = weight.value(T, T.center_mass()) * mass_Pk2po_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpo_pw_T = generate_quadrature_rule(T, 2 * (degree() + 1) + weight.deg(T));
    auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk2po, quad_2kpo_pw_T);
    std::function<double(const Eigen::Vector2d &)> weight_T 
              = [&T, &weight](const Eigen::Vector2d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pk2po_T = compute_weighted_gram_matrix(weight_T, basis_Pk2po_T_quad, basis_Pk2po_T_quad, quad_2kpo_pw_T, "sym");
  }

  // Compute matrix of L2 product  
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionCell(iT), dimensionCell(iT));

  // We need the potential in the cell
  Eigen::MatrixXd Potential_T = cellOperators(iT).potential;

  // Edge penalty terms
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
        
    QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * (degree()+2) );
    
    // weight and scaling hE
    double max_weight_quad_E = weight.value(T, quad_2kp2_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kp2_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kp2_E[iqn].vector()));
      } // for
    }
    double w_hE = max_weight_quad_E * E.measure();

    // The penalty term int_E (PT v - v_E) * (PT v - v_E) is computed by developping.
    auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk2po, quad_2kp2_E);
    auto basis_Pk2p2_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk2p2, quad_2kp2_E);
    Eigen::MatrixXd gram_Pk2poT_Pk2p2E = compute_gram_matrix(basis_Pk2po_T_quad, basis_Pk2p2_E_quad, quad_2kp2_E);
    
    Eigen::MatrixXd Potential_E = extendOperator(T, E, edgeOperators(E).potential);

    // Contribution of edge E
    L2P += w_hE * ( Potential_T.transpose() * compute_gram_matrix(basis_Pk2po_T_quad, quad_2kp2_E) * Potential_T
                   - Potential_T.transpose() * gram_Pk2poT_Pk2p2E * Potential_E
                   - Potential_E.transpose() * gram_Pk2poT_Pk2p2E.transpose() * Potential_T
                   + Potential_E.transpose() * compute_gram_matrix(basis_Pk2p2_E_quad, quad_2kp2_E) * Potential_E );

  } // for iE

  L2P *= penalty_factor;

  // Cell term
  L2P += Potential_T.transpose() * w_mass_Pk2po_T * Potential_T;

  return L2P;

}


//-----------------------------------------------------------------------------
// Evaluate the potential at a point
//-----------------------------------------------------------------------------
VectorRd XNabla::evaluatePotential_Edge(const size_t iE, const Eigen::VectorXd & vE, const VectorRd & x) const {
  Eigen::VectorXd potential_dofs = edgeOperators(iE).potential * vE;
  VectorRd rv = VectorRd::Zero();
  for (size_t i = 0; i < edgeBases(iE).Polyk2p2->dimension();i++) {
    rv += potential_dofs(i)*edgeBases(iE).Polyk2p2->function(i,x);
  }
  return rv;
}

VectorRd XNabla::evaluatePotential(const size_t iT, const Eigen::VectorXd & vT, const VectorRd & x) const {
  Eigen::VectorXd potential_dofs = cellOperators(iT).potential * vT;
  VectorRd rv = VectorRd::Zero();
  for (size_t i = 0; i < cellBases(iT).Polyk2po->dimension();i++) {
    rv += potential_dofs(i)*cellBases(iT).Polyk2po->function(i,x);
  }
  return rv;
  /* TODO time both and keep fastest
  Eigen::MatrixXd anc_values = Eigen::MatrixXd::Zero(cellBases(iT).Polyk2po->dimension(),2);
  for (size_t i=0; i<cellBases(iT).Polyk2po->dimension(); i++){
    anc_values.row(i) = cellBases(iT).Polyk2po->function(i, x);  
  }

  // return value
  return anc_values.transpose() * cellOperators(iT).potential * vT;
  */
}


