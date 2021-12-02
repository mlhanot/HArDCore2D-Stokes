
#include <basis.hpp>
#include <parallel_for.hpp>

#include "xcurlstokes.hpp"

using namespace HArDCore2D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XCurlStokes::XCurlStokes(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : DDRSpace(stokes_core.mesh(),
	     1 + 2, // 1 + dim 
	     PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree() - 1) + PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()),
	     PolynomialSpaceDimension<Cell>::Poly(stokes_core.degree() - 1)
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output),
    m_edge_operators(stokes_core.mesh().n_edges()),
    m_cell_operators(stokes_core.mesh().n_cells())
{
  m_output << "[XCurlStokes] Initializing" << std::endl;
  if (use_threads) {
    m_output << "[XCurlStokes] Parallel execution" << std::endl;
  } else {
    m_output << "[XCurlStokes] Sequential execution" << std::endl;
  }
  
  // Construct edge curl and potentials
  std::function<void(size_t, size_t)> construct_all_edge_curls_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          m_edge_operators[iE].reset( new LocalOperators(_compute_edge_curl_potential(iE)) );
        } // for iE
      };

  m_output << "[XCurlStokes] Constructing edge curls and potentials" << std::endl;
  parallel_for(mesh().n_edges(), construct_all_edge_curls_potentials, use_threads);

  // Construct cell curls and potentials
  std::function<void(size_t, size_t)> construct_all_cell_curls_potentials
    = [this](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          m_cell_operators[iT].reset( new LocalOperators(_compute_cell_curl_potential(iT)) );
        } // for iT
      };

  m_output << "[XCurlStokes] Constructing cell curls and potentials" << std::endl;
  parallel_for(mesh().n_cells(), construct_all_cell_curls_potentials, use_threads);
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XCurlStokes::interpolate(const FunctionType & q,const FunctionGradType & Dq, const int deg_quad) const
{
  Eigen::VectorXd qh = Eigen::VectorXd::Zero(dimension()); // dimension is the global dimension of dofs

  // Degree of quadrature rules
  size_t dqr = (deg_quad >= 0 ? deg_quad : 2 * degree() + 3);

  // Interpolate at vertices
  std::function<void(size_t, size_t)> interpolate_vertices
  = [this, &qh, q, Dq](size_t start, size_t end)->void
    {
      for (size_t iV = start; iV < end; iV++) {
        const Vertex & V = *mesh().vertex(iV);
        qh(globalOffset(V)) = q(mesh().vertex(iV)->coords());
        qh.segment(globalOffset(V) + 1,2) = Dq(mesh().vertex(iV)->coords());
      } // for iV
    };
  parallel_for(mesh().n_vertices(), interpolate_vertices, m_use_threads);

  
  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &qh, q, Dq, &dqr](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          const Edge & E = *mesh().edge(iE);
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr);
          if (degree() > 0) {
            auto basis_Pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykmo, quad_dqr_E);
            qh.segment(globalOffset(E), PolynomialSpaceDimension<Edge>::Poly(degree() - 1)) 
              = l2_projection(q, *edgeBases(iE).Polykmo, quad_dqr_E, basis_Pkmo_E_quad);
          }
          FunctionType DqnE // function that take the tangential component
              = [this, Dq, &E](const Eigen::Vector2d & coord)->double {
                  return E.tangent().dot(Dq(coord));
              };
          auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_dqr_E);
          qh.segment(globalOffset(E) + PolynomialSpaceDimension<Edge>::Poly(degree() - 1),PolynomialSpaceDimension<Edge>::Poly(degree())) 
            = l2_projection(DqnE, *edgeBases(iE).Polyk, quad_dqr_E, basis_Pk_E_quad);
        } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);

  // Interpolate at cells
  if (degree() == 0) return qh; // no cell dofs
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &qh, q, &dqr](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);
          QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr);
          auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykmo, quad_dqr_T);
          qh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::Poly(degree() - 1)) 
            = l2_projection(q, *cellBases(iT).Polykmo, quad_dqr_T, basis_Pkmo_T_quad);
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);

  return qh;
}

//------------------------------------------------------------------------------
// Curl and potential reconstructions
//------------------------------------------------------------------------------

XCurlStokes::LocalOperators XCurlStokes::_compute_edge_curl_potential(size_t iE)
{
  const Edge & E = *mesh().edge(iE);
  
  //------------------------------------------------------------------------------
  // Curl
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2 * degree());
  auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk, quad_2k_E);
  auto MGE = compute_gram_matrix(basis_Pk_E_quad, basis_Pk_E_quad, quad_2k_E, "sym");

  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGE
    = Eigen::MatrixXd::Zero(edgeBases(iE).Polyk->dimension(), 2 + edgeBases(iE).Polykmo->dimension()); //
  for (size_t i = 0; i < edgeBases(iE).Polyk->dimension(); i++) {
    BGE(i, 0) = -edgeBases(iE).Polyk->function(i, mesh().edge(iE)->vertex(0)->coords());
    BGE(i, 1) = edgeBases(iE).Polyk->function(i, mesh().edge(iE)->vertex(1)->coords());
  } // for i

  QuadratureRule quad_2kmo_E = generate_quadrature_rule(E, 2 * (degree() - 1));
  
  if (degree() > 0) {    
    auto grad_Pk_tE_E_quad = scalar_product(
                                            evaluate_quad<Gradient>::compute(*edgeBases(iE).Polyk, quad_2kmo_E),
                                            E.tangent()
                                            );
    auto basis_Pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykmo, quad_2kmo_E);
    BGE.rightCols(PolynomialSpaceDimension<Edge>::Poly(degree() - 1))
      = -compute_gram_matrix(grad_Pk_tE_E_quad, basis_Pkmo_E_quad, quad_2kmo_E);
  }
 
  //------------------------------------------------------------------------------
  // CE contains the operator "(px0,px1,pi(k-1)p) -> p'", insert remaining dofs "(px0,dxpx0,dypx0,px1,dxpx1,dypx1,pi(k-1)p,pi(k)dpne) -> vxx0,vyx0,vxx1,vyx1,pi(k)vne,pi(k)vnep (= - p')"
  Eigen::MatrixXd CE = MGE.ldlt().solve(BGE);
  Eigen::MatrixXd CEf = Eigen::MatrixXd::Zero(4+2*edgeBases(iE).Polyk->dimension(),dimensionEdge(iE));
  CEf(0,1) = 1.;
  CEf(1,2) = 1.;
  CEf(2,4) = 1.;
  CEf(3,5) = 1.;
  CEf.bottomRows(edgeBases(iE).Polyk->dimension()).leftCols(1) = -CE.leftCols(1);
  CEf.bottomRows(edgeBases(iE).Polyk->dimension()).middleCols(3,1) = -CE.middleCols(1,1);
  CEf.bottomRows(edgeBases(iE).Polyk->dimension()).middleCols(6,edgeBases(iE).Polykmo->dimension()) = -CE.rightCols(edgeBases(iE).Polykmo->dimension());
  CEf.middleRows(4,edgeBases(iE).Polyk->dimension()).rightCols(edgeBases(iE).Polyk->dimension()) = Eigen::MatrixXd::Identity(edgeBases(iE).Polyk->dimension(),edgeBases(iE).Polyk->dimension());


  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  Eigen::MatrixXd BPE
    = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()) + 1,2 + edgeBases(iE).Polykmo->dimension());

  // Enforce the gradient of the potential reconstruction
  BPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree())) = BGE;

  // Enforce the average value of the potential reconstruction
  if (degree() == 0) {
    // We set the average equal to the mean of vertex values
    BPE.bottomRows(1)(0, 0) = 0.5 * E.measure();
    BPE.bottomRows(1)(0, 1) = 0.5 * E.measure();
  } else {
    QuadratureRule quad_kmo_E = generate_quadrature_rule(E, degree() - 1);
    auto basis_Pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykmo, quad_kmo_E);
    
    // We set the average value of the potential equal to the average of the edge unknown
    for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree() - 1); i++) {
      for (size_t iqn = 0; iqn < quad_kmo_E.size(); iqn++) {
        BPE.bottomRows(1)(0, 2 + i) += quad_kmo_E[iqn].w * basis_Pkmo_E_quad[i][iqn];
      } // for iqn
    } // for i
  }
  
  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  Eigen::MatrixXd MPE
    = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()) + 1, PolynomialSpaceDimension<Edge>::Poly(degree() + 1));
  auto grad_Pkpo_tE_E_quad = scalar_product(
					    evaluate_quad<Gradient>::compute(*edgeBases(iE).Polykpo, quad_2k_E),
					    E.tangent()
					    );
  MPE.topRows(PolynomialSpaceDimension<Edge>::Poly(degree()))
    = compute_gram_matrix(basis_Pk_E_quad, grad_Pkpo_tE_E_quad, quad_2k_E);
  
  QuadratureRule quad_kpo_E = generate_quadrature_rule(E, degree() + 1);  
  auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polykpo, quad_kpo_E);  
  for (size_t i = 0; i < PolynomialSpaceDimension<Edge>::Poly(degree() + 1); i++) {
    for (size_t iqn = 0; iqn < quad_kpo_E.size(); iqn++) {
      MPE.bottomRows(1)(0, i) += quad_kpo_E[iqn].w * basis_Pkpo_E_quad[i][iqn];
    } // for iqn
  } // for i
  
  // insert missing dofs on the right
  Eigen::MatrixXd PE = MPE.partialPivLu().solve(BPE);
  Eigen::MatrixXd PEf = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()) + 1,dimensionEdge(iE));
  PEf.leftCols(1) = PE.leftCols(1);
  PEf.middleCols(3,1) = PE.middleCols(1,1);
  PEf.middleCols(6,edgeBases(iE).Polykmo->dimension()) = PE.rightCols(edgeBases(iE).Polykmo->dimension());
    
  return LocalOperators(CEf, PEf);
}

//------------------------------------------------------------------------------

XCurlStokes::LocalOperators XCurlStokes::_compute_cell_curl_potential(size_t iT)
{
  const Cell & T = *mesh().cell(iT);

  //------------------------------------------------------------------------------
  // Curl
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2 * degree());
  auto basis_Pk2_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk2, quad_2k_T);
  auto MGT = compute_gram_matrix(basis_Pk2_T_quad, basis_Pk2_T_quad, quad_2k_T, "sym");

  //------------------------------------------------------------------------------
  // Right-hand side matrix
  
  Eigen::MatrixXd BGT
    = Eigen::MatrixXd::Zero(cellBases(iT).Polyk2->dimension(), dimensionCell(iT));
  Eigen::Matrix2d InvPerp;
  InvPerp << 0.,1.,-1.,0.;

  // Boundary contribution
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    
    QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree() + 1));
    auto basis_Pk2_nTE_E_quad
      = scalar_product<VectorRd>(evaluate_quad<Function>::compute(*cellBases(iT).Polyk2, quad_2kpo_E), InvPerp*T.edge_normal(iE)); // Force eigen product early but template argument deduction fail otherwise
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polykpo, quad_2kpo_E);
    Eigen::MatrixXd BGT_E
      = compute_gram_matrix(basis_Pk2_nTE_E_quad, basis_Pkpo_E_quad, quad_2kpo_E) * edgeOperators(E).potential; // The potential reconstruct the continuous polynomial and insert the missing dofs 

    // Assemble local contribution
    BGT.col(localOffset(T, *E.vertex(0))) += BGT_E.col(0); // p(x0),dxp(x0),dyp(x1)
    BGT.col(localOffset(T, *E.vertex(1))) += BGT_E.col(3);
    if (degree() > 0) {
      BGT.block(0, localOffset(T, E), cellBases(iT).Polyk2->dimension(), dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)))
        += BGT_E.rightCols(PolynomialSpaceDimension<Edge>::Poly(degree() - 1) + PolynomialSpaceDimension<Edge>::Poly(degree()));
    } // if degree() > 0
  } // for iE

  // Cell contribution
  if (degree() > 0) {
    QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree() +1));
    auto curl_Pk2_T_quad = evaluate_quad_rot_compute(*cellBases(iT).Polyk2, quad_2kpo_T);
    auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykmo, quad_2kpo_T);

    BGT.rightCols(PolynomialSpaceDimension<Cell>::Poly(degree() - 1))
      += compute_gram_matrix(curl_Pk2_T_quad, basis_Pkmo_T_quad, quad_2kpo_T);
  } // if degree() > 0

  Eigen::MatrixXd GT = MGT.ldlt().solve(BGT);
  
  //------------------------------------------------------------------------------
  // Potential
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Left-hand side matrix
  
  QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree() + 1));
  auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_T);
  auto curl_Gckp2_T_quad = evaluate_quad<Curl>::compute(*cellBases(iT).GolyComplkp2, quad_2kpo_T); // Goly to have the isomorphism
  Eigen::MatrixXd MPT = compute_gram_matrix(curl_Gckp2_T_quad, basis_Pkpo_T_quad, quad_2kpo_T);

  //------------------------------------------------------------------------------
  // Right-hand side matrix

  // Cell contribution
  Eigen::MatrixXd BPT
    = compute_gram_matrix(
			   evaluate_quad<Function>::compute(*cellBases(iT).GolyComplkp2, quad_2kpo_T),
			   evaluate_quad<Function>::compute(*cellBases(iT).Polyk2, quad_2kpo_T),
			   quad_2kpo_T
			   ) * GT;
  
  // Boundary contribution
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
    
    QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2 * (degree() + 2));
    auto basis_Gckp2_nTE_E_quad
      = scalar_product<VectorRd>(evaluate_quad<Function>::compute(*cellBases(iT).GolyComplkp2, quad_2kp2_E), InvPerp*T.edge_normal(iE));
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polykpo, quad_2kp2_E);
    Eigen::MatrixXd BPT_E
      = -compute_gram_matrix(basis_Gckp2_nTE_E_quad, basis_Pkpo_E_quad, quad_2kp2_E) * edgeOperators(E).potential;

    // Assemble local contribution
    BPT.col(localOffset(T, *E.vertex(0))) += BPT_E.col(0);
    BPT.col(localOffset(T, *E.vertex(1))) += BPT_E.col(3); // offset since dimensionVertex = 3
    if (degree() > 0) {
      BPT.block(0, localOffset(T, E), cellBases(iT).GolyComplkp2->dimension(), dimensionEdge(E) - dimensionVertex(*E.vertex(0)) - dimensionVertex(*E.vertex(1)))
        += BPT_E.rightCols(PolynomialSpaceDimension<Edge>::Poly(degree() - 1) + PolynomialSpaceDimension<Edge>::Poly(degree()));
    } // if degree() > 0
  } // for iE
  
  //------------------------------------------------------------------------------
  // Projection on subspace of xnabla
  //------------------------------------------------------------------------------
  //
  auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_2k_T);
  Eigen::MatrixXd PLGT_R = compute_gram_matrix(basis_Gck_T_quad, basis_Gck_T_quad, quad_2k_T, "sym");
  Eigen::MatrixXd PRGT_R = compute_gram_matrix(basis_Gck_T_quad, basis_Pk2_T_quad, quad_2k_T);
  
  Eigen::MatrixXd PGT = Eigen::MatrixXd::Zero(cellBases(iT).Golykmo->dimension()+cellBases(iT).GolyComplk->dimension(),cellBases(iT).Polyk2->dimension());
  if (degree() > 0) {
    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_2k_T);
    Eigen::MatrixXd PRGT_L = compute_gram_matrix(basis_Gkmo_T_quad, basis_Pk2_T_quad, quad_2k_T);
    Eigen::MatrixXd PLGT_L = compute_gram_matrix(basis_Gkmo_T_quad, basis_Gkmo_T_quad, quad_2k_T, "sym");

    PGT.topRows(cellBases(iT).Golykmo->dimension()) = PLGT_L.ldlt().solve(PRGT_L);
  }
  PGT.bottomRows(cellBases(iT).GolyComplk->dimension()) = PLGT_R.ldlt().solve(PRGT_R);


  //------------------------------------------------------------------------------
  // Reassemble as complete curl (value in dofs xnabla)
  //------------------------------------------------------------------------------
  //
  size_t xnabla_V_ndofs = 2;
  size_t xnabla_E_ndofs = 2*PolynomialSpaceDimension<Edge>::Poly(degree());
  size_t xnabla_T_ndofs = cellBases(iT).Golykmo->dimension() + cellBases(iT).GolyComplk->dimension();
  size_t dimensionXNabla = T.n_vertices()*xnabla_V_ndofs + T.n_edges()*xnabla_E_ndofs + xnabla_T_ndofs;
  Eigen::MatrixXd uCT = Eigen::MatrixXd::Zero(dimensionXNabla,dimensionCell(T));
  for (size_t iV = 0; iV < T.n_vertices(); iV++) {
    uCT(iV*xnabla_V_ndofs,iV*numLocalDofsVertex() + 1) = 1.;
    uCT(iV*xnabla_V_ndofs + 1,iV*numLocalDofsVertex() + 2) = 1.;
  }
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    Edge & E = *T.edge(iE);
    uCT.middleRows(T.n_vertices()*xnabla_V_ndofs + iE*xnabla_E_ndofs,xnabla_E_ndofs) = 
              extendOperator(T,E,edgeOperators(E.global_index()).curl).bottomRows(xnabla_E_ndofs);
  }
  uCT.bottomRows(xnabla_T_ndofs) = PGT * GT;

  return LocalOperators(GT, MPT.partialPivLu().solve(BPT),PGT,uCT);

}


//-----------------------------------------------------------------------------
// local L2 inner product
//-----------------------------------------------------------------------------
Eigen::MatrixXd XCurlStokes::computeL2Product(
                                        const size_t iT,
                                        const double & penalty_factor,
                                        const Eigen::MatrixXd & mass_Pkpo_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 
  
  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pkpo_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pkpo_T.rows()==1){
      // We have to compute the mass matrix
      QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree()+1));
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * compute_gram_matrix(evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_T), quad_2kpo_T);
    }else{
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * mass_Pkpo_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpo_pw_T = generate_quadrature_rule(T, 2 * (degree() + 1) + weight.deg(T));
    auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_pw_T);
    std::function<double(const Eigen::Vector2d &)> weight_T 
              = [&T, &weight](const Eigen::Vector2d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pkpo_T = compute_weighted_gram_matrix(weight_T, basis_Pkpo_T_quad, basis_Pkpo_T_quad, quad_2kpo_pw_T, "sym");
  }

  // Compute matrix of L2 product  
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionCell(iT), dimensionCell(iT));

  // We need the potential in the cell
  Eigen::MatrixXd Potential_T = cellOperators(iT).potential;

  // Edge penalty terms
  for (size_t iE = 0; iE < T.n_edges(); iE++) {
    const Edge & E = *T.edge(iE);
        
    QuadratureRule quad_2kpo_E = generate_quadrature_rule(E, 2 * (degree()+1) );
    
    // weight and scaling hE
    double max_weight_quad_E = weight.value(T, quad_2kpo_E[0].vector());
    // If the weight is not constant, we want to take the largest along the edge
    if (weight.deg(T)>0){
      for (size_t iqn = 1; iqn < quad_2kpo_E.size(); iqn++) {
        max_weight_quad_E = std::max(max_weight_quad_E, weight.value(T, quad_2kpo_E[iqn].vector()));
      } // for
    }
    double w_hE = max_weight_quad_E * E.measure();

    // The penalty term int_E (PT q - q_E) * (PT r - r_E) is computed by developping.
    auto basis_Pkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykpo, quad_2kpo_E);
    auto basis_Pkpo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polykpo, quad_2kpo_E);
    Eigen::MatrixXd gram_PkpoT_PkpoE = compute_gram_matrix(basis_Pkpo_T_quad, basis_Pkpo_E_quad, quad_2kpo_E);
    
    Eigen::MatrixXd Potential_E = extendOperator(T, E, edgeOperators(E).potential);

    // Contribution of edge E
    L2P += w_hE * ( Potential_T.transpose() * compute_gram_matrix(basis_Pkpo_T_quad, quad_2kpo_E) * Potential_T
                   - Potential_T.transpose() * gram_PkpoT_PkpoE * Potential_E
                   - Potential_E.transpose() * gram_PkpoT_PkpoE.transpose() * Potential_T
                   + Potential_E.transpose() * compute_gram_matrix(basis_Pkpo_E_quad, quad_2kpo_E) * Potential_E );

    // Add the contribution of q_E' 
    Eigen::MatrixXd potential_Ep = Eigen::MatrixXd::Zero(PolynomialSpaceDimension<Edge>::Poly(degree()),dimensionEdge(E));
    potential_Ep.rightCols(PolynomialSpaceDimension<Edge>::Poly(degree())) = Eigen::MatrixXd::Identity(PolynomialSpaceDimension<Edge>::Poly(degree()),PolynomialSpaceDimension<Edge>::Poly(degree()));
    auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk, quad_2kpo_E);

    Eigen::MatrixXd Potential_Ep = extendOperator(T, E, potential_Ep);
    L2P += w_hE * Potential_Ep.transpose() * compute_gram_matrix(basis_Pk_E_quad, quad_2kpo_E) * Potential_Ep;

    // Add the contribution of q_V'
    Eigen::MatrixXd potential_Vp = Eigen::MatrixXd::Zero(4,dimensionEdge(E));
    potential_Vp(0,1) = 1.;
    potential_Vp(1,2) = 1.;
    potential_Vp(2,4) = 1.;
    potential_Vp(3,5) = 1.;
    Eigen::MatrixXd Potential_Vp = extendOperator(T, E, potential_Vp);

    L2P += w_hE*w_hE * Potential_Vp.transpose()*Potential_Vp;
  } // for iE

  L2P *= penalty_factor;

  // Cell term
  L2P += Potential_T.transpose() * w_mass_Pkpo_T * Potential_T;

  return L2P;

}


//-----------------------------------------------------------------------------
// Evaluate the potential at a point
//-----------------------------------------------------------------------------
double XCurlStokes::evaluatePotential_Edge(const size_t iE, const Eigen::VectorXd & vE, const VectorRd & x) const {
  Eigen::VectorXd potential_dofs = edgeOperators(iE).potential * vE;
  double rv = 0.;
  for (size_t i = 0; i < edgeBases(iE).Polykpo->dimension();i++) {
    rv += potential_dofs(i)*edgeBases(iE).Polykpo->function(i,x);
  }
  return rv;
}

double XCurlStokes::evaluatePotential(const size_t iT, const Eigen::VectorXd & vT, const VectorRd & x) const {
  // Ancestor of basis of P^{k+1}(T) and values at x
  MonomialScalarBasisCell monomial_ancestor = cellBases(iT).Polykpo->ancestor();
  Eigen::VectorXd anc_values = Eigen::VectorXd::Zero(monomial_ancestor.dimension());
  for (size_t i=0; i<monomial_ancestor.dimension(); i++){
    anc_values(i) = monomial_ancestor.function(i, x);  
  }

  // return value
  return anc_values.transpose() * (cellBases(iT).Polykpo->matrix()).transpose() * cellOperators(iT).potential * vT;

}


