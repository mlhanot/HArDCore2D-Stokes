#include <basis.hpp>
#include <parallel_for.hpp>

#include "xvl.hpp"

using namespace HArDCore2D;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XVL::XVL(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : DDRSpace(stokes_core.mesh(),
	     0,  
	     PolynomialSpaceDimension<Edge>::Poly(stokes_core.degree()+1)*2,
	     PolynomialSpaceDimensionRTb(stokes_core.degree()+1)
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output) {
  
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XVL::interpolate(const FunctionType & W, const int deg_quad) const
{
  Eigen::VectorXd Wh = Eigen::VectorXd::Zero(dimension()); // dimension is the global dimension of dofs

  // Degree of quadrature rules
  size_t dqr = (deg_quad >= 0 ? deg_quad : 2 * degree() + 3);
  
  // Interpolate at edges
  std::function<void(size_t, size_t)> interpolate_edges
    = [this, &Wh, W,  &dqr](size_t start, size_t end)->void
      {
        for (size_t iE = start; iE < end; iE++) {
          const Edge & E = *mesh().edge(iE);
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E, dqr);
          auto basis_Pk2po_E_quad = evaluate_quad<Function>::compute(*edgeBases(iE).Polyk2po, quad_dqr_E);

          std::function<VectorRd (const Eigen::Vector2d &)> WnE 
              = [this, W, &E](const Eigen::Vector2d & coord)->VectorRd {
                  return W(coord) * E.tangent();
              };
          Wh.segment(globalOffset(E),2*PolynomialSpaceDimension<Edge>::Poly(degree()+1)) 
            = l2_projection(WnE, *edgeBases(iE).Polyk2po, quad_dqr_E, basis_Pk2po_E_quad);

        } // for iE
      };
  parallel_for(mesh().n_edges(), interpolate_edges, m_use_threads);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &Wh, W, &dqr](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);
          QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr);
          auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_dqr_T);
          Wh.segment(globalOffset(T), PolynomialSpaceDimensionRTb(degree()+1)) 
            = l2_projection(W, *cellBases(iT).RTbkpo, quad_dqr_T, basis_RTbkpo_T_quad);
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);

  return Wh;
}

//-----------------------------------------------------------------------------
// local L2 inner product
//-----------------------------------------------------------------------------
Eigen::MatrixXd XVL::computeL2Product(
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
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * compute_gram_matrix(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kpo_T), quad_2kpo_T);
    }else{
      w_mass_Pkpo_T = weight.value(T, T.center_mass()) * mass_Pkpo_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpo_pw_T = generate_quadrature_rule(T, 2 * (degree() + 1) + weight.deg(T));
    auto basis_RTbkpo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kpo_pw_T);
    std::function<double(const Eigen::Vector2d &)> weight_T 
              = [&T, &weight](const Eigen::Vector2d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pkpo_T = compute_weighted_gram_matrix(weight_T, basis_RTbkpo_T_quad, basis_RTbkpo_T_quad, quad_2kpo_pw_T, "sym");
  }

  // Compute matrix of L2 product  
  Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(dimensionCell(iT), dimensionCell(iT));

  // Projector on cell
  Eigen::MatrixXd Potential_T = Eigen::MatrixXd::Zero(this->numLocalDofsCell(),dimensionCell(iT));
  Potential_T.rightCols(this->numLocalDofsCell()) = Eigen::MatrixXd::Identity(this->numLocalDofsCell(),this->numLocalDofsCell());

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

    // The penalty term int_E (W.nE - W_E) * (W.nE - W_E) is computed by developping.
    auto basis_RTbkpo_TnE_quad = scalar_product(evaluate_quad<Function>::compute(*cellBases(iT).RTbkpo, quad_2kp2_E), E.tangent());
    auto basis_Pk2po_E_quad = evaluate_quad<Function>::compute(*edgeBases(E.global_index()).Polyk2po, quad_2kp2_E);
    Eigen::MatrixXd gram_RTbkpoTnE_Pk2poE = compute_gram_matrix(basis_RTbkpo_TnE_quad, basis_Pk2po_E_quad, quad_2kp2_E);
    
    Eigen::MatrixXd Potential_E = extendOperator(T, E, Eigen::MatrixXd::Identity(dimensionEdge(E),dimensionEdge(E)));
    // Contribution of edge E
    L2P += w_hE * ( Potential_T.transpose() * compute_gram_matrix(basis_RTbkpo_TnE_quad, quad_2kp2_E) * Potential_T
                   -  Potential_T.transpose() * gram_RTbkpoTnE_Pk2poE * Potential_E
                   -  Potential_E.transpose() * gram_RTbkpoTnE_Pk2poE.transpose() * Potential_T
                   + Potential_E.transpose() * compute_gram_matrix(basis_Pk2po_E_quad, quad_2kp2_E) * Potential_E );

  } // for iE

  L2P *= penalty_factor;

  // Cell term
  L2P += Potential_T.transpose() * w_mass_Pkpo_T * Potential_T;

  return L2P;

}


//-----------------------------------------------------------------------------
// Evaluate the potential at a point
//-----------------------------------------------------------------------------
MatrixRd XVL::evaluatePotential(const size_t iT, const Eigen::VectorXd & vT, const VectorRd & x) const {
  Eigen::MatrixXd anc_values = Eigen::MatrixXd::Zero(cellBases(iT).RTbkpo->dimension(),4);
  for (size_t i=0; i<cellBases(iT).RTbkpo->dimension(); i++){
    Eigen::Map<Eigen::VectorXd> vtmp((cellBases(iT).RTbkpo->function(i, x)).data(), 4);

    anc_values.row(i) = vtmp;
    }

  // return value
  return Eigen::Map<MatrixRd>(static_cast<Eigen::VectorXd>(anc_values.transpose() * vT).data());

}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

XSL::XSL(const StokesCore & stokes_core, bool use_threads, std::ostream & output)
  : DDRSpace(stokes_core.mesh(),
	     0,  
	     0,
	     PolynomialSpaceDimension<Cell>::Poly(stokes_core.degree())
	     ),
    m_stokes_core(stokes_core),
    m_use_threads(use_threads),
    m_output(output) {
  
}

//------------------------------------------------------------------------------
// Interpolator
//------------------------------------------------------------------------------

Eigen::VectorXd XSL::interpolate(const FunctionType & w, const int deg_quad) const
{
  Eigen::VectorXd wh = Eigen::VectorXd::Zero(dimension()); // dimension is the global dimension of dofs

  // Degree of quadrature rules
  size_t dqr = (deg_quad >= 0 ? deg_quad : 2 * degree() + 3);

  // Interpolate at cells
  std::function<void(size_t, size_t)> interpolate_cells
    = [this, &wh, w, &dqr](size_t start, size_t end)->void
      {
        for (size_t iT = start; iT < end; iT++) {
          const Cell & T = *mesh().cell(iT);
          QuadratureRule quad_dqr_T = generate_quadrature_rule(T, dqr);
          auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk, quad_dqr_T);
          wh.segment(globalOffset(T), PolynomialSpaceDimension<Cell>::Poly(degree())) 
            = l2_projection(w, *cellBases(iT).Polyk, quad_dqr_T, basis_Pk_T_quad);
        } // for iT
      };
  parallel_for(mesh().n_cells(), interpolate_cells, m_use_threads);

  return wh;
}

//-----------------------------------------------------------------------------
// local L2 inner product
//-----------------------------------------------------------------------------
Eigen::MatrixXd XSL::computeL2Product(
                                        const size_t iT,
                                        const Eigen::MatrixXd & mass_Pk_T,
                                        const IntegralWeight & weight
                                        ) const
{
  const Cell & T = *mesh().cell(iT); 
  
  // create the weighted mass matrix, with simple product if weight is constant
  Eigen::MatrixXd w_mass_Pk_T;
  if (weight.deg(T)==0){
    // constant weight
    if (mass_Pk_T.rows()==1){
      // We have to compute the mass matrix
      QuadratureRule quad_2kpo_T = generate_quadrature_rule(T, 2 * (degree()+1));
      w_mass_Pk_T = weight.value(T, T.center_mass()) * compute_gram_matrix(evaluate_quad<Function>::compute(*cellBases(iT).Polyk, quad_2kpo_T), quad_2kpo_T);
    }else{
      w_mass_Pk_T = weight.value(T, T.center_mass()) * mass_Pk_T;
    }
  }else{
    // weight is not constant, we create a weighted mass matrix
    QuadratureRule quad_2kpo_pw_T = generate_quadrature_rule(T, 2 * (degree() + 1) + weight.deg(T));
    auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polyk, quad_2kpo_pw_T);
    std::function<double(const Eigen::Vector2d &)> weight_T 
              = [&T, &weight](const Eigen::Vector2d &x)->double {
                  return weight.value(T, x);
                };
    w_mass_Pk_T = compute_weighted_gram_matrix(weight_T, basis_Pk_T_quad, basis_Pk_T_quad, quad_2kpo_pw_T, "sym");
  }


  return w_mass_Pk_T;

}


//-----------------------------------------------------------------------------
// Evaluate the potential at a point
//-----------------------------------------------------------------------------
double XSL::evaluatePotential(const size_t iT, const Eigen::VectorXd & vT, const VectorRd & x) const {
  double rv = 0.;
  for (size_t i=0; i<cellBases(iT).Polyk->dimension(); i++){
    rv += cellBases(iT).Polyk->function(i, x)*vT(i);
    }
  // return value
  return rv;

}

