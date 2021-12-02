#ifndef XVL_HPP
#define XVL_HPP

#include <ddrspace.hpp>
#include <integralweight.hpp>

#include "stokescore.hpp"

namespace HArDCore2D
{

  /// Discrete L2 space: L2 product and global interpolator
  // Dofs E : pk2po(W.nE)
  // Dofs F : pRbckpo(W), pRbk(W), pRk2(W)
class XVL : public DDRSpace
{
public:
  typedef std::function<Eigen::Matrix2d(const Eigen::Vector2d &)> FunctionType;
  typedef std::function<Eigen::Vector2d(const Eigen::Vector2d &)> FunctionDivType;

  /// Constructor
  XVL(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

  /// Return the mesh
  const Mesh & mesh() const
  {
    return m_stokes_core.mesh();
  }
  
  /// Return the polynomial degree
  const size_t & degree() const
  {
    return m_stokes_core.degree();
  }
  
  /// Interpolator of a continuous function
  Eigen::VectorXd interpolate(
              const FunctionType & W, ///< The function to interpolate
              const int deg_quad = -1 ///< The optional degre of quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
              ) const;

  /// Return cell bases for the cell of index iT
  inline const StokesCore::CellBases & cellBases(size_t iT) const
  {
    return m_stokes_core.cellBases(iT);
  }

  /// Return cell bases for cell T
  inline const StokesCore::CellBases & cellBases(const Cell & T) const
  {
    return m_stokes_core.cellBases(T.global_index());
  }
  
  /// Return edge bases for the edge of index iE
  inline const StokesCore::EdgeBases & edgeBases(size_t iE) const
  {
    return m_stokes_core.edgeBases(iE);
  }

  /// Return edge bases for edge E
  inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
  {
    return m_stokes_core.edgeBases(E.global_index());
  }

  /// Compute the matrix of the (weighted) L2-product for the cell of index iT.
  // The mass matrix of P^{k+1}(T) is the most expensive mass matrix in the calculation of this norm, which
  // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
  Eigen::MatrixXd computeL2Product(
                                   const size_t iT, ///< index of the cell
                                   const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                   const Eigen::MatrixXd & mass_Pkpo_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of P^{k+1}(T); if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                   const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                   ) const;

  /// Evaluate the value of the potential at a point x
  MatrixRd evaluatePotential(
                          const size_t iT, ///< index of the cell in which to take the potential
                          const Eigen::VectorXd & vT, ///< vector of local DOFs
                          const VectorRd & x ///< point at which to evaluate the potential
                          ) const;

private:    

  const StokesCore & m_stokes_core;
  bool m_use_threads;
  std::ostream & m_output;

};
class XSL : public DDRSpace
{
public:
  typedef std::function<double(const Eigen::Vector2d &)> FunctionType;

  /// Constructor
  XSL(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

  /// Return the mesh
  const Mesh & mesh() const
  {
    return m_stokes_core.mesh();
  }
  
  /// Return the polynomial degree
  const size_t & degree() const
  {
    return m_stokes_core.degree();
  }
  
  /// Interpolator of a continuous function
  Eigen::VectorXd interpolate(
              const FunctionType & W, ///< The function to interpolate
              const int deg_quad = -1 ///< The optional degre of quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
              ) const;

  /// Return cell bases for the cell of index iT
  inline const StokesCore::CellBases & cellBases(size_t iT) const
  {
    return m_stokes_core.cellBases(iT);
  }

  /// Return cell bases for cell T
  inline const StokesCore::CellBases & cellBases(const Cell & T) const
  {
    return m_stokes_core.cellBases(T.global_index());
  }
  
  /// Return edge bases for the edge of index iE
  inline const StokesCore::EdgeBases & edgeBases(size_t iE) const
  {
    return m_stokes_core.edgeBases(iE);
  }

  /// Return edge bases for edge E
  inline const StokesCore::EdgeBases & edgeBases(const Edge & E) const
  {
    return m_stokes_core.edgeBases(E.global_index());
  }

  /// Compute the matrix of the (weighted) L2-product for the cell of index iT.
  // The mass matrix of P^{k}(T) is the most expensive mass matrix in the calculation of this norm, which
  // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
  Eigen::MatrixXd computeL2Product(
                                   const size_t iT, ///< index of the cell
                                   const Eigen::MatrixXd & mass_Pk_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of P^{k}(T); if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                   const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                   ) const;

  /// Evaluate the value of the potential at a point x
  double evaluatePotential(
                          const size_t iT, ///< index of the cell in which to take the potential
                          const Eigen::VectorXd & vT, ///< vector of local DOFs
                          const VectorRd & x ///< point at which to evaluate the potential
                          ) const;

private:    

  const StokesCore & m_stokes_core;
  bool m_use_threads;
  std::ostream & m_output;

};

} // end of namespace HArDCore2D

#endif
