#ifndef XNABLA_HPP
#define XNABLA_HPP

#include <ddrspace.hpp>
#include <integralweight.hpp>

#include "stokescore.hpp"

namespace HArDCore2D
{

  /// Discrete H1 space: local operators, L2 product and global interpolator
  // Dofs V : vx(xV), vy(xV)
  // Dofs E : pk(vE), pk(vEp)
  // Dofs F : gkmo(v), gck(v)
  class XNabla : public DDRSpace
  {
  public:
    typedef std::function<Eigen::Vector2d(const Eigen::Vector2d &)> FunctionType;
    typedef std::function<Eigen::Matrix2d(const Eigen::Vector2d &)> FunctionGradType;
    typedef std::function<double(const Eigen::Vector2d &)> FunctionDivType;

    /// A structure to store local operators (gradient, divergence and potential)
    struct LocalOperators
    {
      LocalOperators(
                     const Eigen::MatrixXd & _grad, ///< Grad operator
                     const Eigen::MatrixXd & _div, ///< Div operator
                     const Eigen::MatrixXd & _potential, ///< Potential operator
                     const Eigen::MatrixXd & _ugrad = Eigen::MatrixXd::Zero(0,0) ///< Full (cell + edge) Grad operator
                     )
        : gradient(_grad),
          divergence(_div),
          potential(_potential),
          ugradient(_ugrad)
      {
        // Do nothing
      }
      
      Eigen::MatrixXd gradient;
      Eigen::MatrixXd divergence;
      Eigen::MatrixXd potential;
      Eigen::MatrixXd ugradient; // Not available for edges
    };
    
    /// Constructor
    XNabla(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

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
                const FunctionType & v, ///< The function to interpolate
                const int deg_quad = -1 ///< The optional degre of quadrature rules to compute the interpolate. If negative, then 2*degree()+3 will be used.
                ) const;

    /// Return edge operators for the edge of index iE
    inline const LocalOperators & edgeOperators(size_t iE) const
    {
      return *m_edge_operators[iE];
    }

    /// Return edge operators for edge E
    inline const LocalOperators & edgeOperators(const Edge & E) const
    {
      return *m_edge_operators[E.global_index()];
    }
    
    /// Return cell operators for the cell of index iT
    inline const LocalOperators & cellOperators(size_t iT) const
    {
      return *m_cell_operators[iT];
    }

    /// Return cell operators for cell T
    inline const LocalOperators & cellOperators(const Cell & T) const
    {
      return *m_cell_operators[T.global_index()];
    }

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
    // The mass matrix of P^{k+1}(T)^2 is the most expensive mass matrix in the calculation of this norm, which
    // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
    Eigen::MatrixXd computeL2Product(
                                     const size_t iT, ///< index of the cell
                                     const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & mass_Pk2po_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of P^{k+1}(T)^2; if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                     const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                     ) const;

    /// Evaluate the value of the potential at a point x
    VectorRd evaluatePotential_Edge(
                            const size_t iE, ///< index of the cell in which to take the potential
                            const Eigen::VectorXd & vE, ///< vector of local DOFs
                            const VectorRd & x ///< point at which to evaluate the potential
                            ) const;
    VectorRd evaluatePotential(
                            const size_t iT, ///< index of the cell in which to take the potential
                            const Eigen::VectorXd & vT, ///< vector of local DOFs
                            const VectorRd & x ///< point at which to evaluate the potential
                            ) const;

  private:    
    LocalOperators _compute_edge_grad_potential(size_t iE);
    LocalOperators _compute_cell_grad_potential(size_t iT);    

    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

    // Containers for local operators
    std::vector<std::unique_ptr<LocalOperators> > m_edge_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_cell_operators;
  };

} // end of namespace HArDCore2D

#endif
