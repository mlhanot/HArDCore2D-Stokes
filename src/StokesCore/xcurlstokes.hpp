#ifndef XCURLSTOKES_HPP
#define XCURLSTOKES_HPP

#include <ddrspace.hpp>
#include <integralweight.hpp>

#include "stokescore.hpp"

namespace HArDCore2D
{

  /// Discrete H1 space: local operators, L2 product and global interpolator
  // Dofs V : q(xV), dxq(xV), dyq(xV)
  // Dofs E : pkmo(q), pk(dEq)
  // Dofs F : pkmo(q)
  class XCurlStokes : public DDRSpace
  {
  public:
    typedef std::function<double(const Eigen::Vector2d &)> FunctionType;
    typedef std::function<Eigen::Vector2d(const Eigen::Vector2d &)> FunctionGradType;

    /// A structure to store local operators (gradient and potential)
    struct LocalOperators
    {
      LocalOperators(
                     const Eigen::MatrixXd & _curl, ///< Curl operator
                     const Eigen::MatrixXd & _potential, ///< Potential operator
                     const Eigen::MatrixXd & _proj = Eigen::MatrixXd::Zero(0,0),
                     const Eigen::MatrixXd & _ucurl = Eigen::MatrixXd::Zero(0,0)
                     )
        : curl(_curl),
          potential(_potential),
          proj(_proj),
          ucurl(_ucurl)
      {
        // Do nothing
      }
      
      Eigen::MatrixXd curl;
      Eigen::MatrixXd potential;
      Eigen::MatrixXd proj; // Polyk2 -> Gkmo+Gck, Not available for edges
      Eigen::MatrixXd ucurl; // DofsCell -> DofsCell (including edges dofs), Not available for edges
    };
    
    /// Constructor
    XCurlStokes(const StokesCore & stokes_core, bool use_threads = true, std::ostream & output = std::cout);

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
                const FunctionType & q, ///< The function to interpolate
                const FunctionGradType & Dq, /// Gradient of the function to interpolate
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
    // The mass matrix of P^{k+1}(T) is the most expensive mass matrix in the calculation of this norm, which
    // is why there's the option of passing it as parameter if it's been already pre-computed when the norm is called.
    Eigen::MatrixXd computeL2Product(
                                     const size_t iT, ///< index of the cell
                                     const double & penalty_factor = 1., ///< pre-factor for stabilisation term
                                     const Eigen::MatrixXd & mass_Pkpo_T = Eigen::MatrixXd::Zero(1,1), ///< if pre-computed, the mass matrix of P^{k+1}(T); if none is pre-computed, passing Eigen::MatrixXd::Zero(1,1) will force the calculation
                                     const IntegralWeight & weight = IntegralWeight(1.) ///< weight function in the L2 product, defaults to 1
                                     ) const;

    /// Evaluate the value of the potential at a point x
    double evaluatePotential_Edge(
                            const size_t iE, ///< index of the edge in which to take the potential
                            const Eigen::VectorXd & vE, ///< vector of local DOFs
                            const VectorRd & x ///< point at which to evaluate the potential
                            ) const;
    double evaluatePotential(
                            const size_t iT, ///< index of the cell in which to take the potential
                            const Eigen::VectorXd & vT, ///< vector of local DOFs
                            const VectorRd & x ///< point at which to evaluate the potential
                            ) const;

  private:    
    LocalOperators _compute_edge_curl_potential(size_t iE);
    LocalOperators _compute_cell_curl_potential(size_t iT);    

    const StokesCore & m_stokes_core;
    bool m_use_threads;
    std::ostream & m_output;

    // Containers for local operators
    std::vector<std::unique_ptr<LocalOperators> > m_edge_operators;
    std::vector<std::unique_ptr<LocalOperators> > m_cell_operators;
  };

} // end of namespace HArDCore2D

#endif
