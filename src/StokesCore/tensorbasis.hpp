#ifndef TENSORBASIS_HPP
#define TENSORBASIS_HPP

#include <basis.hpp>

namespace HArDCore2D {

  static inline size_t PolynomialSpaceDimensionRRolyb(int k) {
      return PolynomialSpaceDimension<Cell>::Poly(k) - 1 ;
  }

  static inline size_t PolynomialSpaceDimensionRolybCompl(int k) {
      return PolynomialSpaceDimension<Cell>::Poly(k - 2);
  }

  static inline size_t PolynomialSpaceDimensionRTb(int k) {
      return PolynomialSpaceDimensionRolybCompl(k) + PolynomialSpaceDimensionRRolyb(k-1) + 2*PolynomialSpaceDimension<Cell>::Roly(k - 1);
  }

  //---------------------------------------------------------------------
  //      BASES FOR Rbc
  //---------------------------------------------------------------------
  /// Basis for traceless subspace of Rc(T)^2
  class RolybComplBasisCell
  {
    public:
    typedef MatrixRd FunctionValue;
    typedef void GradientValue;
    typedef void CurlValue;
    typedef VectorRd DivergenceValue;
    typedef double TraceValue;

    typedef Cell GeometricSupport;

    static const TensorRankE tensorRank = Matrix;
    static const bool hasFunction = true;
    static const bool hasGradient = false;
    static const bool hasCurl = false;
    static const bool hasDivergence = true;
    static const bool hasTrace = true;

    /// Constructor
    RolybComplBasisCell(
        const Cell &T, ///< A mesh element
        size_t degree  ///< The maximum polynomial degree to be considered
    );

    /// Dimension of the basis
    inline size_t dimension() const
    {
      return PolynomialSpaceDimension<Cell>::Poly(m_degree - 2);
    }

    /// Evaluate the i-th basis function at point x
    FunctionValue function(size_t i, const VectorRd &x) const;

    /// Evaluate the divergence of the i-th basis function at point x
    DivergenceValue divergence(size_t i, const VectorRd &x) const;

    /// Evaluate the trace of the i-th basis function at point x
    TraceValue trace(size_t i, const VectorRd &x) const { return 0.;}

    private:
    /// Coordinate transformation
    inline VectorRd _coordinate_transform(const VectorRd &x) const
    {
      return (x - m_xT) / m_hT;
    }

    size_t m_degree;
    VectorRd m_xT;
    double m_hT;
    std::vector<VectorZd> m_powers;
  };

  //---------------------------------------------------------------------
  //      BASES FOR Rb
  //---------------------------------------------------------------------
  /// Basis for Rb(T) = div-1(grad(Pk))
  class RolybBasisCell
  {
    public:
    typedef MatrixRd FunctionValue;
    typedef void GradientValue;
    typedef void CurlValue;
    typedef VectorRd DivergenceValue;
    typedef double TraceValue;

    typedef Cell GeometricSupport;

    static const TensorRankE tensorRank = Matrix;
    static const bool hasFunction = true;
    static const bool hasGradient = false;
    static const bool hasCurl = false;
    static const bool hasDivergence = true;
    static const bool hasTrace = true;

    /// Constructor
    RolybBasisCell(
        const Cell &T, ///< A mesh element
        size_t degree  ///< The maximum polynomial degree to be considered
    );

    /// Dimension of the basis
    inline size_t dimension() const
    {
      return PolynomialSpaceDimension<Cell>::Poly(m_degree) - 1;
    }

    /// Evaluate the i-th basis function at point x
    FunctionValue function(size_t i, const VectorRd &x) const;

    /// Evaluate the divergence of the i-th basis function at point x
    DivergenceValue divergence(size_t i, const VectorRd &x) const;

    /// Evaluate the trace of the i-th basis function at point x
    TraceValue trace(size_t i, const VectorRd &x) const;

    private:
    /// Coordinate transformation
    inline VectorRd _coordinate_transform(const VectorRd &x) const
    {
      return (x - m_xT) / m_hT;
    }

    size_t m_degree;
    VectorRd m_xT;
    double m_hT;
    std::vector<VectorZd> m_powers;
  };
  //-------------------Extended CURL BASIS-----------------------------------------------------------

  /// Basis for the space of curls (vectorial rot) of polynomials.
  /** To construct a basis of R^k, assumes that the vector basis from which it is constructed is a basis for P^{k+1}/P^0 */
  /// Extended to support divergence
  template <typename BasisType>
  class CurlBasiswDiv 
  {
    public:
      typedef VectorRd FunctionValue;
      typedef Eigen::Matrix<double, dimspace, dimspace> GradientValue;
      typedef Eigen::Matrix<double, dimspace, dimspace> CurlValue;
      typedef double DivergenceValue;

      typedef typename BasisType::GeometricSupport GeometricSupport;

      static const TensorRankE tensorRank = Vector;
      static const bool hasFunction = true;
      static const bool hasGradient = false;
      static const bool hasCurl = false;
      static const bool hasDivergence = true;

      /// Constructor
      CurlBasiswDiv(const BasisType &basis)
          : m_basis(basis)
      {
        static_assert(BasisType::tensorRank == Scalar && std::is_same<typename BasisType::GeometricSupport, Cell>::value,
                      "Curl basis can only be constructed starting from scalar bases on elements");
        static_assert(BasisType::hasCurl,
                      "Curl basis requires curl() for the original basis to be available");
      }

      /// Compute the dimension of the basis
      inline size_t dimension() const
      {
        return m_basis.dimension();
      }

      /// Evaluate the i-th basis function at point x
      inline FunctionValue function(size_t i, const VectorRd &x) const
      {
        return m_basis.curl(i, x);
      }

      // Evaluate the divergence i-th basis function at point x
      inline DivergenceValue divergence(size_t i, const VectorRd &x) const
      {
        return 0.;
      }


    private:
      size_t m_degree;
      BasisType m_basis;
  };

  //-------------------Extended GolyCompl BASIS-----------------------------------------------------------
  /// Basis for the complement G^{c,k}(T) in P^k(F)^2 of the range of the gradient on a face.
  // Extended to support curl
  // TODO We call the rot operator curl to reuse the Family class, rename it rot and extend the class
  class GolyComplBasiswCurlCell
  {
    public:
      typedef VectorRd FunctionValue;
      typedef Eigen::Matrix<double, dimspace, dimspace> GradientValue;
      typedef double CurlValue; // rot
      typedef double DivergenceValue;

      typedef Cell GeometricSupport;

      static const TensorRankE tensorRank = Vector;
      static const bool hasFunction = true;
      static const bool hasGradient = false;
      static const bool hasCurl = true; // rot
      static const bool hasDivergence = false;

      /// Constructor
      GolyComplBasiswCurlCell(
          const Cell &T, ///< A mesh element
          size_t degree  ///< The maximum polynomial degree to be considered
      ) : m_degree(degree) {
        m_rot.row(0) << 0., 1.;
        m_rot.row(1) << -1., 0.;
        if (degree > 0) {
          m_Rck_basis.reset(new RolyComplBasisCell(T, degree));
        }
      }

      /// Dimension of the basis
      inline size_t dimension() const
      {
        return PolynomialSpaceDimension<Cell>::GolyCompl(m_degree);
      }

      /// Evaluate the i-th basis function at point x
      FunctionValue function(size_t i, const VectorRd &x) const {
        return m_rot * m_Rck_basis->function(i,x);
      }

      /// Evaluate the curl (rot) of the i-th basis function at point x 
      CurlValue curl(size_t i, const VectorRd &x) const {
        return -m_Rck_basis->divergence(i,x);
      }

    private:
      size_t m_degree;
      Eigen::Matrix2d m_rot;  // Rotation of -pi/2: the basis of Gck is the rotated basis of Rck
      std::shared_ptr<RolyComplBasisCell> m_Rck_basis;  // A basis of Rck, whose rotation gives a basis of Gck
  };

  //---------------------------------------------------------------------
  //      Extended Family with trace
  //---------------------------------------------------------------------
  template <typename BasisType>
  class trFamily : public Family<BasisType> {
    public:
      typedef typename BasisType::TraceValue TraceValue;

      static const bool hasTrace = BasisType::hasTrace;
          
      using Family<BasisType>::Family; // Inherit all constructor from Family (we only need to add the support for trace)
      trFamily( const Family<BasisType> & basis_family) : Family<BasisType>(basis_family) {}; // Extend a family with trace

      /// Evaluate the i-th trace at point x
      TraceValue trace(size_t i, const VectorRd & x) const
      {
        static_assert(hasTrace, "Call to trace() not available");
        
        TraceValue f = this->matrix()(i, 0) * this->ancestor().trace(0, x);
        for (auto j = 1; j < this->matrix().cols(); j++) {
          f += this->matrix()(i, j) * this->ancestor().trace(j, x);
        } // for j
        return f;
      }
      
      /// Evaluate the i-th trace at a quadrature point iqn, knowing all the values of ancestor basis functions at the quadrature nodes (provided by eval_quad)
      TraceValue trace(size_t i, size_t iqn, const boost::multi_array<TraceValue, 2> &ancestor_value_quad) const
      {
        static_assert(hasTrace, "Call to trace() not available");

        TraceValue f = this->matrix()(i, 0) * ancestor_value_quad[0][iqn];
        for (auto j = 1; j < this->matrix().cols(); j++) {
          f += this->matrix()(i, j) * ancestor_value_quad[j][iqn];
        } // for j
        return f;
      }
  };

  //---------------------------------------------------------------------
  //      Extended TensorizedVectorFamily with rot
  //---------------------------------------------------------------------

  template<typename ScalarFamilyType, size_t N>
  class TensorizedVectorFamilywrot : public TensorizedVectorFamily<ScalarFamilyType,N>
  {
  public:
    typedef double RotValue;

    static const bool hasRot = ( ScalarFamilyType::hasGradient && N==2 );
   
    using TensorizedVectorFamily<ScalarFamilyType,N>::TensorizedVectorFamily; // Inherit all constructor
    
    /// Evaluate the rot of the i-th basis function at point x
    RotValue rot(size_t i, const VectorRd & x) const
    {
      static_assert(hasRot, "Call to rot() not available");
      
      int rs = (i < this->ancestor().dimension())? -1 : 1;
      int ls = (i < this->ancestor().dimension())? 1 : 0;
      return rs*this->ancestor().gradient(i % this->ancestor().dimension(), x)(ls);
    }
    
    /// Evaluate the rot of the i-th basis function at a quadrature point iqn, knowing all the gradients of ancestor basis functions at the quadrature nodes (provided by eval_quad)
    RotValue rot(size_t i, size_t iqn, const boost::multi_array<VectorRd, 2> &ancestor_gradient_quad) const
    {
      static_assert(hasRot, "Call to rot() not available");

      int rs = (i < this->ancestor().dimension())? -1 : 1;
      int ls = (i < this->ancestor().dimension())? 1 : 0;
      return rs*ancestor_gradient_quad[i % this->ancestor().dimension()][iqn](ls);      
    }
  };

  //---------------------------------------------------------------------
  //      Matrix basis from vector
  //---------------------------------------------------------------------
  // Construct matrix where each row is an vector component
  template <typename VectorFamilyType, size_t N>
  class MatrixVectorFamily {
    public:
        typedef typename Eigen::Matrix<double, N, N> FunctionValue;
        typedef typename Eigen::Matrix<double, N, N*dimspace> GradientValue;
        typedef MatrixVectorFamily<VectorFamilyType,N> CurlValue;
        typedef typename Eigen::Matrix<double, N, 1> DivergenceValue;
        typedef double TraceValue;

        typedef typename VectorFamilyType::GeometricSupport GeometricSupport;

        static const TensorRankE tensorRank = Matrix;
        static const bool hasFunction = VectorFamilyType::hasFunction;
        static const bool hasGradient = false;
        static const bool hasDivergence = VectorFamilyType::hasDivergence;
        static const bool hasCurl = false;
        static const bool hasTrace = true;

        MatrixVectorFamily(const VectorFamilyType & vector_family)
          : m_vector_family(vector_family),
            m_E(N) {
          static_assert(VectorFamilyType::tensorRank == Vector,
                "VectorMatrix family can only be constructed from vector families");
                
          // Construct the basis for NxN matrices
          for (size_t j = 0; j < N; j++){
              m_E[j] = Eigen::Matrix<double, N, 1>::Zero();
              m_E[j](j,0) = 1.;
          }
        }

      // Return the dimension of the family
      inline size_t dimension() const {
          return m_vector_family.dimension() * N;
      }

      // Evaluate the i-th basis function at point x
      FunctionValue function(size_t i, const VectorRd & x) const {
        static_assert(hasFunction, "Call to function() not available");
        
        return m_E[i / m_vector_family.dimension()] * m_vector_family.function(i % m_vector_family.dimension(), x).transpose() ; // <N,1> * <1,N> = <N,N>
      }
      
      /// Evaluate the i-th basis function at a quadrature point iqn, knowing all the values of ancestor basis functions at the quadrature nodes (provided by eval_quad)
      FunctionValue function(size_t i, size_t iqn, const boost::multi_array<double, 2> &ancestor_value_quad) const {
        static_assert(hasFunction, "Call to function() not available");

        return m_E[i / m_vector_family.dimension()] * ancestor_value_quad[i % m_vector_family.dimension()][iqn].transpose() ;
      }

      /// Evaluate the divergence of the i-th basis function at point x
      DivergenceValue divergence(size_t i, const VectorRd & x) const {
        static_assert(hasDivergence, "Call to divergence() not available");

        return m_vector_family.divergence(i % m_vector_family.dimension(), x) * m_E[i/m_vector_family.dimension()];
      }
      
      /// Evaluate the divergence of the i-th basis function at a quadrature point iqn, knowing all the gradients of ancestor basis functions at the quadrature nodes (provided by eval_quad)
      DivergenceValue divergence(size_t i, size_t iqn, const boost::multi_array<VectorRd, 2> &ancestor_divergence_quad) const {
        static_assert(hasDivergence, "Call to divergence() not available");

        return ancestor_divergence_quad[i % m_vector_family.dimension()][iqn] * m_E[i/m_vector_family.dimension()];      
      }

      /// Evaluate the trace of the i-th basis function at point x
      TraceValue trace(size_t i, const VectorRd &x) const {
          return this->function(i,x).trace();
      }

      /// Return the ancestor (family that has been tensorized)
      inline const VectorFamilyType &ancestor() const
      {
        return m_vector_family;
      }
      
      /// Return the dimension of the matrices in the family
      inline const size_t matrixSize() const
      {
        return N;
      }
    
    private:
      VectorFamilyType m_vector_family;
      std::vector<Eigen::Matrix<double, N, 1>> m_E;
  };

  //---------------------------------------------------------------------
  //      Direct sum of spaces
  //---------------------------------------------------------------------
  // Construct matrix where each row is an vector component
  template <typename FirstFamilyType, typename SecondFamilyType>
  class SumFamily {
    public:
      typedef typename FirstFamilyType::FunctionValue FunctionValue;
      typedef typename FirstFamilyType::GradientValue GradientValue;
      typedef typename FirstFamilyType::CurlValue CurlValue;
      typedef typename FirstFamilyType::DivergenceValue DivergenceValue;
      typedef typename FirstFamilyType::TraceValue TraceValue;

      typedef typename FirstFamilyType::GeometricSupport GeometricSupport;

      static const TensorRankE tensorRank = FirstFamilyType::tensorRank;
      static const bool hasFunction = (FirstFamilyType::hasFunction && SecondFamilyType::hasFunction);
      static const bool hasGradient = (FirstFamilyType::hasGradient && SecondFamilyType::hasGradient);
      static const bool hasDivergence = (FirstFamilyType::hasDivergence && SecondFamilyType::hasDivergence);
      static const bool hasCurl = (FirstFamilyType::hasCurl && SecondFamilyType::hasCurl);
      static const bool hasTrace = (FirstFamilyType::hasTrace && SecondFamilyType::hasTrace);

      SumFamily(const FirstFamilyType & first_family, const SecondFamilyType & second_family)
        : m_first_family(first_family),
         m_second_family(second_family) 
          {
        static_assert(FirstFamilyType::tensorRank == SecondFamilyType::tensorRank ,
              "SumFamily family can only be constructed from families of same rank");
      }

      // Return the dimension of the family
      inline size_t dimension() const {
          return m_first_family.dimension() + m_second_family.dimension();
      }

      // Evaluate the i-th basis function at point x
      FunctionValue function(size_t i, const VectorRd & x) const {
        static_assert(hasFunction, "Call to function() not available");
        
        return (i < m_first_family.dimension())? m_first_family.function(i,x) : m_second_family.function(i - m_first_family.dimension(),x);
      }

      /// Evaluate the divergence of the i-th basis function at point x
      DivergenceValue divergence(size_t i, const VectorRd & x) const {
        static_assert(hasDivergence, "Call to divergence() not available");

        return (i < m_first_family.dimension())? m_first_family.divergence(i,x) : m_second_family.divergence(i - m_first_family.dimension(),x);
      }
      
      /// Evaluate the divergence of the i-th basis function at point x
      GradientValue gradient(size_t i, const VectorRd & x) const {
        static_assert(hasGradient, "Call to gradient() not available");

        return (i < m_first_family.dimension())? m_first_family.gradient(i,x) : m_second_family.gradient(i - m_first_family.dimension(),x);
      }

      /// Evaluate the divergence of the i-th basis function at point x
      CurlValue curl(size_t i, const VectorRd & x) const {
        static_assert(hasCurl, "Call to curl() not available");

        return (i < m_first_family.dimension())? m_first_family.curl(i,x) : m_second_family.curl(i - m_first_family.dimension(),x);
      }    

      /// Evaluate the trace of the i-th basis function at point x
      TraceValue trace(size_t i, const VectorRd &x) const {
          static_assert(hasTrace, "Call to trace() not available");

          return (i < m_first_family.dimension())? m_first_family.trace(i,x) : m_second_family.trace(i - m_first_family.dimension(),x);
      }

      /// Return the ancestor (family that has been tensorized)
      inline const FirstFamilyType &first_ancestor() const {
        return m_first_family;
      }
      /// Return the ancestor (family that has been tensorized)

      inline const SecondFamilyType &second_ancestor() const {
        return m_second_family;
      }
      
      
    private:
      FirstFamilyType m_first_family;
      SecondFamilyType m_second_family;
  };

  // TODO Add rot to enum BasisFunctionE and specialize evaluate_quad for sum of vector spaces

    /// This overloading of the scalar_product function computes the scalar product between an evaluation of a basis and a constant value; basis values must be a matrix and constant value must be a vector.
  static inline boost::multi_array<VectorRd, 2> scalar_product(
            const boost::multi_array<MatrixRd, 2> &basis_quad, ///< The basis evaluation
            const VectorRd &v                                  ///< The vector to take the scalar product with
            ) {
      boost::multi_array<VectorRd, 2> basis_dot_v_quad(boost::extents[basis_quad.shape()[0]][basis_quad.shape()[1]]);
      std::transform(basis_quad.origin(), basis_quad.origin() + basis_quad.num_elements(),
                     basis_dot_v_quad.origin(), [&v](const MatrixRd &x) -> VectorRd { return x*v; });
      return basis_dot_v_quad;
    }

  template<typename BasisType>
  static boost::multi_array<typename BasisType::TraceValue, 2>
  evaluate_quad_trace_compute(
      const BasisType & basis,    ///< The basis
      const QuadratureRule & quad ///< The quadrature rule
      ) {

    boost::multi_array<typename BasisType::TraceValue, 2>
        basis_quad( boost::extents[basis.dimension()][quad.size()] );
    
    for (size_t i = 0; i < basis.dimension(); i++) {
      for (size_t iqn = 0; iqn < quad.size(); iqn++) {
        basis_quad[i][iqn] = basis.trace(i, quad[iqn].vector());
      } // for iqn
    } // for i

    return basis_quad;
  }

  namespace detail {
    // Evaluate the trace value at x
    template<typename BasisType>
    struct basis_evaluation_traits<BasisType, Trace>
    {
      static_assert(BasisType::hasTrace, "Call to trace not available");
      typedef typename BasisType::TraceValue ReturnValue;
      static inline ReturnValue evaluate(const BasisType & basis, size_t i, const VectorRd & x)
      {
        return basis.trace(i, x);
      }
      
      // Computes trace value at quadrature node iqn, knowing trace of ancestor basis at quadrature nodes
      static inline ReturnValue evaluate(
                    const BasisType &basis, 
                    size_t i, 
                    size_t iqn,
                    const boost::multi_array<ReturnValue, 2> &ancestor_trace_quad
                    )
      {
        return basis.trace(i, iqn, ancestor_trace_quad);
      }
    };
  }

  template<typename BasisType>
  static boost::multi_array<typename BasisType::RotValue, 2>
  evaluate_quad_rot_compute(
      const BasisType & basis,    ///< The basis
      const QuadratureRule & quad ///< The quadrature rule
      ) {

    boost::multi_array<typename BasisType::RotValue, 2>
        basis_quad( boost::extents[basis.dimension()][quad.size()] );
    
    for (size_t i = 0; i < basis.dimension(); i++) {
      for (size_t iqn = 0; iqn < quad.size(); iqn++) {
        basis_quad[i][iqn] = basis.rot(i, quad[iqn].vector());
      } // for iqn
    } // for i

    return basis_quad;
  }

} // end of namespace HArDCore2D
#endif

