// Implementation of the discrete Stokes sequence

#ifndef STOKES_HPP
#define STOKES_HPP

#ifdef WITH_PASTIX
  #include <Eigen/PaStiXSupport>
#elif defined WITH_UMFPACKLU
  #include <Eigen/UmfPackSupport>
#elif defined WITH_MKL
  #define EIGEN_USE_MKL_ALL
  #include <Eigen/PardisoSupport>
#endif

#include <Eigen/Sparse>

#include <stokescore.hpp>
#include <xnabla.hpp>
#include <xvl.hpp>

#include <parallel_for.hpp>

#if __cplusplus > 201703L // std>c++17
#define __LIKELY [[likely]]
#define __UNLIKELY [[unlikely]]
#else
#define __LIKELY
#define __UNLIKELY
#endif

/*!
 * @defgroup Stokes
 * @brief Implementation of the solver for the Stokes problem
 */

namespace HArDCore2D
{

  // Preset for harmonics forms, to add other inplement their cases in assemble_system and setup_Harmonics
  enum Harmonics_premade {
    None,
    Velocity,
    Pressure,
    Custom
  };

  // Forward declaration, compute the location of unknow in the global primitive space (xnabla, xsl, xvl...)
  template<typename Core> size_t L2Gdofs(const Core &core,size_t iT,size_t i); 

  /*!
  * @addtogroup Stokes
  * @{
  */

  class StokesProblem { 
    public:
      typedef Eigen::SparseMatrix<double> SystemMatrixType;
      //typedef Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> SolverType;
      #ifdef ITTERATIVE
      typedef Eigen::BiCGSTAB<SystemMatrixType> SolverType;
      const std::string SolverName = "BiCGSTAB with DiagonalPreconditioner";
      #elif defined ITTERATIVE_LU
      typedef Eigen::BiCGSTAB<SystemMatrixType,Eigen::IncompleteLUT<double>> SolverType;
      const std::string SolverName = "BiCGSTAB with IncompleteLUT";
      #elif defined WITH_PASTIX
      typedef Eigen::PastixLU<SystemMatrixType> SolverType;
      const std::string SolverName = "PastixLU";
      #elif defined WITH_UMFPACKLU
      typedef Eigen::UmfPackLU<SystemMatrixType> SolverType;
      const std::string SolverName = "UmfPackLU";
      #elif defined WITH_MKL
      typedef Eigen::PardisoLU<SystemMatrixType> SolverType;
      const std::string SolverName = "PardisoLU";
      #else
      typedef Eigen::SparseLU<SystemMatrixType,Eigen::COLAMDOrdering<int> > SolverType;
      const std::string SolverName = "SparseLU";
      #endif // ITTERATIVE
      typedef std::function<VectorRd(const VectorRd &)> SourceFunctionType;

      /// Constructor
      StokesProblem(const Mesh &mesh, const size_t degree, bool _use_threads = true, std::ostream & output = std::cout) 
        : use_threads(_use_threads), m_output(output),
          m_stokescore(mesh,degree),
          m_xnabla(m_stokescore),
          m_xvl(m_stokescore),
          m_xsl(m_stokescore),
          DDOFs(Eigen::VectorXi::Zero(m_xnabla.dimension()+m_xsl.dimension())),
          m_Dval(Eigen::VectorXd::Zero(m_xnabla.dimension()+m_xsl.dimension())),
          cells_L2G(mesh.n_cells()),
          cells_L2Gunk(mesh.n_cells()) {
          parallel_for(mesh.n_cells(),m_compute_local2global_dofs,use_threads);
      }
      
      /// Return the dimension of solutions 
      size_t systemEffectiveDim() const 
      {
        return systemTotalDim() + dimH - dimDBC;
      }

      /// Return the dimension of the dofs in the tensor product spaces, excluding harmonics spaces
      size_t systemTotalDim() const 
      {
        return m_xnabla.dimension() + m_xsl.dimension();
      }
      
      /// Set the space of harmonics forms, must be called before assemble_system
      void setup_Harmonics(Harmonics_premade htype);
      /// Assemble the system, compute the RHS at the same time if a source is provided
      void assemble_system(const SourceFunctionType &f = nullptr,size_t degree = 0);
      /// Set rhs vector from function
      void set_rhs (const SourceFunctionType &f, size_t degree = 0);
      /// Set rhs vector from vector 
      void set_rhs (const Eigen::VectorXd &rhs);
      /// Setup the solver
      void compute();
      /// Solve the system and store the solution in the given vector
      Eigen::VectorXd solve();
      /// Solve the system for the given rhs and store the solution in the given vector
      Eigen::VectorXd solve(const Eigen::VectorXd &rhs);
      // Take a vector without dirichlets and return a vector with dirichlet value reinserted (systemEffectiveDim() -> systemTotalDim();
      Eigen::VectorXd reinsertDirichlet(const Eigen::VectorXd &u) const;
      // Set Dirichlet from function, f must return true on Dirichlet boundary
      void set_Dirichlet_boundary(std::function<bool(const VectorRd &)> const & f);

      /// Return the core
      const StokesCore & stokescore() const 
      {
        return m_stokescore;
      }

      /// Return XNabla space
      const XNabla & xnabla() const 
      {
        return m_xnabla;
      }
      
      /// Return XVL space
      const XVL & xvl() const 
      {
        return m_xvl;
      }
      
      /// Return XVL space
      const XSL & xsl() const 
      {
        return m_xsl;
      }

      void setup_Dirichlet_everywhere(); // Set DDOFs to enforce a Dirichlet condition on the whole boundary, automatically call setup_dofsmap()
      void interpolate_boundary_value(XNabla::FunctionType const & f, size_t degree = 0); // setup boundary value by interpolating a function; rhs must be recomputed after any change made to the boundary values
      void setup_dofsmap(); // Call after editing DDOFs to register the changes
      void setup_Dirichlet_values(Eigen::VectorXd const &vals); // Interpolation on xnabla + xsl of the target values; rhs must be recomputed after any change made to the boundary values

      bool use_threads;
    private:
      std::ostream & m_output;
      SystemMatrixType m_system;
      SystemMatrixType m_bdrMatrix;
      Eigen::VectorXd m_rhs;
      SolverType m_solver;
      StokesCore m_stokescore;
      XNabla m_xnabla;
      XVL m_xvl;
      XSL m_xsl;
      Eigen::VectorXi DDOFs; //DDOFs : 1 if Dirichlet dof, 0 otherwise
      Eigen::VectorXd m_Dval;
      size_t dimH = 0;
      Harmonics_premade m_htype = Harmonics_premade::None;
      size_t dimDBC = 0;
      Eigen::VectorXi DDOFs_map; //mapping generated from DDOFs converting dofs including Dirichlets to dofs excluding Dirichlet dofs (and -1 to Dirichlet dofs)
      std::vector<Eigen::VectorXi> cells_L2G; // Contains the mapping of local dof to global index
      std::vector<Eigen::VectorXi> cells_L2Gunk; // Contains the mapping of local dof to global index, with -1 if it correspond to a Dirichlet DOF
      /// Compute the gram_matrix of the integral int_F v, v in Polyk2po on the left
      Eigen::MatrixXd compute_IntXNabla(size_t iT) const;
      /// Compute the gram_matrix of the integral int_F q, q in Polyk on the left
      Eigen::MatrixXd compute_IntXSL(size_t iT) const;
      /// Compute source for the rhs
      Eigen::MatrixXd compute_IntPf(size_t iT, const SourceFunctionType & f, size_t degree) const;
      /// Compute a map of local to global dofs
      std::function<void(size_t,size_t)> m_compute_local2global_dofs = [this](size_t start,size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          Eigen::VectorXi loc = Eigen::VectorXi::Zero(m_xnabla.dimensionCell(iT)+m_xsl.dimensionCell(iT));
          for (size_t i = 0; i < m_xnabla.dimensionCell(iT);i++) {
            loc(i) = L2Gdofs(m_xnabla,iT,i);
          }
          for (size_t i = 0; i < m_xsl.dimensionCell(iT);i++) {
            loc(m_xnabla.dimensionCell(iT) + i) = m_xnabla.dimension() + L2Gdofs(m_xsl,iT,i);
          }
          cells_L2G[iT] = loc;
          cells_L2Gunk[iT] = loc;
        }
      }; // end of std::function

  };


  template<typename Core>
  size_t L2Gdofs(const Core &core,size_t iT,size_t i) {
    Cell &T = *core.mesh().cell(iT);
    size_t n_dofsV = core.numLocalDofsVertex();
    size_t n_V = T.n_vertices();
    size_t n_dofsE = core.numLocalDofsEdge();
    size_t n_E = T.n_edges();
    size_t E_offset = n_V*n_dofsV;
    size_t T_offset = E_offset + n_E*n_dofsE;
    if (i < E_offset) { // i is a Vertex unknown
      size_t Local_Vi = i/n_dofsV;
      size_t Local_Vioffset = i%n_dofsV;
      return core.globalOffset(*T.vertex(Local_Vi)) + Local_Vioffset;
    } else if (i < T_offset) { // i is an Edge unknown
      size_t Local_Ei = (i - E_offset)/n_dofsE;
      size_t Local_Eioffset = (i - E_offset)%n_dofsE;
      return core.globalOffset(*T.edge(Local_Ei)) + Local_Eioffset;
    } else {
      return core.globalOffset(T) + i - T_offset;
    }
  }

  void StokesProblem::setup_dofsmap() {
    // Setup DOFs_map
    DDOFs_map.resize(systemTotalDim());
    size_t acc = 0;
    for (size_t itt = 0; itt < systemTotalDim(); itt++) {
      if (0 == DDOFs(itt)) {
        DDOFs_map(itt) = itt - acc;
      } else { // Dof is on Dirichlet boundary, 
        DDOFs_map(itt) = -1;
        acc++;
      }
    }
    dimDBC = acc;
    // Correct L2Gunk
    for (size_t iT = 0; iT < m_stokescore.mesh().n_cells(); iT++) {
      Eigen::VectorXi loc_L2Gunk = Eigen::VectorXi::Zero(cells_L2G[iT].size());
      for (int i = 0; i < loc_L2Gunk.size(); i++) {
        loc_L2Gunk(i) = DDOFs_map(cells_L2G[iT](i));
      }
      cells_L2Gunk[iT] = loc_L2Gunk;
    }
  }

  void StokesProblem::assemble_system(const SourceFunctionType &f,size_t degree) {
    std::vector<Eigen::MatrixXd> ALoca(m_stokescore.mesh().n_cells());
    std::vector<Eigen::MatrixXd> ALocbp(m_stokescore.mesh().n_cells());
    std::vector<Eigen::MatrixXd> ALocbq(m_stokescore.mesh().n_cells());
    std::vector<Eigen::MatrixXd> ALocuh(m_stokescore.mesh().n_cells());
    std::vector<Eigen::MatrixXd> ALocvh(m_stokescore.mesh().n_cells());
    std::vector<Eigen::VectorXd> RLoc(m_stokescore.mesh().n_cells());
    std::function<void(size_t start, size_t end)> assemble_local = [this,&ALoca,&ALocbp,&ALocbq,&ALocuh,&ALocvh](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Eigen::MatrixXd loca = m_xnabla.cellOperators(iT).ugradient.transpose()*m_xvl.computeL2Product(iT)*m_xnabla.cellOperators(iT).ugradient;
        ALoca[iT] = loca;
        Eigen::MatrixXd locbp = -m_xnabla.cellOperators(iT).divergence.transpose()*m_xsl.computeL2Product(iT);
        ALocbp[iT] = locbp;
        ALocbq[iT] = -locbp.transpose();
        Eigen::MatrixXd locvh;
        switch(m_htype) {
          case(Harmonics_premade::Velocity) :
            locvh = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntXNabla(iT);
            break;
          case(Harmonics_premade::Pressure) :
            locvh = compute_IntXSL(iT);
            break;
          default :
            locvh = Eigen::Matrix<double,0,0>::Zero();
        }
        ALocuh[iT] = locvh.transpose();
        ALocvh[iT] = locvh;
      }
    };
    m_output << "[StokesProblem] Assembling local contributions..."<<std::flush;
    parallel_for(m_stokescore.mesh().n_cells(),assemble_local,use_threads); // Assemble all local contributions
    m_output << "\r[StokesProblem] Assembled local contributions.  ";
    // if a source is given
    const bool f_exists = f != nullptr;
    m_output << ((f_exists)? " Assembling rhs ..." : " Skipping rhs assembly as no source is provided.")<<std::flush;
    if (f_exists) {
      size_t dqr = (degree > 0) ? degree : 2*m_stokescore.degree() + 3;
      std::function<void(size_t start, size_t end)> assemble_local_rhs = [this,&f,dqr,&RLoc](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          Eigen::VectorXd loc = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntPf(iT,f,dqr);
          RLoc[iT] = loc;
        }
      };
      parallel_for(m_stokescore.mesh().n_cells(),assemble_local_rhs,use_threads);
      m_output << "\r[StokesProblem] Assembled local contributions. Assembled rhs.        ";
    } // End if f defined
    m_output << "\n";

    std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs,std::list<Eigen::Triplet<double>> *triplets_bdr)> batch_local_assembly = [this,&f_exists,&ALoca,&ALocbp,&ALocbq,&ALocuh,&ALocvh,&RLoc](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs,std::list<Eigen::Triplet<double>> *triplets_bdr)->void {
      for (size_t iT = start; iT < end; iT++) {
        const Eigen::VectorXi L2Gunk = cells_L2Gunk[iT];
        const size_t offset_xsl_j = m_xnabla.dimensionCell(iT);
        for (size_t i = 0; i < m_xnabla.dimensionCell(iT); i++) { // i in XNabla
          int gi = L2Gunk(i);
          if (gi < 0) continue;
          for (size_t j = 0; j < m_xnabla.dimensionCell(iT); j++) { // i in XNabla, j in XNabla
            int gj = L2Gunk(j);
            if (gj >= 0) {
              triplets->emplace_back(gi,gj,ALoca[iT](i,j));
            } else {
              triplets_bdr->emplace_back(gi,cells_L2G[iT](j),ALoca[iT](i,j));
            }
          }
          for (size_t j = 0; j < m_xsl.dimensionCell(iT); j++) { // i in XNabla, j in XSL
            int gj = L2Gunk(offset_xsl_j + j);
            if (gj >= 0) {
              triplets->emplace_back(gi,gj,ALocbp[iT](i,j));
            } else {
              triplets_bdr->emplace_back(gi,cells_L2G[iT](offset_xsl_j + j),ALocbp[iT](i,j));
            }
          }
          if (m_htype == Harmonics_premade::Velocity) {
            for (size_t j = 0; j < dimH; j++) { // i in XNabla, j in H
              int gj = systemEffectiveDim() - dimH + j;
              triplets->emplace_back(gi,gj,ALocvh[iT](i,j));
            }
          }
          if (f_exists) { // RHS
            (*rhs)(gi) += RLoc[iT](i);
          }
        }
        const size_t offset_i = m_xnabla.dimensionCell(iT);
        for (size_t i = 0; i < m_xsl.dimensionCell(iT); i++) { // i in XSL
          int gi = L2Gunk(offset_i + i);
          if (gi < 0) continue;
          for (size_t j = 0; j < m_xnabla.dimensionCell(iT); j++) { // i in XSL, j in XNabla
            int gj = L2Gunk(j);
            if (gj >= 0) { __LIKELY
              triplets->emplace_back(gi,gj,ALocbq[iT](i,j));
            } else { __UNLIKELY
              triplets_bdr->emplace_back(gi,cells_L2G[iT](j),ALocbq[iT](i,j));
            }
          } // Block xsl x xsl is null
          if (m_htype == Harmonics_premade::Pressure) {
            for (size_t j = 0; j < dimH; j++) { // i in XNabla, j in H
              int gj = systemEffectiveDim() - dimH + j;
              triplets->emplace_back(gi,gj,ALocvh[iT](i,j));
            }
          }
        }
        for (size_t i = 0; i < dimH; i++) { // i in H
          int gi = systemEffectiveDim() - dimH + i;
          if (m_htype == Harmonics_premade::Velocity) {
            for (size_t j = 0; j < m_xnabla.dimensionCell(iT); j++) { // i in H, j in XNabla
              int gj = L2Gunk(j);
              if (gj >= 0) {
                triplets->emplace_back(gi,gj,ALocuh[iT](i,j));
              } else {
                triplets_bdr->emplace_back(gi,cells_L2G[iT](j),ALocuh[iT](i,j));
              }
            }
          } else if (m_htype == Harmonics_premade::Pressure) {
            for (size_t j = 0; j < m_xsl.dimensionCell(iT); j++) { // i in H, j in XSL
              int gj = L2Gunk(offset_xsl_j + j);
              if (gj >= 0) { __LIKELY
                triplets->emplace_back(gi,gj,ALocuh[iT](i,j));
              } else { __UNLIKELY
                triplets_bdr->emplace_back(gi,cells_L2G[iT](offset_xsl_j + j),ALocuh[iT](i,j));
              }
            }
          } 
        } // Other blocks on these rows are null
      }
    };

    m_output << "[StokesProblem] Assembling global system from local contributions..."<<std::flush;
    std::tie(m_system,m_rhs,m_bdrMatrix) = parallel_assembly_system(m_stokescore.mesh().n_cells(),systemEffectiveDim(),std::make_pair(systemEffectiveDim(),systemTotalDim()),batch_local_assembly,use_threads);
    // Incorporate the value Dirichlet dofs
    m_rhs -= m_bdrMatrix*m_Dval;
    m_output << "\r[StokesProblem] Assembled global system                              "<<std::endl;
  }

  void StokesProblem::set_rhs (const SourceFunctionType &f, size_t degree) {
    size_t dqr = (degree > 0) ? degree : 2*m_stokescore.degree() + 3;
    std::vector<Eigen::VectorXd> RLoc(m_stokescore.mesh().n_cells());
    std::function<void(size_t start, size_t end)> assemble_local_rhs = [this,&f,dqr,&RLoc](size_t start, size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Eigen::VectorXd loc = m_xnabla.cellOperators(iT).potential.transpose()*compute_IntPf(iT,f,dqr);
        RLoc[iT] = loc;
      }
    };
    parallel_for(m_stokescore.mesh().n_cells(),assemble_local_rhs,use_threads);

    std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [this,&RLoc](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
      for (size_t iT = start; iT < end; iT++) {
        for (size_t i = 0; i < m_xnabla.dimensionCell(iT); i++) { // i in XNabla
          int gi = cells_L2Gunk[iT](i);
          if (gi >= 0) 
            (*rhs)(gi) += RLoc[iT](i);
        }
      }
    };

    m_rhs = parallel_assembly_system(m_stokescore.mesh().n_cells(),systemEffectiveDim(),batch_local_assembly,use_threads).second;
    // Incorporate the value Dirichlet dofs
    m_rhs -= m_bdrMatrix*m_Dval;
  }

  void StokesProblem::set_rhs (const Eigen::VectorXd &rhs) {
    if (rhs.size() != m_rhs.size()) {
      std::cerr << "[StokesProblem] Setting rhs from vector failed, size dismatched. Expected :"<<m_rhs.size()<<" got :"<<rhs.size()<<std::endl;
      return;
    }
    m_rhs = rhs;
  }
  void StokesProblem::compute() {
    m_output << "[StokesProblem] Setting solver "<<SolverName<<" with "<<systemEffectiveDim()<<" degrees of freedom"<<std::endl;
    m_solver.compute(m_system);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to factorize the system" << std::endl;
      throw std::runtime_error("Factorization failed");
    }
  }
      
  Eigen::VectorXd StokesProblem::solve() {
    Eigen::VectorXd u = m_solver.solve(m_rhs);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solve failed");
    }
    return u;
  }

  Eigen::VectorXd StokesProblem::solve(const Eigen::VectorXd &rhs) {
    Eigen::VectorXd u = m_solver.solve(rhs);
    if (m_solver.info() != Eigen::Success) {
      std::cerr << "[StokesProblem] Failed to solve the system" << std::endl;
      throw std::runtime_error("Solve failed");
    }
    return u;
  }

  Eigen::VectorXd StokesProblem::reinsertDirichlet(const Eigen::VectorXd &u) const {
    assert(u.size() == (int)systemEffectiveDim());
    Eigen::VectorXd rv = Eigen::VectorXd::Zero(systemTotalDim());
    size_t acc = 0;
    for (size_t itt = 0; itt < systemTotalDim(); itt++) {
      if (DDOFs(itt) > 0) { // Dirichlet dof, skip it
        rv(itt) = m_Dval(itt);
        acc++;
      } else {
        rv(itt) = u(itt - acc);
      }
    }
    return rv;
  }

  Eigen::MatrixXd StokesProblem::compute_IntXNabla(size_t iT) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_kpo_T = generate_quadrature_rule(T,m_stokescore.degree() + 1);
    auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk2po,quad_kpo_T);
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(m_stokescore.cellBases(iT).Polyk2po->dimension(),dimspace);
    for (size_t i = 0; i < basis_Pk2po_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_kpo_T.size(); iqn++) {
        rv.row(i) += quad_kpo_T[iqn].w*basis_Pk2po_T_quad[i][iqn];
      }
    }
    return rv;
  }

  // Compute the vector giving the integral of q, q in xsl
  Eigen::MatrixXd StokesProblem::compute_IntXSL(size_t iT) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_k_T = generate_quadrature_rule(T,m_stokescore.degree());
    auto basis_Pk_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk,quad_k_T);
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(m_stokescore.cellBases(iT).Polyk->dimension(),1);
    for (size_t i = 0; i < basis_Pk_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_k_T.size(); iqn++) {
        rv(i,0) += quad_k_T[iqn].w*basis_Pk_T_quad[i][iqn];
      }
    }
    return rv;
  }

  // Return the evaluation of the integral of f against each elements of the basis Polyk2po
  Eigen::MatrixXd StokesProblem::compute_IntPf(size_t iT, const SourceFunctionType & f,size_t degree) const {
    Cell &T = *m_stokescore.mesh().cell(iT);
    QuadratureRule quad_dqr_T = generate_quadrature_rule(T,m_stokescore.degree() + 1 + degree);
    auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*m_stokescore.cellBases(iT).Polyk2po,quad_dqr_T);
    std::vector<VectorRd> intf;
    intf.resize(quad_dqr_T.size()); // store the value of f at each node
    for (size_t iqn = 0; iqn < quad_dqr_T.size(); iqn++) {
      intf[iqn] = f(quad_dqr_T[iqn].vector());
    }
    Eigen::VectorXd rv = Eigen::VectorXd::Zero(m_stokescore.cellBases(iT).Polyk2po->dimension());
    for (size_t i = 0; i < basis_Pk2po_T_quad.shape()[0]; i++) {
      for (size_t iqn = 0; iqn < quad_dqr_T.size(); iqn++) {
        rv(i) += quad_dqr_T[iqn].w*(basis_Pk2po_T_quad[i][iqn]).dot(intf[iqn]);
      }
    }
    return rv;
  }

  void StokesProblem::set_Dirichlet_boundary(std::function<bool(const VectorRd &)> const & f) {
    DDOFs.setZero();
    const Mesh &mesh = m_stokescore.mesh();
    // Itterate over vertices
    for (size_t iV = 0; iV < mesh.n_vertices();iV++) {
      const Vertex &V = *mesh.vertex(iV);
      if (not V.is_boundary() || not f(V.coords())) continue;
      size_t offset = V.global_index()*m_xnabla.numLocalDofsVertex();
      for (size_t i = 0; i < m_xnabla.numLocalDofsVertex();i++) {
        DDOFs(offset + i) = 1;
      }
      offset = m_xnabla.dimension() + V.global_index()*m_xsl.numLocalDofsVertex();
      for (size_t i = 0; i < m_xsl.numLocalDofsVertex();i++) {
        DDOFs(offset + i) = 1;
      }
    }
    // Itterate over edges
    for (size_t iE = 0; iE < mesh.n_edges();iE++) {
      const Edge &E = *mesh.edge(iE);
      if (not E.is_boundary() || not (f(E.vertex(0)->coords()) && f(E.vertex(1)->coords()))) continue;
      size_t offset = mesh.n_vertices()*m_xnabla.numLocalDofsVertex() + E.global_index()*m_xnabla.numLocalDofsEdge();
      for (size_t i = 0; i < m_xnabla.numLocalDofsEdge();i++) {
        DDOFs(offset + i) = 1;
      }
      offset = m_xnabla.dimension() + mesh.n_vertices()*m_xsl.numLocalDofsVertex() + E.global_index()*m_xsl.numLocalDofsEdge();
      for (size_t i = 0; i < m_xsl.numLocalDofsEdge();i++) {
        DDOFs(offset + i) = 1;
      }
    }
    setup_dofsmap();
  }



  void StokesProblem::setup_Dirichlet_everywhere() {
    const Mesh &mesh = m_stokescore.mesh();
    // Itterate over vertices
    for (size_t iV = 0; iV < mesh.n_vertices();iV++) {
      const Vertex &V = *mesh.vertex(iV);
      if (not V.is_boundary()) continue;
      size_t offset = V.global_index()*m_xnabla.numLocalDofsVertex();
      for (size_t i = 0; i < m_xnabla.numLocalDofsVertex();i++) {
        DDOFs(offset + i) = 1;
      }
      offset = m_xnabla.dimension() + V.global_index()*m_xsl.numLocalDofsVertex();
      for (size_t i = 0; i < m_xsl.numLocalDofsVertex();i++) {
        DDOFs(offset + i) = 1;
      }
    }
    // Itterate over edges
    for (size_t iE = 0; iE < mesh.n_edges();iE++) {
      const Edge &E = *mesh.edge(iE);
      if (not E.is_boundary()) continue;
      size_t offset = mesh.n_vertices()*m_xnabla.numLocalDofsVertex() + E.global_index()*m_xnabla.numLocalDofsEdge();
      for (size_t i = 0; i < m_xnabla.numLocalDofsEdge();i++) {
        DDOFs(offset + i) = 1;
      }
      offset = m_xnabla.dimension() + mesh.n_vertices()*m_xsl.numLocalDofsVertex() + E.global_index()*m_xsl.numLocalDofsEdge();
      for (size_t i = 0; i < m_xsl.numLocalDofsEdge();i++) {
        DDOFs(offset + i) = 1;
      }
    }
    setup_dofsmap();
  }

  void StokesProblem::setup_Harmonics(Harmonics_premade htype) {
    m_htype = htype;
    switch(htype) {
      case (Harmonics_premade::None) :
        dimH = 0;
        return;
      case (Harmonics_premade::Velocity) :
        dimH = dimspace;
        return;
      case (Harmonics_premade::Pressure) :
        dimH = 1;
        return;
      default :
        dimH = 0;
        m_output << "[StokesProblem] Warning : harmonics type not yet implemented" << std::endl;
        return;
    }
  }

  void StokesProblem::interpolate_boundary_value(XNabla::FunctionType const & f,size_t degree) {
    size_t dqr = (degree > 0) ? degree : 2*m_xnabla.degree() + 3;
    m_Dval.head(m_xnabla.dimension()) = m_xnabla.interpolate(f,dqr);
  }

  void StokesProblem::setup_Dirichlet_values(Eigen::VectorXd const &vals) {
    assert(m_Dval.size() == vals.size());
    m_Dval = vals;
  }

///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Helper to analyse
///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  double Norm_H1p(const XNabla &xnabla, const XVL &xvl,const Eigen::VectorXd &v,bool use_threads=true) {
    Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&v,&local_sqnorms](size_t start,size_t end)->void {
      for (size_t iT = start;iT < end; iT++) {
        local_sqnorms[iT] = (xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,v)).dot(xvl.computeL2Product(iT)*xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,v));
      }
    };
    parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);

    return std::sqrt(std::abs(local_sqnorms.sum()));
  }

  template<typename Core>
  double Norm_St(const Core &core,const Eigen::VectorXd &v,bool use_threads=true) {
    Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&v,&local_sqnorms](size_t start,size_t end)->void {
      for (size_t iT = start;iT < end; iT++) {
        local_sqnorms[iT] = core.restrictCell(iT,v).dot(core.computeL2Product(iT)*core.restrictCell(iT,v));
      }
    };
    parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

    return std::sqrt(std::abs(local_sqnorms.sum()));
  }

///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Helper to export function to vtu
///-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  // Uniformize the size computation between scalar and VectorRd
  template<typename type> size_t get_sizeof() {
    if constexpr (std::is_same<type,double>::value) { // if constexpr in template with a condition not value-dependent after instantiation then the discarded block is not instantiated. Else we would fail to compile double::SizeAtCompileTime;
      return 1;
    } else {
      return type::SizeAtCompileTime;
    }
  }

  // The first argument of a member function is this* 
  template<typename Core>
  Eigen::VectorXd get_vertices_values(const Core &core, const Eigen::VectorXd &vh) {
    size_t size_rv = get_sizeof<typename std::invoke_result<decltype(&Core::evaluatePotential),Core*,size_t,const Eigen::VectorXd &,const VectorRd &>::type>();
    Eigen::VectorXd vval = Eigen::VectorXd::Zero(size_rv*core.mesh().n_vertices());
    for (size_t i = 0; i < core.mesh().n_vertices();i++) {
      size_t adjcell_id = core.mesh().vertex(i)->cell(0)->global_index();
      vval.segment(i*size_rv,size_rv) << core.evaluatePotential(adjcell_id,core.restrictCell(adjcell_id,vh),core.mesh().vertex(i)->coords());
    }
    return vval;
  }
  template<typename F,typename Core>
  Eigen::VectorXd evaluate_vertices_values(const F &f,const Core &core) {
    size_t size_rv = get_sizeof<typename std::invoke_result<F,const Eigen::VectorXd &>::type>();
    Eigen::VectorXd vval = Eigen::VectorXd::Zero(size_rv*core.mesh().n_vertices());
    for (size_t i = 0; i < core.mesh().n_vertices();i++) {
      vval.segment(i*size_rv,size_rv) << f(core.mesh().vertex(i)->coords());
    }
    return vval;
  }

} // End namespace

#endif
