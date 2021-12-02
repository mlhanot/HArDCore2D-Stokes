
#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xvl.hpp>
#include <xnabla.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

#define FORMATD(W)                                                      \
  ""; formatted_output(std::cout,W+8) << std::setiosflags(std::ios_base::left | std::ios_base::scientific) << std::setprecision(W) << std::setfill(' ')

using namespace HArDCore2D;

const std::vector<std::string> mesh_files = {"../typ2_meshes/" "hexa1_1.typ2",
                                             "../typ2_meshes/" "hexa1_2.typ2",
                                             "../typ2_meshes/" "hexa1_3.typ2",
                                             "../typ2_meshes/" "hexa1_4.typ2"};/*,
                                             "../typ2_meshes/" "hexa1_5.typ2"};*/

constexpr size_t nb_points = 2000;

template<size_t > int validate_potential();

// Something is leaking memory
int main() {
  std::cout << std::endl << "\033[31m[main] Test with degree 0\033[0m" << std::endl; 
  validate_potential<0>();
  std::cout << std::endl << "\033[31m[main] Test with degree 1\033[0m" << std::endl;
  validate_potential<1>();
  std::cout << std::endl << "\033[31m[main] Test with degree 2\033[0m" << std::endl;
  validate_potential<2>();
  std::cout << std::endl << "\033[31m[main] Test with degree 3\033[0m" << std::endl;
  validate_potential<3>();
  return 0;
}

template<typename Core>
double Norm_St(const Core &,const typename Core::FunctionType &,bool use_threads = true);
template<typename Core>
double Norm_St(const Core &,const Eigen::VectorXd &,bool use_threads = true);
template<typename Core>
double computeL2Continuous(const Core &core,const std::function<VectorRd(const VectorRd &)> & f,size_t degree = 0, bool use_threads = true);
template<typename Core>
double computeL2ContinuousBoundaryEdges(const Core &core,const std::function<MatrixRd(const VectorRd &)> & f,size_t degree = 0, bool use_threads = true);
double evaluate_adjNabla(const XNabla &xnabla,const XVL &xvl, const typename XVL::FunctionType &W, const XVL::FunctionDivType &divW, const Eigen::VectorXd &v, const Eigen::VectorXd &Nav, size_t degree = 0, bool use_threads = true);
double evaluate_adjNablaLInf(const XNabla &xnabla,const XVL &xvl, const typename XVL::FunctionType &W, const XVL::FunctionDivType &divW, size_t degree = 0, bool use_threads = true);
double evaluate_adjLaplacian(const XNabla &xnabla,const XVL &xvl, const typename XNabla::FunctionType &w, const XNabla::FunctionType &Lw, const Eigen::VectorXd &v, const Eigen::VectorXd &Nav, size_t degree = 0, bool use_threads = true);
double evaluate_adjLaplacianLInf(const XNabla &xnabla,const XVL &xvl, const typename XNabla::FunctionType &w, const XNabla::FunctionType &Lw, size_t degree = 0, bool use_threads = true);
 
inline double compute_rate(const std::vector<Eigen::VectorXd> &a, const std::vector<double> &h, size_t i, size_t j) {
  if (a[i](j) > 0) {
    return (std::log(a[i](j)) - std::log(a[i-1](j)))/(std::log(h[i]) - std::log(h[i-1]));
  } else {
    return 0.;
  }
}

inline Eigen::VectorXd fill_Nau(const XNabla &xnabla, const XVL &xvl, const Eigen::VectorXd &v) {
  Eigen::VectorXd Nav = Eigen::VectorXd::Zero(xvl.dimension());
  for (size_t iE = 0; iE < xnabla.mesh().n_edges(); iE++) {
    Edge &E = *xnabla.mesh().edge(iE);
    Nav.segment(xvl.globalOffset(E),xvl.numLocalDofsEdge()) = xnabla.edgeOperators(iE).gradient * xnabla.restrictEdge(iE,v);
  }
  for (size_t iT = 0; iT < xnabla.mesh().n_cells(); iT++) {
    Cell &T = *xnabla.mesh().cell(iT);
    Nav.segment(xvl.globalOffset(T),xvl.numLocalDofsCell()) = xnabla.cellOperators(iT).gradient * xnabla.restrictCell(iT,v);
  }
  return Nav;
}

template<typename T, typename T2>
XNabla::FunctionGradType FormalGrad(T &vx, T2 &vy) {
  return [&vx,&vy](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv(0,0) = vx.evaluate(x,1,0);
    rv(0,1) = vx.evaluate(x,0,1);
    rv(1,0) = vy.evaluate(x,1,0);
    rv(1,1) = vy.evaluate(x,0,1);
    return rv;};
}

template<typename T, typename T2>
XNabla::FunctionType FormalLaplacian(T &vx, T2 &vy) {
  return [&vx,&vy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = vx.evaluate(x,2,0) + vx.evaluate(x,0,2);
    rv(1) = vy.evaluate(x,2,0) + vy.evaluate(x,0,2);
    return rv;};
}

template<typename T, typename T2>
XVL::FunctionDivType FormalDiv(T &vxx, T2 &vxy, T &vyx, T2&vyy) {
  return [&vxx,&vxy,&vyx,&vyy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = vxx.evaluate(x,1,0) + vxy.evaluate(x,0,1);
    rv(1) = vyx.evaluate(x,1,0) + vyy.evaluate(x,0,1);
    return rv;};
}

template<size_t degree>
int validate_potential() {
  // Create test functions Nabla
  PolyTest<degree> m_Pkx(Initialization::Random);
  PolyTest<degree> m_Pky(Initialization::Random);
  PolyTest<degree + 3> m_Pkxp3(Initialization::Random);
  PolyTest<degree + 3> m_Pkyp3(Initialization::Random);
  TrigTest<degree> m_Ttrigx(Initialization::Random);
  TrigTest<degree> m_Ttrigy(Initialization::Random);
  /*
  ZerodXWrapper<PolyTest<degree>,degree> Pkx(&m_Pkx);
  ZerodYWrapper<PolyTest<degree>,degree> Pky(&m_Pky);
  XNabla::FunctionType Pk = [&Pkx,&Pky](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkx.evaluate(x), Pky.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType dPk = FormalGrad(Pkx,Pky);
  XNabla::FunctionType LPk = FormalLaplacian(Pkx,Pky);
  
  XNabla::FunctionType NTtrig = [&m_Ttrigx,&m_Ttrigy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << m_Ttrigx.evaluate(x), m_Ttrigy.evaluate(x);
    return rv;
  };
  */
  // Function with Nabla . n = 0
  TrigTestdZero<degree> m_Tzx0(Initialization::Random);
  TrigTestdZero<degree> m_Tzy0(Initialization::Random);
  TrigTestdZero<degree> m_Tzx1(Initialization::Random);
  TrigTestdZero<degree> m_Tzy1(Initialization::Random);
  TrigTestdZero<degree> m_Tzx2(Initialization::Random);
  TrigTestdZero<degree> m_Tzy2(Initialization::Random);
  XNabla::FunctionType Tz0 = [&m_Tzx0,&m_Tzy0](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << m_Tzx0.evaluate(x), m_Tzy0.evaluate(x);
    return rv;
  };
  XNabla::FunctionType LTz0 = FormalLaplacian(m_Tzx0,m_Tzy0);
  XNabla::FunctionType Tz1 = [&m_Tzx1,&m_Tzy1](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << m_Tzx1.evaluate(x), m_Tzy1.evaluate(x);
    return rv;
  };
  XNabla::FunctionType LTz1 = FormalLaplacian(m_Tzx1,m_Tzy1);
  XNabla::FunctionType Tz2 = [&m_Tzx2,&m_Tzy2](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << m_Tzx2.evaluate(x), m_Tzy2.evaluate(x);
    return rv;
  };
  XNabla::FunctionType LTz2 = FormalLaplacian(m_Tzx2,m_Tzy2);

  // Create test functions XVL
  PolyTest<degree> m_Pkx2(Initialization::Random);
  PolyTest<degree> m_Pky2(Initialization::Random);
  PolyTest<degree + 3> m_Pkx2p3(Initialization::Random);
  PolyTest<degree + 3> m_Pky2p3(Initialization::Random);
  TrigTest<degree> m_Ttrigx2(Initialization::Random);
  TrigTest<degree> m_Ttrigy2(Initialization::Random);

  ZeroXWrapper<PolyTest<degree>,degree> Pkxx(&m_Pkx);
  ZeroXWrapper<PolyTest<degree>,degree> Pkyx(&m_Pkx2);
  ZeroYWrapper<PolyTest<degree>,degree> Pkxy(&m_Pky);
  ZeroYWrapper<PolyTest<degree>,degree> Pkyy(&m_Pky2);
  XVL::FunctionType P2k = [&Pkxx,&Pkxy,&Pkyx,&Pkyy](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkxx.evaluate(x), Pkxy.evaluate(x),Pkyx.evaluate(x), Pkyy.evaluate(x);
    return rv;
  };
  XVL::FunctionDivType divP2k = FormalDiv(Pkxx,Pkxy,Pkyx,Pkyy);

  ZeroXWrapper<PolyTest<degree+3>,degree+3> Pkp3xx(&m_Pkxp3);
  ZeroXWrapper<PolyTest<degree+3>,degree+3> Pkp3yx(&m_Pkx2p3);
  ZeroYWrapper<PolyTest<degree+3>,degree+3> Pkp3xy(&m_Pkyp3);
  ZeroYWrapper<PolyTest<degree+3>,degree+3> Pkp3yy(&m_Pky2p3);
  XVL::FunctionType P2kp3 = [&Pkp3xx,&Pkp3xy,&Pkp3yx,&Pkp3yy](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkp3xx.evaluate(x), Pkp3xy.evaluate(x),Pkp3yx.evaluate(x), Pkp3yy.evaluate(x);
    return rv;
  };
  XVL::FunctionDivType divP2kp3 = FormalDiv(Pkp3xx,Pkp3xy,Pkp3yx,Pkp3yy);

  ZeroXWrapper<TrigTest<degree>,degree> Ttrigxx(&m_Ttrigx);
  ZeroXWrapper<TrigTest<degree>,degree> Ttrigyx(&m_Ttrigy);
  ZeroYWrapper<TrigTest<degree>,degree> Ttrigxy(&m_Ttrigx2);
  ZeroYWrapper<TrigTest<degree>,degree> Ttrigyy(&m_Ttrigy2);
  XVL::FunctionType Ttrig = [&Ttrigxx,&Ttrigxy,&Ttrigyx,&Ttrigyy](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Ttrigxx.evaluate(x), Ttrigxy.evaluate(x),Ttrigyx.evaluate(x), Ttrigyy.evaluate(x);
    return rv;
  };
  XVL::FunctionDivType divTtrig = FormalDiv(Ttrigxx,Ttrigxy,Ttrigyx,Ttrigyy);


  // Storage for resulting value
  std::vector<double> meshsize;
  std::vector<Eigen::VectorXd> normEpsNa;
  std::vector<Eigen::VectorXd> normEpsL;
  std::vector<Eigen::VectorXd> normEpsNaNormalized;
  std::vector<Eigen::VectorXd> normEpsLNormalized;
  std::vector<double> normv;
  std::vector<double> normNav;

  // Iterate over meshes
  for (auto mesh_file : mesh_files) {
    // Build the mesh
    MeshBuilder builder = MeshBuilder(mesh_file);
    std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
    std::cout << FORMAT(25) << "[main] Mesh size" << mesh_ptr->h_max() << std::endl;
    // Store the size of the mesh
    meshsize.emplace_back(mesh_ptr->h_max());

    // Create core 
    StokesCore stokes_core(*mesh_ptr,degree);
    std::cout << "[main] StokesCore constructed" << std::endl;

    // Create discrete space XNabla
    XNabla xnabla(stokes_core);
    std::cout << "[main] XNabla constructed" << std::endl;

    // Create discrete space XVL
    XVL xvl(stokes_core);
    std::cout << "[main] XVL constructed" << std::endl;

    // TestFunction v
    //Eigen::VectorXd v = Eigen::VectorXd::Zero(xnabla.dimension()); fill_random_vector(v);
    /*
    Eigen::VectorXd v = xnabla.interpolate(divTtrig);
    Eigen::VectorXd Nav = fill_Nau(xnabla,xvl,v);

    normv.emplace_back(Norm_St(xnabla,v));
    normNav.emplace_back(Norm_St(xvl,Nav));
    std::cout << "[main] Norm H1 v computed" << std::endl;
    */

    Eigen::VectorXd norm = Eigen::VectorXd::Zero(3); // number of functions
    /* 
    norm(0) = evaluate_adjNabla(xnabla,xvl,P2k,divP2k,v,Nav);
    norm(1) = evaluate_adjNabla(xnabla,xvl,P2kp3,divP2kp3,v,Nav);
    norm(2) = evaluate_adjNabla(xnabla,xvl,Ttrig,divTtrig,v,Nav);
    */
    norm(0) = evaluate_adjNablaLInf(xnabla,xvl,P2k,divP2k);
    norm(1) = evaluate_adjNablaLInf(xnabla,xvl,P2kp3,divP2kp3);
    norm(2) = evaluate_adjNablaLInf(xnabla,xvl,Ttrig,divTtrig);
    normEpsNa.emplace_back(norm);
    //norm = norm/(normv.back() + normNav.back());
    normEpsNaNormalized.emplace_back(norm);
    std::cout << "[main] Epsilon Nabla computed" << std::endl;
    /*
    norm(0) = evaluate_adjLaplacian(xnabla,xvl,Tz0,LTz0,v,Nav);
    norm(1) = evaluate_adjLaplacian(xnabla,xvl,Tz1,LTz1,v,Nav);
    norm(2) = evaluate_adjLaplacian(xnabla,xvl,Tz2,LTz2,v,Nav);
    */
    norm(0) = evaluate_adjLaplacianLInf(xnabla,xvl,Tz0,LTz0);
    norm(1) = evaluate_adjLaplacianLInf(xnabla,xvl,Tz1,LTz1);
    norm(2) = evaluate_adjLaplacianLInf(xnabla,xvl,Tz2,LTz2);
    normEpsL.emplace_back(norm);
    //norm = norm/(normv.back() + normNav.back());
    normEpsLNormalized.emplace_back(norm);
    std::cout << "[main] Epsilon L computed" << std::endl;
  } // end for mesh_files

  for (int j = 0; j < normEpsL[0].size(); j++) {
    std::cout << "AbsEpsNa  AbsEpsL" << std::endl;
    for (size_t i = 0; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<normEpsNa[i](j)<<normEpsL[i](j)<<std::endl;}
    std::cout << "RateEpsNa  RateEpsL" << std::endl;
    for (size_t i = 1; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<compute_rate(normEpsNaNormalized,meshsize,i,j)<<compute_rate(normEpsLNormalized,meshsize,i,j)<<std::endl;}
  }
  
  return 0;
}

template<typename Core>
double computeL2Continuous(const Core &core,const std::function<VectorRd(const VectorRd &)> & f,size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  degree = (degree > 0)? degree : 2*(core.degree() + 3);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&f,&degree,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      Cell & T = *core.mesh().cell(iT);
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      
      double rv = 0.;
      for (auto node : quad_dqr_T) {
        rv += node.w*f(node.vector()).dot(f(node.vector()));
      }
      local_sqnorms[iT] = rv;
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}
template<typename Core>
double computeL2ContinuousBoundaryEdges(const Core &core,const std::function<MatrixRd(const VectorRd &)> & f,size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_edges());
  degree = (degree > 0)? degree : 2*(core.degree() + 3);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&f,&degree,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iE = start;iE < end; iE++) {
      Edge & E = *core.mesh().edge(iE);
      double rv = 0.;
      if (E.is_boundary()) {
        QuadratureRule quad_dqr_E = generate_quadrature_rule(E, degree);
        // Get outgoing vector
        assert(E.n_cells() == 1);
        Cell &T = *E.cell(0);
        VectorRd outn = T.edge_normal(T.index_edge(&E));
      
        for (auto node : quad_dqr_E) {
          rv += node.w*(f(node.vector())*outn).dot(f(node.vector())*outn);
        }
      }
      local_sqnorms[iE] = rv;
    }
  };
  parallel_for(core.mesh().n_edges(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double evaluate_adjNabla(const XNabla &xnabla,const XVL &xvl, const typename XVL::FunctionType &W, const XVL::FunctionDivType &divW, const Eigen::VectorXd &v, const Eigen::VectorXd &Nav, size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  degree = (degree > 0)? degree : 2*(xnabla.degree() + 3);
  
  Eigen::VectorXd IW = xvl.interpolate(W);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&divW,&IW,&v,&Nav,&degree,&local_sqnorms](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Cell &T = *xnabla.mesh().cell(iT);
      double rv = 0;
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      // (IW,Nav)L2
      rv += xvl.restrictCell(iT,IW).dot(xvl.computeL2Product(iT)*xvl.restrictCell(iT,Nav));
      // Int div W, Pv
      for (auto node : quad_dqr_T) {
        rv += node.w*divW(node.vector()).dot(xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,v),node.vector()));
      }
      local_sqnorms[iT] = rv;
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::abs(local_sqnorms.sum());
}

double evaluate_adjLaplacian(const XNabla &xnabla,const XVL &xvl, const typename XNabla::FunctionType &w, const XNabla::FunctionType &Lw, const Eigen::VectorXd &v, const Eigen::VectorXd &Nav, size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  degree = (degree > 0)? degree : 2*(xnabla.degree() + 3);
  
  Eigen::VectorXd Iw = xnabla.interpolate(w);
  Eigen::VectorXd NaIw = fill_Nau(xnabla,xvl,Iw);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&Lw,&NaIw,&v,&degree,&local_sqnorms](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Cell &T = *xnabla.mesh().cell(iT);
      double rv = 0;
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      // (NaIw,Nav)L2
      rv += xvl.restrictCell(iT,NaIw).dot(xvl.computeL2Product(iT)*xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,v));
      // Int Lw, Pv
      for (auto node : quad_dqr_T) {
        rv += node.w*Lw(node.vector()).dot(xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,v),node.vector()));
      }
      local_sqnorms[iT] = rv;
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::abs(local_sqnorms.sum());
}

double Norm_H1(const XNabla &xnabla, const XVL &xvl,const Eigen::VectorXd &v,bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&v,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      local_sqnorms[iT] = (xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,v)).dot(xvl.computeL2Product(iT)*xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,v));
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}

double evaluate_adjNablaLInf(const XNabla &xnabla,const XVL &xvl, const typename XVL::FunctionType &W, const XVL::FunctionDivType &divW, size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  degree = (degree > 0)? degree : 2*xnabla.degree() + 3;
  
  Eigen::VectorXd IW = xvl.interpolate(W);
  Eigen::VectorXd Gv = Eigen::VectorXd::Zero(xnabla.dimension());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&IW,&divW,&Gv,&degree,&local_sqnorms](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Cell &T = *xnabla.mesh().cell(iT);
      double rv = 0.;
      Eigen::VectorXd v = xnabla.restrictCell(iT,Gv);
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      // (Iw,Nav)L2
      rv = (xvl.restrictCell(iT,IW)).dot(xvl.computeL2Product(iT)*xnabla.cellOperators(iT).ugradient*v);
      // Int div w, Pv
      for (auto node : quad_dqr_T) {
        rv += node.w*divW(node.vector()).dot(xnabla.evaluatePotential(iT,v,node.vector()));
      }
      local_sqnorms[iT] = rv;
    }
  };
  //double DB_max = 0.; double DB_min = 100.;
  size_t nb_points_loc = (xnabla.dimension() > nb_points) ? nb_points : xnabla.dimension();
  Eigen::VectorXd localval = Eigen::VectorXd::Zero(nb_points_loc);
  for (size_t i = 0; i < nb_points_loc;i += xnabla.dimension()/nb_points_loc) { 
    Gv(i) = 1.;
    parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);
    double Nv = (Norm_St(xnabla,Gv) + Norm_H1(xnabla,xvl,Gv,true));
    Gv(i) = 0.;
    localval(i) = std::abs(local_sqnorms.sum())/Nv;
    //DB_max = (DB_max > Nv)? DB_max : Nv; DB_min = (DB_min < Nv)? DB_min : Nv;
  }
  //std::cout<<"[evaluate_adjNablaLInf] size : "<<localval.size()<<" Min :"<<DB_min<<" Max : "<<DB_max<<std::endl;

  return localval.maxCoeff();
}

double evaluate_adjLaplacianLInf(const XNabla &xnabla,const XVL &xvl, const typename XNabla::FunctionType &w, const XNabla::FunctionType &Lw, size_t degree, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  degree = (degree > 0)? degree : 2*xnabla.degree() + 3;
  
  Eigen::VectorXd Iw = xnabla.interpolate(w);
  Eigen::VectorXd Gv = Eigen::VectorXd::Zero(xnabla.dimension());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&xnabla,&xvl,&Iw,&Lw,&Gv,&degree,&local_sqnorms](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Cell &T = *xnabla.mesh().cell(iT);
      double rv = 0.;
      Eigen::VectorXd v = xnabla.restrictCell(iT,Gv);
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      // (NaIw,Nav)L2
      rv = (xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,Iw)).dot(xvl.computeL2Product(iT)*xnabla.cellOperators(iT).ugradient*v);
      // Int Lw, Pv
      for (auto node : quad_dqr_T) {
        rv += node.w*Lw(node.vector()).dot(xnabla.evaluatePotential(iT,v,node.vector()));
      }
      local_sqnorms[iT] = rv;
    }
  };
  //double DB_max = 0.; double DB_min = 100.;
  size_t nb_points_loc = (xnabla.dimension() > nb_points) ? nb_points : xnabla.dimension();
  Eigen::VectorXd localval = Eigen::VectorXd::Zero(nb_points_loc);
  for (size_t i = 0; i < nb_points_loc;i += xnabla.dimension()/nb_points_loc) { 
    Gv(i) = 1.;
    parallel_for(xnabla.mesh().n_cells(),compute_local_squarednorms,use_threads);
    double Nv = (Norm_St(xnabla,Gv) + Norm_H1(xnabla,xvl,Gv,true));
    Gv(i) = 0.;
    localval(i) = std::abs(local_sqnorms.sum())/Nv;
    //DB_max = (DB_max > Nv)? DB_max : Nv; DB_min = (DB_min < Nv)? DB_min : Nv;
  }
  //std::cout<<"[evaluate_adjLaplacianLInf] size : "<<localval.size()<<" Min :"<<DB_min<<" Max : "<<DB_max<<std::endl;

  return localval.maxCoeff();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename Core>
double Norm_St(const Core &core,const typename Core::FunctionType &v,bool use_threads) {
  // Dimension of potentials dofs
  size_t PDim = 0;
  if (std::is_same<Core,XNabla>::value) {PDim = core.cellBases(0).Polyk2po->dimension();}
  else if (std::is_same<Core,XVL>::value) {PDim = core.cellBases(0).RTbkpo->dimension();}

  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  Eigen::VectorXd Iv = core.interpolate(v);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&Iv,&PDim,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      local_sqnorms[iT] = core.restrictCell(iT,Iv).dot(core.computeL2Product(iT)*core.restrictCell(iT,Iv));
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}

template<typename Core>
double Norm_St(const Core &core,const Eigen::VectorXd &v,bool use_threads) {
  // Dimension of potentials dofs
  size_t PDim = 0;
  if (std::is_same<Core,XNabla>::value) {PDim = core.cellBases(0).Polyk2po->dimension();}
  else if (std::is_same<Core,XVL>::value) {PDim = core.cellBases(0).RTbkpo->dimension();}

  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&v,&PDim,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      local_sqnorms[iT] = core.restrictCell(iT,v).dot(core.computeL2Product(iT)*core.restrictCell(iT,v));
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}
