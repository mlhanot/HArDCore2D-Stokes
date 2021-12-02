
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
                                             "../typ2_meshes/" "hexa1_4.typ2",
                                             "../typ2_meshes/" "hexa1_5.typ2"};

template<size_t > int validate_potential();

constexpr double scale = 1e0;

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

double Norm_Potential(const XNabla &xnabla,const XNabla::FunctionType &,bool use_threads = true);
template<typename Core>
double Norm_St(const Core &xnabla,const typename Core::FunctionType &,bool use_threads = true);
 
// Number below 1e-7 (below 1e-14 before the squareroot) suffer from double precision inacuracy (evaluating a**2 - 2*a*b + b**2 instead of (a - b)**2).
inline double compute_rate(const std::vector<Eigen::VectorXd> &a, const std::vector<double> &h, size_t i, size_t j) {
  if (a[i](j) > 1e-7) {
    return (std::log(a[i](j)) - std::log(a[i-1](j)))/(std::log(h[i]) - std::log(h[i-1]));
  } else {
    return 0.;
  }
}

template<size_t degree>
int validate_potential() {
  // Create test functions Nabla
  PolyTest<degree> Pkx(Initialization::Random,scale);
  PolyTest<degree> Pky(Initialization::Random,scale);
  PolyTest<degree + 1> Pkpox(Initialization::Random,scale);
  PolyTest<degree + 1> Pkpoy(Initialization::Random,scale);
  PolyTest<degree + 2> Pkp2x(Initialization::Random,scale);
  PolyTest<degree + 2> Pkp2y(Initialization::Random,scale);
  PolyTest<degree + 3> Pkp3x(Initialization::Random,scale);
  PolyTest<degree + 3> Pkp3y(Initialization::Random,scale);
  TrigTest<degree> Ttrigx(Initialization::Random,scale);
  TrigTest<degree> Ttrigy(Initialization::Random,scale);
  XNabla::FunctionType Pk = [&Pkx,&Pky](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkx.evaluate(x), Pky.evaluate(x);
    return rv;
  };
  XNabla::FunctionType Pkpo = [&Pkpox,&Pkpoy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkpox.evaluate(x), Pkpoy.evaluate(x);
    return rv;
  };
  XNabla::FunctionType Pkp2 = [&Pkp2x,&Pkp2y](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkp2x.evaluate(x), Pkp2y.evaluate(x);
    return rv;
  };
  XNabla::FunctionType Pkp3 = [&Pkp3x,&Pkp3y](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkp3x.evaluate(x), Pkp3y.evaluate(x);
    return rv;
  };
  XNabla::FunctionType Ttrig = [&Ttrigx,&Ttrigy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Ttrigx.evaluate(x), Ttrigy.evaluate(x);
    return rv;
  };

  // Create test functions XVL
  PolyTest<degree> Pkx2(Initialization::Random,scale);
  PolyTest<degree> Pky2(Initialization::Random,scale);
  PolyTest<degree + 1> Pkpox2(Initialization::Random,scale);
  PolyTest<degree + 1> Pkpoy2(Initialization::Random,scale);
  PolyTest<degree + 2> Pkp2x2(Initialization::Random,scale);
  PolyTest<degree + 2> Pkp2y2(Initialization::Random,scale);
  PolyTest<degree + 3> Pkp3x2(Initialization::Random,scale);
  PolyTest<degree + 3> Pkp3y2(Initialization::Random,scale);
  TrigTest<degree> Ttrigx2(Initialization::Random,scale);
  TrigTest<degree> Ttrigy2(Initialization::Random,scale);
  XVL::FunctionType P2k = [&Pkx,&Pky,&Pkx2,&Pky2](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkx.evaluate(x), Pky.evaluate(x),Pkx2.evaluate(x), Pky2.evaluate(x);
    return rv;
  };
  XVL::FunctionType P2kpo = [&Pkpox,&Pkpoy,&Pkpox2,&Pkpoy2](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkpox.evaluate(x), Pkpoy.evaluate(x),Pkpox2.evaluate(x), Pkpoy2.evaluate(x);
    return rv;
  };
  XVL::FunctionType P2kp2 = [&Pkp2x,&Pkp2y,&Pkp2x2,&Pkp2y2](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkp2x.evaluate(x), Pkp2y.evaluate(x),Pkp2x2.evaluate(x), Pkp2y2.evaluate(x);
    return rv;
  };
  XVL::FunctionType P2kp3 = [&Pkp3x,&Pkp3y,&Pkp3x2,&Pkp3y2](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Pkp3x.evaluate(x), Pkp3y.evaluate(x),Pkp3x2.evaluate(x), Pkp3y2.evaluate(x);
    return rv;
  };
  XVL::FunctionType T2trig = [&Ttrigx,&Ttrigy,&Ttrigx2,&Ttrigy2](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv << Ttrigx.evaluate(x), Ttrigy.evaluate(x),Ttrigx2.evaluate(x), Ttrigy2.evaluate(x);
    return rv;
  };

  // Storage for resulting value
  std::vector<double> meshsize;
  std::vector<Eigen::VectorXd> normPNa;
  std::vector<Eigen::VectorXd> normSNa;
  std::vector<Eigen::VectorXd> normSL2;

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
    
    Eigen::VectorXd norm = Eigen::VectorXd::Zero(5); // number of functions
    
    norm(0) = Norm_Potential(xnabla,Pk);
    norm(1) = Norm_Potential(xnabla,Pkpo);
    norm(2) = Norm_Potential(xnabla,Pkp2);
    norm(3) = Norm_Potential(xnabla,Pkp3);
    norm(4) = Norm_Potential(xnabla,Ttrig);
    normPNa.emplace_back(norm);
    std::cout << "[main] Norm potential on XNabla computed" << std::endl;
    
    norm(0) = Norm_St(xnabla,Pk);
    norm(1) = Norm_St(xnabla,Pkpo);
    norm(2) = Norm_St(xnabla,Pkp2);
    norm(3) = Norm_St(xnabla,Pkp3);
    norm(4) = Norm_St(xnabla,Ttrig);
    normSNa.emplace_back(norm);
    std::cout << "[main] Norm stabilization on XNabla computed" << std::endl;
    
    norm(0) = Norm_St(xvl,P2k);
    norm(1) = Norm_St(xvl,P2kpo);
    norm(2) = Norm_St(xvl,P2kp2);
    norm(3) = Norm_St(xvl,P2kp3);
    norm(4) = Norm_St(xvl,T2trig);
    normSL2.emplace_back(norm);
    std::cout << "[main] Norm stabilization on XVL computed" << std::endl;
  } // end for mesh_files

  for (int j = 0; j < normPNa[0].size(); j++) {
    std::cout << "AbsPNa   AbsSNa   AbsSL2" << std::endl;
    for (size_t i = 0; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<normPNa[i](j)<<normSNa[i](j)<<normSL2[i](j)<<std::endl;}
    std::cout << "RatePNa  RateSNa  RateSL2" << std::endl;
    for (size_t i = 1; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<compute_rate(normPNa,meshsize,i,j)<<compute_rate(normSNa,meshsize,i,j)<<compute_rate(normSL2,meshsize,i,j)<<std::endl;}
  }
  
  return 0;
}

template<typename Core>
double computeL2Continuous(const Core &core,const std::function<double(const VectorRd &, size_t)> & f,size_t degree = 0, bool use_threads = true) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  degree = (degree > 0)? degree : 2*(core.degree() + 3);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&f,&degree,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      Cell & T = *core.mesh().cell(iT);
      QuadratureRule quad_dqr_T = generate_quadrature_rule(T, degree);
      
      double rv = 0.;
      for (auto node : quad_dqr_T) {
        rv += node.w*f(node.vector(), iT);
      }
      local_sqnorms[iT] = rv;
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  //return std::sqrt(std::abs(local_sqnorms.sum()));
  return std::sqrt(local_sqnorms.cwiseAbs().maxCoeff());
}

template<typename Core, typename ValueType>
double computeL2Err(const Core &core,const std::function<ValueType(const VectorRd &, size_t iT)>  &f1, const std::function<ValueType(const VectorRd &, size_t iT)> &f2,size_t degree = 0, bool use_threads = true) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT)).cwiseProduct(f1(x,iT) - f2(x,iT)).sum();
  };
  return computeL2Continuous(core,f,degree,use_threads);
}
template<typename Core>
double computeL2Err(const Core &core,const std::function<double(const VectorRd &, size_t iT)>  &f1, const std::function<double(const VectorRd &, size_t iT)> &f2,size_t degree = 0, bool use_threads = true) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT))*(f1(x,iT) - f2(x,iT));
  };
  return computeL2Continuous(core,f,degree,use_threads);
}

double Norm_Potential(const XNabla &xnabla,const XNabla::FunctionType &v,bool use_threads) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v);
  
  // PIv to local function
  std::function<VectorRd(const VectorRd &, size_t)> pIv = [&xnabla, &Iv](const VectorRd &x,size_t iT)->VectorRd {
    return xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,Iv),x);
  };

  // v to local function
  std::function<VectorRd(const VectorRd &, size_t)> vL = [&v](const VectorRd &x,size_t iT)->VectorRd {
    return v(x);
  };

  return computeL2Err(xnabla,pIv, vL,2*(xnabla.degree() + 4),use_threads);
}

template<typename Core>
double Norm_St(const Core &core,const typename Core::FunctionType &v,bool use_threads) {
  // Dimension of potentials dofs
  size_t PDim = 0;
  if (std::is_same<Core,XNabla>::value) {PDim = core.cellBases(0).Polyk2po->dimension();}
  else if (std::is_same<Core,XVL>::value) {PDim = core.cellBases(0).RTbkpo->dimension();}

  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  Eigen::VectorXd Iv = core.interpolate(v);//,2*core.degree() + 9);
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&Iv,&PDim,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      local_sqnorms[iT] = core.restrictCell(iT,Iv).dot(core.computeL2Product(iT,1.,Eigen::MatrixXd::Zero(PDim,PDim))*core.restrictCell(iT,Iv));
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  //std::cout<<"Max  Min  Mean"<<std::endl;
  //std::cout<<FORMATD(2)<<local_sqnorms.maxCoeff()<<local_sqnorms.minCoeff()<<local_sqnorms.mean()<<std::endl;

  //return std::sqrt(std::abs(local_sqnorms.sum()));
  return std::sqrt(local_sqnorms.cwiseAbs().maxCoeff());
}

    /* Test limitation of precision
    //---------------------------------------------------------------------
    Eigen::VectorXd local_sqnorms_E = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
    Eigen::VectorXd Iv = xnabla.interpolate(Pk);
    int locp,locm;
    std::function<void(size_t,size_t)> compute_local_err_E0 = [&xnabla,&Iv,&Pk,&local_sqnorms_E](size_t start,size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Cell &T = *xnabla.mesh().cell(iT);
        double rv = 0.;
        for (size_t iEl = 0; iEl < T.n_edges();iEl++) {
          Edge &E = *T.edge(iEl);
          size_t iE = E.global_index();
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E,2*xnabla.degree() + 4);
          for (auto node : quad_dqr_E) {
            rv += node.w*(Pk(node.vector()) - xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector())).dot(Pk(node.vector()) - xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector()));
          }
        }
        local_sqnorms_E[iT] = rv;
      }
    };
    parallel_for(xnabla.mesh().n_cells(),compute_local_err_E0,false);
    std::cout<<"Max Min Mean"<<std::endl;
    std::cout<<FORMATD(2)<<local_sqnorms_E.maxCoeff(&locp)<<local_sqnorms_E.minCoeff(&locm)<<local_sqnorms_E.mean()<<std::endl;
    std::cout<<"Max loc :"<<locp<<" Min loc :"<<locm<<std::endl;

    std::function<void(size_t,size_t)> compute_local_err_E = [&xnabla,&Iv,&Pk,&local_sqnorms_E](size_t start,size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Cell &T = *xnabla.mesh().cell(iT);
        double rv = 0.;
        for (size_t iEl = 0; iEl < T.n_edges();iEl++) {
          Edge &E = *T.edge(iEl);
          size_t iE = E.global_index();
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E,2*xnabla.degree() + 4);
          for (auto node : quad_dqr_E) {
            //rv += E.measure()*node.w*xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,Iv),node.vector()).dot(xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,Iv),node.vector()));
            rv += E.measure()*node.w*xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector()).dot(xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector()));
            //rv -= E.measure()*node.w*xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,Iv),node.vector()).dot(Pk(node.vector()));
            rv -= E.measure()*node.w*xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector()).dot(Pk(node.vector()));
            rv += E.measure()*node.w*Pk(node.vector()).dot(Pk(node.vector()));
            //rv -= E.measure()*node.w*Pk(node.vector()).dot(xnabla.evaluatePotential(iT,xnabla.restrictCell(iT,Iv),node.vector()));
            rv -= E.measure()*node.w*Pk(node.vector()).dot(xnabla.evaluatePotential_Edge(iE,xnabla.restrictEdge(iE,Iv),node.vector()));
          }
        }
        local_sqnorms_E[iT] = rv;
      }
    };
    parallel_for(xnabla.mesh().n_cells(),compute_local_err_E,false);
    std::cout<<"Max Min Mean"<<std::endl;
    std::cout<<FORMATD(2)<<local_sqnorms_E.maxCoeff(&locp)<<local_sqnorms_E.minCoeff(&locm)<<local_sqnorms_E.mean()<<std::endl;
    std::cout<<"Max loc :"<<locp<<" Min loc :"<<locm<<std::endl;

    Eigen::VectorXd local_sqnorms_T = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
    std::function<void(size_t,size_t)> compute_local_err_T = [&xnabla,&Iv,&Pk,&local_sqnorms_T](size_t start,size_t end)->void {
      for (size_t iT = start; iT < end; iT++) {
        Cell &T = *xnabla.mesh().cell(iT);
        double rv = 0.;
        Eigen::MatrixXd Potential_T = xnabla.cellOperators(iT).potential;
        Eigen::MatrixXd L2P = Eigen::MatrixXd::Zero(xnabla.dimensionCell(iT),xnabla.dimensionCell(iT));
        for (size_t iE = 0; iE < T.n_edges();iE++) {
          Edge &E = *T.edge(iE);
          QuadratureRule quad_dqr_E = generate_quadrature_rule(E,2*xnabla.degree() + 4);
          auto basis_Pk2po_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(iT).Polyk2po, quad_dqr_E);
          L2P += E.measure()*Potential_T.transpose() * compute_gram_matrix(basis_Pk2po_T_quad, quad_dqr_E) * Potential_T;
          for (auto node : quad_dqr_E) {
            rv -= E.measure()*node.w*Pk(node.vector()).dot(Pk(node.vector()));
          }

        }
        //rv = xnabla.restrictCell(iT,Iv).dot(xnabla.computeL2Product(iT,1.,Eigen::MatrixXd::Zero(PDim,PDim))*xnabla.restrictCell(iT,Iv));
        rv += xnabla.restrictCell(iT,Iv).dot(L2P*xnabla.restrictCell(iT,Iv));
        local_sqnorms_T[iT] = rv;
      }
    };
    parallel_for(xnabla.mesh().n_cells(),compute_local_err_T,false);
    std::cout<<"Max Min Mean"<<std::endl;
    std::cout<<FORMATD(2)<<local_sqnorms_T.maxCoeff(&locp)<<local_sqnorms_T.minCoeff(&locm)<<local_sqnorms_T.mean()<<std::endl;
    std::cout<<"Max loc :"<<locp<<" Min loc :"<<locm<<std::endl;
    std::cout<<"Max diff :"<<(local_sqnorms_T - local_sqnorms_E).cwiseAbs().maxCoeff(&locm)<<" at "<<locm<<std::endl;
    return 0;
*/


