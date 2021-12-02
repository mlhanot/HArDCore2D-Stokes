
#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xcurlstokes.hpp>
#include <xnabla.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore2D;

const std::string mesh_file = "../typ2_meshes/" "hexa1_1.typ2";

// Foward declare
double TestPotentialCurl1(const XCurlStokes &,const XCurlStokes::FunctionType &,const XCurlStokes::FunctionGradType &,bool = true);
double TestPotentialCurl1_Edge(const XCurlStokes &,const XCurlStokes::FunctionType &,const XCurlStokes::FunctionGradType &,bool = true);
double TestPotentialCurl2(const XCurlStokes &xcurl,Eigen::VectorXd &uv,bool = true);
double TestCDEdge(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv,bool = true);
double TestCDFace(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv,bool = true);
double TestfullCD(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv,bool = true);

template<typename T>
XCurlStokes::FunctionGradType FormalGrad(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v.evaluate(x,0,1);
    rv(1) = -v.evaluate(x,1,0);
    return rv;};
}

template<size_t > int validate_potential();

int main() {
  std::cout << std::endl << "[main] Test with degree 0" << std::endl; // TODO fix case deg 0;
  validate_potential<0>();
  std::cout << std::endl << "[main] Test with degree 1" << std::endl;
  validate_potential<1>();
  std::cout << std::endl << "[main] Test with degree 2" << std::endl;
  validate_potential<2>();
  std::cout << std::endl << "[main] Test with degree 3" << std::endl;
  validate_potential<3>();
  std::cout << std::endl << "Number of unexpected result : "<< nb_errors << std::endl;
  return nb_errors;
}
  

template<size_t degree>
int validate_potential() {

  // Build the mesh
  MeshBuilder builder = MeshBuilder(mesh_file);
  std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
  std::cout << FORMAT(25) << "[main] Mesh size" << mesh_ptr->h_max() << std::endl;
  
  // Create core 
  StokesCore stokes_core(*mesh_ptr,degree);
  std::cout << "[main] StokesCore constructed" << std::endl;

  // Create discrete space XCurlStokes
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create discrete space XNabla (used to interpolate functions)
  XNabla xnabla(stokes_core);
  std::cout << "[main] XNabla constructed" << std::endl;

  // Create test functions
  PolyTest<degree> Pkx(Initialization::Random);
  PolyTest<degree + 1> Pkpox(Initialization::Random);
  PolyTest<degree + 2> Pkp2x(Initialization::Random);
  PolyTest<degree + 3> Pkp3x(Initialization::Random);
  TrigTest<degree> Ttrigx(Initialization::Random);
  XCurlStokes::FunctionType Pk = [&Pkx](const VectorRd &x)->double {
    return Pkx.evaluate(x);
  };
  XCurlStokes::FunctionGradType DPk = FormalGrad(Pkx);
  XCurlStokes::FunctionType Pkpo = [&Pkpox](const VectorRd &x)->double {
    return Pkpox.evaluate(x);
  };
  XCurlStokes::FunctionGradType DPkpo = FormalGrad(Pkpox);
  XCurlStokes::FunctionType Pkp2 = [&Pkp2x](const VectorRd &x)->double {
    return Pkp2x.evaluate(x);
  };
  XCurlStokes::FunctionGradType DPkp2 = FormalGrad(Pkp2x);
  XCurlStokes::FunctionType Pkp3 = [&Pkp3x](const VectorRd &x)->double {
    return Pkp3x.evaluate(x);
  };
  XCurlStokes::FunctionGradType DPkp3 = FormalGrad(Pkp3x);
  XCurlStokes::FunctionType Ttrig = [&Ttrigx](const VectorRd &x)->double {
    return Ttrigx.evaluate(x);
  };
  XCurlStokes::FunctionGradType DTtrig = FormalGrad(Ttrigx);
  
  // Test 1 : CD Nabla & div
  std::cout << "[main] Begining of test CD" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "CD : CurlE" << std::endl;
  std::cout << "Error for Pk :"<< TestCDEdge(xcurl, xnabla, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDEdge(xcurl, xnabla, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDEdge(xcurl, xnabla, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDEdge(xcurl, xnabla, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestCDEdge(xcurl, xnabla, Ttrig, DTtrig) << endls;
  std::cout << "CD : CurlF" << std::endl;
  std::cout << "Error for Pk :"<< TestCDFace(xcurl, xnabla, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDFace(xcurl, xnabla, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDFace(xcurl, xnabla, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDFace(xcurl, xnabla, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestCDFace(xcurl, xnabla, Ttrig, DTtrig) << endls;
  std::cout << "CD : uCurl" << std::endl;
  std::cout << "Error for Pk :"<< TestfullCD(xcurl, xnabla, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestfullCD(xcurl, xnabla, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestfullCD(xcurl, xnabla, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestfullCD(xcurl, xnabla, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestfullCD(xcurl, xnabla, Ttrig, DTtrig) << endls;

  // Test 2 : pIv = v, v dans Pkpo
  std::cout << "[main] Begining of test Potential Consistency" << std::endl;
  std::cout << "We expect Edge to be zero up to degree k+1 and Face to be zero up to degree k+1" << std::endl;
  std::cout << "Potential Consistency : Edge" << std::endl;
  std::cout << "Error for Pk :" << TestPotentialCurl1_Edge(xcurl, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :" << TestPotentialCurl1_Edge(xcurl, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :" << TestPotentialCurl1_Edge(xcurl, Pkp2, DPkp2,false) << endls;
  std::cout << "Error for Pkp3 :" << TestPotentialCurl1_Edge(xcurl, Pkp3, DPkp3,false) << endls;
  std::cout << "Error for Ttrig :" << TestPotentialCurl1_Edge(xcurl, Ttrig, DTtrig,false) << endls;
  std::cout << "Potential Consistency : Face" << std::endl;
  std::cout << "Error for Pk :" << TestPotentialCurl1(xcurl, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :" << TestPotentialCurl1(xcurl, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :" << TestPotentialCurl1(xcurl, Pkp2, DPkp2,false) << endls;
  std::cout << "Error for Pkp3 :" << TestPotentialCurl1(xcurl, Pkp3, DPkp3,false) << endls;
  std::cout << "Error for Ttrig :" << TestPotentialCurl1(xcurl, Ttrig, DTtrig,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "[main] Begining of test Potential Consistency 2" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  Eigen::VectorXd randomdofs = Eigen::VectorXd::Zero(xcurl.dimension());
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestPotentialCurl2(xcurl, randomdofs) << endls; 
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestPotentialCurl2(xcurl, randomdofs) << endls; 


  return 0;
}

template<typename GeometricSupport> double computeL2Continuous(const XCurlStokes &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads);

template<>
double computeL2Continuous<Cell>(const XCurlStokes &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&f,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      Cell & T = *core.mesh().cell(iT);
      QuadratureRule quad_2kp2_T = generate_quadrature_rule(T, 2*(core.degree()+2));
      
      double rv = 0.;
      for (auto node : quad_2kp2_T) {
        rv += node.w*f(node.vector(), iT);
      }
      local_sqnorms[iT] = rv;
    }
  };
  parallel_for(core.mesh().n_cells(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}
template<>
double computeL2Continuous<Edge>(const XCurlStokes &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads) {
  Eigen::VectorXd local_sqnorms = Eigen::VectorXd::Zero(core.mesh().n_edges());
  std::function<void(size_t,size_t)> compute_local_squarednorms = [&core,&f,&local_sqnorms](size_t start,size_t end)->void {
    for (size_t iE = start;iE < end; iE++) {
      Edge & E = *core.mesh().edge(iE);
      QuadratureRule quad_2kp2_E = generate_quadrature_rule(E, 2*(core.degree()+2));
      
      double rv = 0.;
      for (auto node : quad_2kp2_E) {
        rv += node.w*f(node.vector(), iE);
      }
      local_sqnorms[iE] = rv;
    }
  };
  parallel_for(core.mesh().n_edges(),compute_local_squarednorms,use_threads);

  return std::sqrt(std::abs(local_sqnorms.sum()));
}

template<typename GeometricSupport, typename ValueType>
double computeL2Err(const XCurlStokes &core,const std::function<ValueType(const VectorRd &, size_t iT)>  &f1, const std::function<ValueType(const VectorRd &, size_t iT)> &f2, bool use_threads) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT)).cwiseProduct(f1(x,iT) - f2(x,iT)).sum();
  };
  return computeL2Continuous<GeometricSupport>(core,f,use_threads);
}
template<typename GeometricSupport>
double computeL2Err(const XCurlStokes &core,const std::function<double(const VectorRd &, size_t iT)>  &f1, const std::function<double(const VectorRd &, size_t iT)> &f2, bool use_threads) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT))*(f1(x,iT) - f2(x,iT));
  };
  return computeL2Continuous<GeometricSupport>(core,f,use_threads);
}

double TestPotentialCurl1_Edge(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv);

  // Convert v to local function
  std::function<double(const VectorRd &, size_t)> vL = [&v](const VectorRd &x, size_t iT)->double {return v(x);};

  // Convert PIv to local function
  std::function<double(const VectorRd &, size_t)>  pIv = [&xcurl, &Iv](const VectorRd &x, size_t iE)-> double {
    return xcurl.evaluatePotential_Edge(iE, xcurl.restrictEdge(iE,Iv), x);
  };

  double rv = computeL2Err<Edge>(xcurl, vL, pIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestPotentialCurl1(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv);

  // Convert v to local function
  std::function<double(const VectorRd &, size_t)> vL = [&v](const VectorRd &x, size_t iT)->double {return v(x);};

  // Convert PIv to local function
  std::function<double(const VectorRd &, size_t)>  pIv = [&xcurl, &Iv](const VectorRd &x, size_t iT)-> double {
    return xcurl.evaluatePotential(iT, xcurl.restrictCell(iT,Iv), x);
  };

  double rv = computeL2Err<Cell>(xcurl, vL, pIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestPotentialCurl2(const XCurlStokes &xcurl,Eigen::VectorXd &uv, bool expect_zero) {
  
  // Manually interpolate
  size_t localdim = xcurl.cellBases(0).Polykmo->dimension();
  Eigen::VectorXd projuv = Eigen::VectorXd::Zero(localdim*xcurl.mesh().n_cells());

  for (size_t iT = 0; iT < xcurl.mesh().n_cells();iT++) {
    // Convert Puv to local function
    std::function<double(const VectorRd &)>  Puv = [&xcurl, &uv, &iT](const VectorRd &x)-> double {
      return xcurl.evaluatePotential(iT, xcurl.restrictCell(iT,uv), x);
    };

    const Cell & T = *xcurl.mesh().cell(iT);
    QuadratureRule quad_dqr_T = generate_quadrature_rule(T, 2*xcurl.degree() + 3);
    auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(*xcurl.cellBases(iT).Polykmo,quad_dqr_T);
    projuv.segment(iT*localdim,localdim) = l2_projection(Puv,*xcurl.cellBases(iT).Polykmo,quad_dqr_T,basis_Pkmo_T_quad);
  }

  // Evaluation of qF
  std::function<double(const VectorRd &, size_t)> qFh = [&xcurl, &uv](const VectorRd &x, size_t iT)->double {
    double rv = 0.;
    const Cell &T = *xcurl.mesh().cell(iT);
    size_t offset = xcurl.globalOffset(T);
    auto &Pkmo_base = *xcurl.cellBases(T).Polykmo;
    for (size_t i = 0; i < Pkmo_base.dimension(); i++) {
      rv += uv(offset + i)*Pkmo_base.function(i,x);
    }
    return rv;
  };
  
  // Evaluation of piPuv
  std::function<double(const VectorRd &, size_t)> projpuvh = [&xcurl, &projuv, &localdim](const VectorRd &x, size_t iT)->double {
    double rv = 0.;
    const Cell &T = *xcurl.mesh().cell(iT);
    auto &Pkmo_base = *xcurl.cellBases(T).Polykmo;
    for (size_t i = 0; i < Pkmo_base.dimension(); i++) {
      rv += projuv(iT*localdim + i)*Pkmo_base.function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Cell>(xcurl, qFh, projpuvh, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestCDEdge(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv, 2*xcurl.degree() + 9);
  Eigen::VectorXd Idv = xnabla.interpolate(dv, 2*xcurl.degree() + 9);

  // RHS to local
  std::function<VectorRd(const VectorRd &, size_t)> Idvh = [&xnabla,&Idv](const VectorRd &x, size_t iE)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Edge &E = *xnabla.mesh().edge(iE);
    size_t gOffset = xnabla.globalOffset(E);
    auto &edgeBases = xnabla.edgeBases(iE);
    size_t dimPk = edgeBases.Polyk->dimension();
    for (size_t i = 0; i < dimPk;i++) {
      rv += Idv(gOffset + i)*edgeBases.Polyk->function(i,x)*E.tangent();
      rv += Idv(gOffset + dimPk + i)*edgeBases.Polyk->function(i,x)*E.normal();
    }
    return rv;
  };

  // Compute CE v
  std::function<VectorRd(const VectorRd &, size_t)> CEIv = [&xcurl,&Iv](const VectorRd &x, size_t iE)->VectorRd {
    Eigen::VectorXd GIvE = xcurl.edgeOperators(iE).curl * xcurl.restrictEdge(iE,Iv);
    VectorRd rv(0.,0.);
    const Edge &E = *xcurl.mesh().edge(iE);
    auto &edgeBases = xcurl.edgeBases(iE);
    size_t dimPk = edgeBases.Polyk->dimension();
    for (size_t i = 0; i < dimPk;i++) {
      rv += GIvE(4+i)*edgeBases.Polykpo->function(i,x)*E.tangent(); // 4 is the offset of vertices
      rv += GIvE(4+dimPk + i)*edgeBases.Polykpo->function(i,x)*E.normal();
    }
    return rv;
  };

  double rv = computeL2Err<Edge>(xcurl, Idvh, CEIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestCDFace(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv, 2* xcurl.degree() + 9); 
  Eigen::VectorXd Idv = xnabla.interpolate(dv, 2* xcurl.degree() + 9);

  // vkmo stored in rv.col(0), vgck stored in rv.col(1)
  // RHS to local
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> Idvh = [&xnabla,&Idv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {
    Eigen::Matrix2d rv = Eigen::Matrix2d::Zero();
    const Cell &T = *xnabla.mesh().cell(iT);
    size_t gOffset = xnabla.globalOffset(T);
    auto &cellBases = xnabla.cellBases(iT);
    for (size_t i = 0; i < cellBases.Golykmo->dimension();i++) {
      rv.col(0) += Idv(gOffset + i)*cellBases.Golykmo->function(i,x);
    }
    gOffset += cellBases.Golykmo->dimension();
    for (size_t i = 0; i < cellBases.GolyComplk->dimension();i++) {
      rv.col(1) += Idv(gOffset + i)*cellBases.GolyComplk->function(i,x);
    }
    return rv;
  };

  // std::function<Eigen::Matrix2d(const VectorRd &, size_t)> dvL = [&dv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {return dv(x);};

  // Compute NaF v
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> CFIv = [&xcurl,&Iv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {
    Eigen::VectorXd GIv = xcurl.cellOperators(iT).proj * xcurl.cellOperators(iT).curl * xcurl.restrictCell(iT,Iv);
    Eigen::Matrix2d rv = Eigen::Matrix2d::Zero();
    auto &cellBases = xcurl.cellBases(iT);
    for (size_t i = 0; i < cellBases.Golykmo->dimension();i++) {
      rv.col(0) += GIv(i)*cellBases.Golykmo->function(i,x);
    }
    size_t gOffset = cellBases.Golykmo->dimension();
    for (size_t i = 0; i < cellBases.GolyComplk->dimension();i++) {
      rv.col(1) += GIv(gOffset + i)*cellBases.GolyComplk->function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Cell>(xcurl, Idvh, CFIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestfullCD(const XCurlStokes &xcurl,const XNabla &xnabla, const XCurlStokes::FunctionType &v, const XCurlStokes::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xcurl.interpolate(v, dv, 2* xnabla.degree() + 9); // does matter for Ttrig, default (2*degree + 3) gives 5e-9, (2*degree + 6) gives 4e-12
  Eigen::VectorXd Idv = xnabla.interpolate(dv, 2* xnabla.degree() + 9);
  
  // Test on each cell
  Eigen::VectorXd local_dofs_diff = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  std::function<void(size_t, size_t)> compute_local_dofs = [&xcurl,&xnabla,&Iv,&Idv,&local_dofs_diff](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Eigen::VectorXd dofsCIv = xcurl.cellOperators(iT).ucurl*xcurl.restrictCell(iT,Iv);
      local_dofs_diff[iT] = (dofsCIv - xnabla.restrictCell(iT,Idv)).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xcurl.mesh().n_cells(),compute_local_dofs,true);
  
  double rv = local_dofs_diff.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

