
#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xnabla.hpp>
#include <xvl.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore2D;

const std::string mesh_file = "../typ2_meshes/" "hexa1_1.typ2";

// Foward declare
double TestPotentialNabla1(const XNabla &,const XNabla::FunctionType &,bool = true);
double TestPotentialNabla1_Edge(const XNabla &,const XNabla::FunctionType &,bool = true);
Eigen::Vector2d TestPotentialNabla2(const XNabla &xnabla,Eigen::VectorXd &uv,bool = true);
double TestCDEdge(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv,bool = true);
double TestCDFace(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv,bool = true);
double TestfullCD(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv,bool = true);
double TestCDFace_woproj(const XNabla &xnabla, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv,bool = true);
double TestCDDiv(const XNabla &xnabla, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv,bool = true);

template<typename T>
XNabla::FunctionGradType FormalGrad(T &vx, T&vy) {
  return [&vx,&vy](const VectorRd &x)->MatrixRd {
    MatrixRd rv;
    rv(0,0) = vx.evaluate(x,1,0);
    rv(0,1) = vx.evaluate(x,0,1);
    rv(1,0) = vy.evaluate(x,1,0);
    rv(1,1) = vy.evaluate(x,0,1);
    return rv;};
}

template<size_t > int validate_potential();

int main() {
  std::cout << std::endl << "[main] Test with degree 0" << std::endl; 
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

  // Create discrete space XNabla
  XNabla xnabla(stokes_core);
  std::cout << "[main] XNabla constructed" << std::endl;

  // Create discrete space XVL (used to interpolate functions)
  XVL xvl(stokes_core);
  std::cout << "[main] XVL constructed" << std::endl;

  // Create test functions
  PolyTest<degree> Pkx(Initialization::Random);
  PolyTest<degree> Pky(Initialization::Random);
  PolyTest<degree + 1> Pkpox(Initialization::Random);
  PolyTest<degree + 1> Pkpoy(Initialization::Random);
  PolyTest<degree + 2> Pkp2x(Initialization::Random);
  PolyTest<degree + 2> Pkp2y(Initialization::Random);
  PolyTest<degree + 3> Pkp3x(Initialization::Random);
  PolyTest<degree + 3> Pkp3y(Initialization::Random);
  TrigTest<degree> Ttrigx(Initialization::Random);
  TrigTest<degree> Ttrigy(Initialization::Random);
  XNabla::FunctionType Pk = [&Pkx,&Pky](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkx.evaluate(x), Pky.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType DPk = FormalGrad(Pkx,Pky);
  XNabla::FunctionType Pkpo = [&Pkpox,&Pkpoy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkpox.evaluate(x), Pkpoy.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType DPkpo = FormalGrad(Pkpox,Pkpoy);
  XNabla::FunctionType Pkp2 = [&Pkp2x,&Pkp2y](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkp2x.evaluate(x), Pkp2y.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType DPkp2 = FormalGrad(Pkp2x,Pkp2y);
  XNabla::FunctionType Pkp3 = [&Pkp3x,&Pkp3y](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Pkp3x.evaluate(x), Pkp3y.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType DPkp3 = FormalGrad(Pkp3x,Pkp3y);
  XNabla::FunctionType Ttrig = [&Ttrigx,&Ttrigy](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << Ttrigx.evaluate(x), Ttrigy.evaluate(x);
    return rv;
  };
  XNabla::FunctionGradType DTtrig = FormalGrad(Ttrigx,Ttrigy);
  
  // Test 1 : CD Nabla & div
  std::cout << "[main] Begining of test CD" << std::endl;
  std::cout << "We expected everything to be zero" << std::endl;
  std::cout << "CD : NablaE" << std::endl;
  std::cout << "Error for Pk :"<< TestCDEdge(xnabla, xvl, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDEdge(xnabla, xvl, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDEdge(xnabla, xvl, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDEdge(xnabla, xvl, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestCDEdge(xnabla, xvl, Ttrig, DTtrig) << endls;
  std::cout << "CD : NablaF" << std::endl;
  std::cout << "Error for Pk :"<< TestCDFace(xnabla, xvl, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDFace(xnabla, xvl, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDFace(xnabla, xvl, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDFace(xnabla, xvl, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestCDFace(xnabla, xvl, Ttrig, DTtrig) << endls;
  std::cout << "CD : uNabla" << std::endl;
  std::cout << "Error for Pk :"<< TestfullCD(xnabla, xvl, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestfullCD(xnabla, xvl, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestfullCD(xnabla, xvl, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestfullCD(xnabla, xvl, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestfullCD(xnabla, xvl, Ttrig, DTtrig) << endls;
  std::cout << "CD : Div" << std::endl;
  std::cout << "Error for Pk :"<< TestCDDiv(xnabla, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDDiv(xnabla, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDDiv(xnabla, Pkp2, DPkp2) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDDiv(xnabla, Pkp3, DPkp3) << endls;
  std::cout << "Error for Ttrig :"<< TestCDDiv(xnabla, Ttrig, DTtrig) << endls;

  std::cout << "CD : NablaF without proj" << std::endl;
  std::cout << "We expect it to be zero up to degree k+1" << std::endl;
  std::cout << "Error for Pk :"<< TestCDFace_woproj(xnabla, Pk, DPk) << endls;
  std::cout << "Error for Pkpo :"<< TestCDFace_woproj(xnabla, Pkpo, DPkpo) << endls;
  std::cout << "Error for Pkp2 :"<< TestCDFace_woproj(xnabla, Pkp2, DPkp2,false) << endls;
  std::cout << "Error for Pkp3 :"<< TestCDFace_woproj(xnabla, Pkp3, DPkp3,false) << endls;

  // Test 2 : pIv = v, v dans Pkpo
  std::cout << "[main] Begining of test Potential Consistency" << std::endl;
  std::cout << "We expect Edge to be zero up to degree k+2 and Face to be zero up to degree k+1" << std::endl;
  std::cout << "Potential Consistency : Edge" << std::endl;
  std::cout << "Error for Pk :" << TestPotentialNabla1_Edge(xnabla, Pk) << endls;
  std::cout << "Error for Pkpo :" << TestPotentialNabla1_Edge(xnabla, Pkpo) << endls;
  std::cout << "Error for Pkp2 :" << TestPotentialNabla1_Edge(xnabla, Pkp2) << endls;
  std::cout << "Error for Pkp3 :" << TestPotentialNabla1_Edge(xnabla, Pkp3,false) << endls;
  std::cout << "Error for Ttrig :" << TestPotentialNabla1_Edge(xnabla, Ttrig,false) << endls;
  std::cout << "Potential Consistency : Face" << std::endl;
  std::cout << "Error for Pk :" << TestPotentialNabla1(xnabla, Pk) << endls;
  std::cout << "Error for Pkpo :" << TestPotentialNabla1(xnabla, Pkpo) << endls;
  std::cout << "Error for Pkp2 :" << TestPotentialNabla1(xnabla, Pkp2,false) << endls;
  std::cout << "Error for Pkp3 :" << TestPotentialNabla1(xnabla, Pkp3,false) << endls;
  std::cout << "Error for Ttrig :" << TestPotentialNabla1(xnabla, Ttrig,false) << endls;

  // Test 3 : pipv = vF
  std::cout << "[main] Begining of test Potential Consistency 2" << std::endl;
  std::cout << "We expect everything to be zero" << std::endl;
  Eigen::VectorXd randomdofs = Eigen::VectorXd::Zero(xnabla.dimension());
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestPotentialNabla2(xnabla, randomdofs).transpose() << endls; 
  fill_random_vector(randomdofs);
  std::cout << "Error for pi_Gkmo, pi_Gck :" << TestPotentialNabla2(xnabla, randomdofs).transpose() << endls; 

  return 0;
}

template<typename GeometricSupport> double computeL2Continuous(const XNabla &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads);

template<>
double computeL2Continuous<Cell>(const XNabla &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads) {
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
double computeL2Continuous<Edge>(const XNabla &core,const std::function<double(const VectorRd &, size_t)> & f, bool use_threads) {
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
double computeL2Err(const XNabla &core,const std::function<ValueType(const VectorRd &, size_t iT)>  &f1, const std::function<ValueType(const VectorRd &, size_t iT)> &f2, bool use_threads) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT)).cwiseProduct(f1(x,iT) - f2(x,iT)).sum();
  };
  return computeL2Continuous<GeometricSupport>(core,f,use_threads);
}
template<typename GeometricSupport>
double computeL2Err(const XNabla &core,const std::function<double(const VectorRd &, size_t iT)>  &f1, const std::function<double(const VectorRd &, size_t iT)> &f2, bool use_threads) {
  std::function<double(const VectorRd &, size_t)> f = [&f1,&f2](const VectorRd &x, size_t iT)->double {
    return (f1(x,iT) - f2(x,iT))*(f1(x,iT) - f2(x,iT));
  };
  return computeL2Continuous<GeometricSupport>(core,f,use_threads);
}

double TestPotentialNabla1_Edge(const XNabla &xnabla,const XNabla::FunctionType &v, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v);

  // Convert v to local function
  std::function<VectorRd(const VectorRd &, size_t)> vL = [&v](const VectorRd &x, size_t iT)->VectorRd {return v(x);};

  // Convert PIv to local function
  std::function<VectorRd(const VectorRd &, size_t)>  pIv = [&xnabla, &Iv](const VectorRd &x, size_t iE)-> VectorRd {
    return xnabla.evaluatePotential_Edge(iE, xnabla.restrictEdge(iE,Iv), x);
  };

  double rv = computeL2Err<Edge>(xnabla, vL, pIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestPotentialNabla1(const XNabla &xnabla,const XNabla::FunctionType &v, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v);

  // Convert v to local function
  std::function<VectorRd(const VectorRd &, size_t)> vL = [&v](const VectorRd &x, size_t iT)->VectorRd {return v(x);};

  // Convert PIv to local function
  std::function<VectorRd(const VectorRd &, size_t)>  pIv = [&xnabla, &Iv](const VectorRd &x, size_t iT)-> VectorRd {
    return xnabla.evaluatePotential(iT, xnabla.restrictCell(iT,Iv), x);
  };

  double rv = computeL2Err<Cell>(xnabla, vL, pIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

Eigen::Vector2d TestPotentialNabla2(const XNabla &xnabla,Eigen::VectorXd &uv, bool expect_zero) {
  
  // Manually interpolate
  size_t localdimGck = xnabla.cellBases(0).GolyComplk->dimension();
  size_t localdimGkmo = xnabla.cellBases(0).Golykmo->dimension();
  Eigen::VectorXd pgcuv = Eigen::VectorXd::Zero(localdimGck*xnabla.mesh().n_cells());
  Eigen::VectorXd pguv = Eigen::VectorXd::Zero(localdimGkmo*xnabla.mesh().n_cells());

  for (size_t iT = 0; iT < xnabla.mesh().n_cells();iT++) {
    // Convert Puv to local function
    std::function<VectorRd(const VectorRd &)>  Puv = [&xnabla, &uv, &iT](const VectorRd &x)-> VectorRd {
      return xnabla.evaluatePotential(iT, xnabla.restrictCell(iT,uv), x);
    };

    const Cell & T = *xnabla.mesh().cell(iT);
    QuadratureRule quad_dqr_T = generate_quadrature_rule(T, 2*xnabla.degree() + 3);
    auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(iT).GolyComplk,quad_dqr_T);
    pgcuv.segment(iT*localdimGck,localdimGck) = l2_projection(Puv,*xnabla.cellBases(iT).GolyComplk,quad_dqr_T,basis_Gck_T_quad);
    auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(iT).Golykmo,quad_dqr_T);
    pguv.segment(iT*localdimGkmo,localdimGkmo) = l2_projection(Puv,*xnabla.cellBases(iT).Golykmo,quad_dqr_T,basis_Gkmo_T_quad);
  }

  // Evaluation of vgc and vg
  std::function<VectorRd(const VectorRd &, size_t)> vgch = [&xnabla, &uv](const VectorRd &x, size_t iT)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Cell &T = *xnabla.mesh().cell(iT);
    size_t offset = xnabla.globalOffset(T) + xnabla.cellBases(T).Golykmo->dimension();
    auto &Gck_base = *xnabla.cellBases(T).GolyComplk;
    for (size_t i = 0; i < Gck_base.dimension(); i++) {
      rv += uv(offset + i)*Gck_base.function(i,x);
    }
    return rv;
  };
  std::function<VectorRd(const VectorRd &, size_t)> vgh = [&xnabla, &uv](const VectorRd &x, size_t iT)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Cell &T = *xnabla.mesh().cell(iT);
    size_t offset = xnabla.globalOffset(T);
    auto &Gkmo_base = *xnabla.cellBases(T).Golykmo;
    for (size_t i = 0; i < Gkmo_base.dimension(); i++) {
      rv += uv(offset + i)*Gkmo_base.function(i,x);
    }
    return rv;
  };
  
  // Evaluation of piPuv
  std::function<VectorRd(const VectorRd &, size_t)> pgcuvh = [&xnabla, &pgcuv, &localdimGck](const VectorRd &x, size_t iT)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Cell &T = *xnabla.mesh().cell(iT);
    auto &Gck_base = *xnabla.cellBases(T).GolyComplk;
    for (size_t i = 0; i < Gck_base.dimension(); i++) {
      rv += pgcuv(iT*localdimGck + i)*Gck_base.function(i,x);
    }
    return rv;
  };
  std::function<VectorRd(const VectorRd &, size_t)> pguvh = [&xnabla, &pguv, &localdimGkmo](const VectorRd &x, size_t iT)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Cell &T = *xnabla.mesh().cell(iT);
    auto &Gkmo_base = *xnabla.cellBases(T).Golykmo;
    for (size_t i = 0; i < Gkmo_base.dimension(); i++) {
      rv += pguv(iT*localdimGkmo + i)*Gkmo_base.function(i,x);
    }
    return rv;
  };

  
  VectorRd rv = VectorRd::Zero();
  rv << computeL2Err<Cell>(xnabla, vgh, pguvh, true), computeL2Err<Cell>(xnabla, vgch, pgcuvh, true);
  if (expect_zero && (rv.norm() > threshold)) nb_errors++;
  return rv;
}

double TestCDEdge(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v, 2*xnabla.degree() + 9);
  Eigen::VectorXd Idv = xvl.interpolate(dv, 2*xnabla.degree() + 9);

  // RHS to local
  std::function<VectorRd(const VectorRd &, size_t)> Idvh = [&xvl,&Idv](const VectorRd &x, size_t iE)->VectorRd {
    VectorRd rv = VectorRd::Zero();
    const Edge &E = *xvl.mesh().edge(iE);
    size_t gOffset = xvl.globalOffset(E);
    auto &edgeBases = xvl.edgeBases(iE);
    for (size_t i = 0; i < edgeBases.Polyk2po->dimension();i++) {
      rv += Idv(gOffset + i)*edgeBases.Polyk2po->function(i,x);
    }
    return rv;
  };

  // Compute NaE v
  std::function<VectorRd(const VectorRd &, size_t)> NaEIv = [&xnabla,&Iv](const VectorRd &x, size_t iE)->VectorRd {
    Eigen::VectorXd GIvE = xnabla.edgeOperators(iE).gradient * xnabla.restrictEdge(iE,Iv);
    VectorRd rv(0.,0.);
    auto &edgeBases = xnabla.edgeBases(iE);
    for (size_t i = 0; i < edgeBases.Polyk2po->dimension();i++) {
      rv += GIvE(i)*edgeBases.Polyk2po->function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Edge>(xnabla, Idvh, NaEIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestCDFace(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v, 2* xnabla.degree() + 9); // does matter for Ttrig, default (2*degree + 3) gives 5e-9, (2*degree + 6) gives 4e-12
  Eigen::VectorXd Idv = xvl.interpolate(dv, 2* xnabla.degree() + 9);

  // RHS to local
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> Idvh = [&xvl,&Idv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {
    Eigen::Matrix2d rv = Eigen::Matrix2d::Zero();
    const Cell &T = *xvl.mesh().cell(iT);
    size_t gOffset = xvl.globalOffset(T);
    auto &cellBases = xvl.cellBases(iT);
    for (size_t i = 0; i < cellBases.RTbkpo->dimension();i++) {
      rv += Idv(gOffset + i)*cellBases.RTbkpo->function(i,x);
    }
    return rv;
  };

  // std::function<Eigen::Matrix2d(const VectorRd &, size_t)> dvL = [&dv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {return dv(x);};

  // Compute NaF v
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> NaFIv = [&xnabla,&Iv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {
    Eigen::VectorXd GIv = xnabla.cellOperators(iT).gradient * xnabla.restrictCell(iT,Iv);
    Eigen::Matrix2d rv = Eigen::Matrix2d::Zero();
    auto &cellBases = xnabla.cellBases(iT);
    for (size_t i = 0; i < cellBases.RTbkpo->dimension();i++) {
      rv += GIv(i)*cellBases.RTbkpo->function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Cell>(xnabla, Idvh, NaFIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestfullCD(const XNabla &xnabla,const XVL &xvl, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v, 2* xnabla.degree() + 9); // does matter for Ttrig, default (2*degree + 3) gives 5e-9, (2*degree + 6) gives 4e-12
  Eigen::VectorXd Idv = xvl.interpolate(dv, 2* xnabla.degree() + 9);
  
  // Test on each cell
  Eigen::VectorXd local_dofs_diff = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  std::function<void(size_t, size_t)> compute_local_dofs = [&xnabla,&xvl,&Iv,&Idv,&local_dofs_diff](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      Eigen::VectorXd dofsNaIv = xnabla.cellOperators(iT).ugradient*xnabla.restrictCell(iT,Iv);
      local_dofs_diff[iT] = (dofsNaIv - xvl.restrictCell(iT,Idv)).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_dofs,true);
  
  double rv = local_dofs_diff.maxCoeff();
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestCDFace_woproj(const XNabla &xnabla, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v, 2* xnabla.degree() + 9);

  // RHS to local
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> dvL = [&dv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {return dv(x);};

  // Compute NaF v
  std::function<Eigen::Matrix2d(const VectorRd &, size_t)> NaFIv = [&xnabla,&Iv](const VectorRd &x, size_t iT)->Eigen::Matrix2d {
    Eigen::VectorXd GIv = xnabla.cellOperators(iT).gradient * xnabla.restrictCell(iT,Iv);
    Eigen::Matrix2d rv = Eigen::Matrix2d::Zero();
    auto &cellBases = xnabla.cellBases(iT);
    for (size_t i = 0; i < cellBases.RTbkpo->dimension();i++) {
      rv += GIv(i)*cellBases.RTbkpo->function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Cell>(xnabla, dvL, NaFIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

double TestCDDiv(const XNabla &xnabla, const XNabla::FunctionType &v, const XNabla::FunctionGradType &dv, bool expect_zero) {
  
  // Interpolate
  Eigen::VectorXd Iv = xnabla.interpolate(v, 2*xnabla.degree() + 9);
  // Manually interpolate pk(F)
  std::function<double(const VectorRd &)> dvL = [&dv](const VectorRd &x)->double {return dv(x).trace();};

  size_t localdim = xnabla.cellBases(0).Polyk->dimension();
  Eigen::VectorXd Igv = Eigen::VectorXd::Zero(xnabla.mesh().n_cells()*localdim);
  for (size_t iT = 0; iT < xnabla.mesh().n_cells();iT++) {
    const Cell &T = *xnabla.mesh().cell(iT);
    QuadratureRule quad_2kp3_T = generate_quadrature_rule(T,2*xnabla.degree()+9);
    auto basis_Polyk_T_quad = evaluate_quad<Function>::compute(*xnabla.cellBases(iT).Polyk,quad_2kp3_T);
    Igv.segment(iT*localdim,localdim) = l2_projection(dvL, *xnabla.cellBases(iT).Polyk,quad_2kp3_T,basis_Polyk_T_quad);
  }

  // RHS to local
  std::function<double(const VectorRd &, size_t)> Igvh = [&xnabla,&Igv,&localdim](const VectorRd &x, size_t iT)->double {
    double rv = 0.;
    auto &cellBases = xnabla.cellBases(iT);
    for (size_t i = 0; i < cellBases.Polyk->dimension();i++) {
      rv += Igv(iT*localdim + i)*cellBases.Polyk->function(i,x);
    }
    return rv;
  };
  

  // Compute Div F
  std::function<double(const VectorRd &, size_t)> DIv = [&xnabla,&Iv](const VectorRd &x, size_t iT)->double {
    Eigen::VectorXd DIvh = xnabla.cellOperators(iT).divergence * xnabla.restrictCell(iT,Iv);
    double rv = 0.;
    auto &cellBases = xnabla.cellBases(iT);
    for (size_t i = 0; i < cellBases.Polyk->dimension();i++) {
      rv += DIvh(i)*cellBases.Polyk->function(i,x);
    }
    return rv;
  };

  double rv = computeL2Err<Cell>(xnabla, Igvh, DIv, true);
  if (expect_zero && (rv > threshold)) nb_errors++;
  return rv;
}

