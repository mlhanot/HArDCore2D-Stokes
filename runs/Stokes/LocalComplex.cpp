
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

double ComplexEdgeIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &,const XCurlStokes::FunctionGradType &);
double ComplexCellIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &,const XCurlStokes::FunctionGradType &);
double ComplexFullIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &,const XCurlStokes::FunctionGradType &);
double ComplexCellCG(const XCurlStokes &xcurl,const XNabla &xnabla);
double ComplexCellG(const XNabla &xnabla);


XCurlStokes::FunctionType One = [](const VectorRd &x)->double {return 1.;};
XCurlStokes::FunctionGradType dOne = [](const VectorRd &x)->VectorRd {return VectorRd::Zero();};

template<size_t > int validate_potential();

// Check the complex property on each edge and cells
int main() {
  std::cout << std::endl << "[main] Test with degree 0" << std::endl; 
  validate_potential<0>();
  std::cout << std::endl << "[main] Test with degree 1" << std::endl;
  validate_potential<1>();
  std::cout << std::endl << "[main] Test with degree 2" << std::endl;
  validate_potential<2>();
  std::cout << std::endl << "[main] Test with degree 3" << std::endl;
  validate_potential<3>();
  //std::cout << std::endl << "Number of unexpected result : "<< nb_errors << std::endl;
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
  XCurlStokes xcurl(stokes_core);
  std::cout << "[main] XCurlStokes constructed" << std::endl;

  // Create discrete space XNabla
  XNabla xnabla(stokes_core);
  std::cout << "[main] XNabla constructed" << std::endl;

  std::cout << "Max coeff CEI :"<< ComplexEdgeIC(xcurl,One,dOne) << endls;
  if constexpr (degree > 0) {
    std::cout << "Max coeff CFI :"<< ComplexCellIC(xcurl,One,dOne) << endls;
  } else {
    std::cout << "Skipping Max coeff CFI with degree 0" << endls;
  }
  std::cout << "Max coeff CFI :"<< ComplexFullIC(xcurl,One,dOne) << endls;
  std::cout << "Max coeff GCF :"<< ComplexCellCG(xcurl,xnabla) << endls;
  std::cout << "Max rank missing G :"<< ComplexCellG(xnabla) << endls;

  return 0;
}

double ComplexEdgeIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_edges());
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv);
  std::function<void(size_t,size_t)> compute_local_ops = [&xcurl,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iE = start;iE < end;iE++) {
      local_ops[iE] = (xcurl.edgeOperators(iE).curl*xcurl.restrictEdge(iE,Iv)).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xcurl.mesh().n_edges(),compute_local_ops,true);
  return local_ops.maxCoeff();
}

double ComplexCellIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv);
  std::function<void(size_t,size_t)> compute_local_ops = [&xcurl,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      local_ops[iT] = (xcurl.cellOperators(iT).proj*xcurl.cellOperators(iT).curl*xcurl.restrictCell(iT,Iv)).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xcurl.mesh().n_cells(),compute_local_ops,true);
  return local_ops.maxCoeff();
}

double ComplexFullIC(const XCurlStokes &xcurl,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xcurl.mesh().n_cells());
  Eigen::VectorXd Iv = xcurl.interpolate(v,dv);
  std::function<void(size_t,size_t)> compute_local_ops = [&xcurl,&Iv,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      local_ops[iT] = (xcurl.cellOperators(iT).ucurl*xcurl.restrictCell(iT,Iv)).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xcurl.mesh().n_cells(),compute_local_ops,true);
  return local_ops.maxCoeff();
}

double ComplexCellCG(const XCurlStokes &xcurl,const XNabla &xnabla) {
  Eigen::VectorXd local_ops = Eigen::VectorXd::Zero(xnabla.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_ops = [&xcurl,&xnabla,&local_ops](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end;iT++) {
      local_ops[iT] = (xnabla.cellOperators(iT).divergence*xcurl.cellOperators(iT).ucurl).cwiseAbs().maxCoeff();
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_ops,true);
  return local_ops.maxCoeff();
}

double ComplexCellG(const XNabla &xnabla) {
  Eigen::VectorXi local_coIm = Eigen::VectorXi::Zero(xnabla.mesh().n_cells());
  std::function<void(size_t,size_t)> compute_local_ops = [&xnabla,&local_coIm](size_t start, size_t end)->void {
    for (size_t iT = start;iT < end; iT++) {
      Eigen::MatrixXd locOp = xnabla.cellOperators(iT).divergence;
      local_coIm[iT] = locOp.rows() - locOp.fullPivLu().rank();
    }
  };
  parallel_for(xnabla.mesh().n_cells(),compute_local_ops,true);
  return local_coIm.maxCoeff();
}

