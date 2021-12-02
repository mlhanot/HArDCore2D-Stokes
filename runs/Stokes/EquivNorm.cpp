
#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xcurlstokes.hpp>
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
  
class XNablaL2 : public XNabla {
  public:
    std::vector<Eigen::MatrixXd> L2OPMatrix;
    std::vector<Eigen::MatrixXd> L2Matrix;
    XNablaL2(const StokesCore &stokes_core,bool use_threads = true) : XNabla(stokes_core,use_threads) {
      L2OPMatrix.resize(this->mesh().n_cells());
      L2Matrix.resize(this->mesh().n_cells());
      std::function<void(size_t, size_t)> construct_all_products = [this](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          L2OPMatrix[iT] = _compute_product(iT);
          L2Matrix[iT] = this->computeL2Product(iT);
        }  };
      parallel_for(this->mesh().n_cells(),construct_all_products,use_threads);
    }
    double compute_L2norm(const Eigen::VectorXd &uv, bool use_threads = true) const {
      Eigen::VectorXd pval = Eigen::VectorXd::Zero(this->mesh().n_cells());
      std::function<void(size_t, size_t)> compute_all_norms = [this,&pval,&uv](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          pval(iT) = restrictCell(iT,uv).dot(L2Matrix[iT]*restrictCell(iT,uv));
        } };
      parallel_for(this->mesh().n_cells(),compute_all_norms,use_threads);
      return std::sqrt(pval.sum());
    } 
    double compute_L2OPnorm(const Eigen::VectorXd &uv, bool use_threads = true) const {
      Eigen::VectorXd pval = Eigen::VectorXd::Zero(this->mesh().n_cells());
      std::function<void(size_t, size_t)> compute_all_norms = [this,&pval,&uv](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          pval(iT) = restrictCell(iT,uv).dot(L2OPMatrix[iT]*restrictCell(iT,uv));
        } };
      parallel_for(this->mesh().n_cells(),compute_all_norms,use_threads);
      return std::sqrt(pval.sum());
    } 
  private:
    Eigen::MatrixXd _compute_product(size_t iT) {
      const Cell & T = *this->mesh().cell(iT);
      double h_F = T.diam();
      Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(dimensionCell(T),dimensionCell(T));
      // Cells contributions
      QuadratureRule quad_2k_T = generate_quadrature_rule(T, 2* this->degree());
      auto basis_Gkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Golykmo, quad_2k_T);
      auto basis_Gck_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).GolyComplk, quad_2k_T);
      size_t dimGkmo = cellBases(iT).Golykmo->dimension();
      size_t dimGck = cellBases(iT).GolyComplk->dimension();
      rv.block(localOffset(T),localOffset(T),dimGkmo,dimGkmo) = compute_gram_matrix(basis_Gkmo_T_quad,basis_Gkmo_T_quad,quad_2k_T,"sym");
      rv.block(localOffset(T)+dimGkmo,localOffset(T)+dimGkmo,dimGck,dimGck) = compute_gram_matrix(basis_Gck_T_quad,basis_Gck_T_quad,quad_2k_T,"sym");
      // Edges contributions
      for (size_t iE = 0; iE < T.n_edges();iE++) {
        const Edge & E = *T.edge(iE);
        QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2* this->degree());
        auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk, quad_2k_E);
        size_t dimPk = edgeBases(E).Polyk->dimension();
        rv.block(localOffset(T,E),localOffset(T,E),dimPk,dimPk) = h_F*compute_gram_matrix(basis_Pk_E_quad,basis_Pk_E_quad,quad_2k_E,"sym");
        rv.block(localOffset(T,E)+dimPk,localOffset(T,E)+dimPk,dimPk,dimPk) = h_F*compute_gram_matrix(basis_Pk_E_quad,basis_Pk_E_quad,quad_2k_E,"sym");
      }
      // Vertices contribution
      for (size_t iV = 0; iV < T.n_vertices(); iV++) {
        const Vertex &V = *T.vertex(iV);
        rv(localOffset(T,V),localOffset(T,V)) = h_F*h_F;
        rv(localOffset(T,V)+1,localOffset(T,V)+1) = h_F*h_F;
      }
      return rv;
    }
};

class XCurlL2 : public XCurlStokes {
  public:
    std::vector<Eigen::MatrixXd> L2OPMatrix;
    std::vector<Eigen::MatrixXd> L2Matrix;
    XCurlL2(const StokesCore &stokes_core,bool use_threads = true) : XCurlStokes(stokes_core,use_threads) {
      L2OPMatrix.resize(this->mesh().n_cells());
      L2Matrix.resize(this->mesh().n_cells());
      std::function<void(size_t, size_t)> construct_all_products = [this](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          L2OPMatrix[iT] = _compute_product(iT);
          L2Matrix[iT] = this->computeL2Product(iT);
        }  };
      parallel_for(this->mesh().n_cells(),construct_all_products,use_threads);
    }
    double compute_L2norm(const Eigen::VectorXd &uv, bool use_threads = true) const {
      Eigen::VectorXd pval = Eigen::VectorXd::Zero(this->mesh().n_cells());
      std::function<void(size_t, size_t)> compute_all_norms = [this,&pval,&uv](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          pval(iT) = restrictCell(iT,uv).dot(L2Matrix[iT]*restrictCell(iT,uv));
        } };
      parallel_for(this->mesh().n_cells(),compute_all_norms,use_threads);
      return std::sqrt(pval.sum());
    } 
    double compute_L2OPnorm(const Eigen::VectorXd &uv, bool use_threads = true) const {
      Eigen::VectorXd pval = Eigen::VectorXd::Zero(this->mesh().n_cells());
      std::function<void(size_t, size_t)> compute_all_norms = [this,&pval,&uv](size_t start, size_t end)->void {
        for (size_t iT = start; iT < end; iT++) {
          pval(iT) = restrictCell(iT,uv).dot(L2OPMatrix[iT]*restrictCell(iT,uv));
        } };
      parallel_for(this->mesh().n_cells(),compute_all_norms,use_threads);
      return std::sqrt(pval.sum());
    } 
  private:
    Eigen::MatrixXd _compute_product(size_t iT) {
      const Cell & T = *this->mesh().cell(iT);
      double h_F = T.diam();
      Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(dimensionCell(T),dimensionCell(T));
      // Cells contributions
      QuadratureRule quad_2kmo_T = generate_quadrature_rule(T, 2* (this->degree()-1));
      auto basis_Pkmo_T_quad = evaluate_quad<Function>::compute(*cellBases(iT).Polykmo, quad_2kmo_T);
      size_t dimPkmo = cellBases(iT).Polykmo->dimension();
      rv.block(localOffset(T),localOffset(T),dimPkmo,dimPkmo) = compute_gram_matrix(basis_Pkmo_T_quad,basis_Pkmo_T_quad,quad_2kmo_T,"sym");
      // Edges contributions
      for (size_t iE = 0; iE < T.n_edges();iE++) {
        const Edge & E = *T.edge(iE);
        QuadratureRule quad_2k_E = generate_quadrature_rule(E, 2* this->degree());
        auto basis_Pk_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polyk, quad_2k_E);
        auto basis_Pkmo_E_quad = evaluate_quad<Function>::compute(*edgeBases(E).Polykmo, quad_2k_E);
        size_t dimPk = edgeBases(E).Polyk->dimension();
        size_t dimPkmo = edgeBases(E).Polykmo->dimension();
        rv.block(localOffset(T,E),localOffset(T,E),dimPkmo,dimPkmo) = h_F*compute_gram_matrix(basis_Pkmo_E_quad,basis_Pkmo_E_quad,quad_2k_E,"sym");
        rv.block(localOffset(T,E)+dimPkmo,localOffset(T,E)+dimPkmo,dimPk,dimPk) = h_F*compute_gram_matrix(basis_Pk_E_quad,basis_Pk_E_quad,quad_2k_E,"sym");
      }
      // Vertices contribution
      for (size_t iV = 0; iV < T.n_vertices(); iV++) {
        const Vertex &V = *T.vertex(iV);
        rv(localOffset(T,V),localOffset(T,V)) = h_F*h_F;
        rv(localOffset(T,V)+1,localOffset(T,V)+1) = h_F*h_F;
        rv(localOffset(T,V)+2,localOffset(T,V)+2) = h_F*h_F;
      }
      return rv;
    }
};

Eigen::Vector2d Interpolate_norm(const XCurlL2 &xcurlL2,const XCurlStokes::FunctionType &v,const XCurlStokes::FunctionGradType &dv) { 
  Eigen::Vector2d rv = Eigen::Vector2d::Zero();
  Eigen::VectorXd uv = xcurlL2.interpolate(v,dv);
  rv(0) = xcurlL2.compute_L2norm(uv);
  rv(1) = xcurlL2.compute_L2OPnorm(uv);
  return rv;
}
Eigen::Vector2d Interpolate_norm(const XNablaL2 &xnablaL2,const XNabla::FunctionType &v) {
  Eigen::Vector2d rv = Eigen::Vector2d::Zero();
  Eigen::VectorXd uv = xnablaL2.interpolate(v);
  rv(0) = xnablaL2.compute_L2norm(uv);
  rv(1) = xnablaL2.compute_L2OPnorm(uv);
  return rv;
}

template<typename T>
XCurlStokes::FunctionGradType FormalGrad(T &v) {
  return [&v](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv(0) = v.evaluate(x,0,1);
    rv(1) = -v.evaluate(x,1,0);
    return rv;};
}

template<size_t degree>
int validate_potential() {
  // Create test functions Nabla
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

  // Create test functions XCurl
  XCurlStokes::FunctionType cPk = [&Pkx](const VectorRd &x)->double {
    return Pkx.evaluate(x);
  };
  XCurlStokes::FunctionGradType cDPk = FormalGrad(Pkx);
  XCurlStokes::FunctionType cPkpo = [&Pkpox](const VectorRd &x)->double {
    return Pkpox.evaluate(x);
  };
  XCurlStokes::FunctionGradType cDPkpo = FormalGrad(Pkpox);
  XCurlStokes::FunctionType cPkp2 = [&Pkp2x](const VectorRd &x)->double {
    return Pkp2x.evaluate(x);
  };
  XCurlStokes::FunctionGradType cDPkp2 = FormalGrad(Pkp2x);
  XCurlStokes::FunctionType cPkp3 = [&Pkp3x](const VectorRd &x)->double {
    return Pkp3x.evaluate(x);
  };
  XCurlStokes::FunctionGradType cDPkp3 = FormalGrad(Pkp3x);
  XCurlStokes::FunctionType cTtrig = [&Ttrigx](const VectorRd &x)->double {
    return Ttrigx.evaluate(x);
  };
  XCurlStokes::FunctionGradType cDTtrig = FormalGrad(Ttrigx);

  // Storage for resulting value
  std::vector<Eigen::VectorXd> normNa;
  std::vector<Eigen::VectorXd> normOPNa;
  std::vector<Eigen::VectorXd> normC;
  std::vector<Eigen::VectorXd> normOPC;

  // Iterate over meshes
  for (auto mesh_file : mesh_files) {
    // Build the mesh
    MeshBuilder builder = MeshBuilder(mesh_file);
    std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
    std::cout << FORMAT(25) << "[main] Mesh size" << mesh_ptr->h_max() << std::endl;

    // Create core 
    StokesCore stokes_core(*mesh_ptr,degree);
    std::cout << "[main] StokesCore constructed" << std::endl;

    // Create discrete space XCurlStokes
    XCurlL2 xcurl(stokes_core,true);
    std::cout << "[main] XCurlStokes constructed" << std::endl;

    // Create discrete space XNabla
    XNablaL2 xnabla(stokes_core);
    std::cout << "[main] XNabla constructed" << std::endl;

    Eigen::VectorXd randomdofsxcurl = Eigen::VectorXd::Zero(xcurl.dimension());
    Eigen::VectorXd randomdofsxnabla = Eigen::VectorXd::Zero(xnabla.dimension());
    
    Eigen::VectorXd norm = Eigen::VectorXd::Zero(7); // number of functions
    Eigen::VectorXd normOP = Eigen::VectorXd::Zero(7); // number of functions
    
    Eigen::Vector2d rv;
    // Compute norms on XCurl
    rv = Interpolate_norm(xcurl,cPk,cDPk); norm(0) = rv(0); normOP(0) = rv(1);
    rv = Interpolate_norm(xcurl,cPkpo,cDPkpo); norm(1) = rv(0); normOP(1) = rv(1);
    rv = Interpolate_norm(xcurl,cPkp2,cDPkp2); norm(2) = rv(0); normOP(2) = rv(1);
    rv = Interpolate_norm(xcurl,cPkp3,cDPkp3); norm(3) = rv(0); normOP(3) = rv(1);
    rv = Interpolate_norm(xcurl,cTtrig,cDTtrig); norm(4) = rv(0); normOP(4) = rv(1);
    fill_random_vector(randomdofsxcurl); 
    norm(5) = xcurl.compute_L2norm(randomdofsxcurl); normOP(5)= xcurl.compute_L2OPnorm(randomdofsxcurl);
    fill_random_vector(randomdofsxcurl); 
    norm(6) = xcurl.compute_L2norm(randomdofsxcurl); normOP(6)= xcurl.compute_L2OPnorm(randomdofsxcurl);
    normC.emplace_back(norm);
    normOPC.emplace_back(normOP);
    std::cout << "[main] Norm on XCurl computed" << std::endl;

    // Compute norms on XNabla
    rv = Interpolate_norm(xnabla,Pk); norm(0) = rv(0); normOP(0) = rv(1);
    rv = Interpolate_norm(xnabla,Pkpo); norm(1) = rv(0); normOP(1) = rv(1);
    rv = Interpolate_norm(xnabla,Pkp2); norm(2) = rv(0); normOP(2) = rv(1);
    rv = Interpolate_norm(xnabla,Pkp3); norm(3) = rv(0); normOP(3) = rv(1);
    rv = Interpolate_norm(xnabla,Ttrig); norm(4) = rv(0); normOP(4) = rv(1);
    fill_random_vector(randomdofsxnabla); 
    norm(5) = xnabla.compute_L2norm(randomdofsxnabla); normOP(5)= xnabla.compute_L2OPnorm(randomdofsxnabla);
    fill_random_vector(randomdofsxnabla); 
    norm(6) = xnabla.compute_L2norm(randomdofsxnabla); normOP(6)= xnabla.compute_L2OPnorm(randomdofsxnabla);
    normNa.emplace_back(norm);
    normOPNa.emplace_back(normOP);
    std::cout << "[main] Norm on XNabla computed" << std::endl;
  } // end for mesh_files

  /*
  for (int j = 0; j < normC[0].size(); j++) {
    std::cout << "NormC    NormOPC  NormNa   NormOPNa" << std::endl;
    for (size_t i = 0; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<normC[i](j)<<normOPC[i](j)<<normNa[i](j)<<normOPNa[i](j)<<std::endl;}
  }
  */
  for (int j = 0; j < normC[0].size(); j++) {
    std::cout << "RatioC1  RatioC2  RatioNa1 RatioNa2" << std::endl;
    for (size_t i = 0; i < mesh_files.size();i++) {std::cout<<FORMATD(2)<<normC[i](j)/normOPC[i](j)<<normOPC[i](j)/normC[i](j)<<normNa[i](j)/normOPNa[i](j)<<normOPNa[i](j)/normNa[i](j)<<std::endl;}
  }
  
  return 0;
}

