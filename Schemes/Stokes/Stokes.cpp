#include "Stokes.hpp"

#include <mesh_builder.hpp>
#include <vtu_writer.hpp>

#include <iomanip>
class formatted_output {
    private:
      int width;
      std::ostream& stream_obj;
    public:
      formatted_output(std::ostream& obj, int w): width(w), stream_obj(obj) {}
      template<typename T>
      formatted_output& operator<<(const T& output) {
        stream_obj << std::setw(width) << output;

        return *this;}

      formatted_output& operator<<(std::ostream& (*func)(std::ostream&)) {
        func(stream_obj);
        return *this;}
  };
#define FORMATD(W)                                                      \
  ""; formatted_output(std::cout,W+8) << std::setiosflags(std::ios_base::left | std::ios_base::scientific) << std::setprecision(W) << std::setfill(' ')

inline double compute_rate(const std::vector<double> &a, const std::vector<double> &h, size_t i) {
  return (std::log(a[i]) - std::log(a[i-1]))/(std::log(h[i]) - std::log(h[i-1]));
}

using namespace HArDCore2D;

/// Test parameters
bool constexpr write_sols = false;
#ifndef SOLN
#define SOLN 5
#endif
#ifndef TESTCASE
#define TESTCASE 4
#endif

#if TESTCASE == 0
const std::vector<std::string> mesh_files = {"../../typ2_meshes/" "hexa1_1.typ2",
                                             "../../typ2_meshes/" "hexa1_2.typ2",
                                             "../../typ2_meshes/" "hexa1_3.typ2",
                                             "../../typ2_meshes/" "hexa1_4.typ2"};/*,
                                             "../../typ2_meshes/" "hexa1_5.typ2"};*/

#elif TESTCASE == 1
const std::vector<std::string> mesh_files = {"../../typ2_meshes/" "mesh2_1.typ2",
                                             "../../typ2_meshes/" "mesh2_2.typ2",
                                             "../../typ2_meshes/" "mesh2_3.typ2",
                                             "../../typ2_meshes/" "mesh2_4.typ2"};/*,
                                             "../../typ2_meshes/" "mesh2_5.typ2"};*/

#elif TESTCASE == 2
const std::vector<std::string> mesh_files = {"../../typ2_meshes/" "mesh3_1.typ2",
                                             "../../typ2_meshes/" "mesh3_2.typ2",
                                             "../../typ2_meshes/" "mesh3_3.typ2",
                                             "../../typ2_meshes/" "mesh3_4.typ2",
                                             "../../typ2_meshes/" "mesh3_5.typ2"};

#elif TESTCASE == 3
const std::vector<std::string> mesh_files = {"../../typ2_meshes/" "mesh4_1_1.typ2",
                                             "../../typ2_meshes/" "mesh4_1_2.typ2",
                                             "../../typ2_meshes/" "mesh4_1_3.typ2",
                                             "../../typ2_meshes/" "mesh4_1_4.typ2"};/*,
                                             "../../typ2_meshes/" "mesh4_1_5.typ2"};*/

#elif TESTCASE == 4
const std::vector<std::string> mesh_files = {"../../typ2_meshes/" "mesh4_2_1.typ2",
                                             "../../typ2_meshes/" "mesh4_2_2.typ2",
                                             "../../typ2_meshes/" "mesh4_2_3.typ2"};/*,
                                             "../../typ2_meshes/" "mesh4_2_4.typ2",
                                             "../../typ2_meshes/" "mesh4_2_5.typ2"};*/

#elif TESTCASE == 5
const std::vector<std::string> mesh_files = {"../../typ2_meshes/anisotropic/" "hexa_straight10x10.typ2",
                                             "../../typ2_meshes/anisotropic/" "hexa_straight20x20.typ2",
                                             //"../../typ2_meshes/anisotropic/" "hexa_straight20x40.typ2",
                                             "../../typ2_meshes/anisotropic/" "hexa_straight40x40.typ2",
                                             "../../typ2_meshes/anisotropic/" "hexa_straight80x80.typ2"};
#endif
template<size_t > void validate_Stokes();

int main()
{
  std::cout << std::endl << "\033[31m[main] Test with degree 0\033[0m" << std::endl; 
  validate_Stokes<0>();
  std::cout << std::endl << "\033[31m[main] Test with degree 1\033[0m" << std::endl;
  validate_Stokes<1>();
  std::cout << std::endl << "\033[31m[main] Test with degree 2\033[0m" << std::endl;
  validate_Stokes<2>();
  std::cout << std::endl << "\033[31m[main] Test with degree 3\033[0m" << std::endl;
  validate_Stokes<3>();
  return 0;
}

#if SOLN == 0
template<size_t k_int> // Dirichlet, non homogenous
struct Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  
  std::function<VectorRd(const VectorRd &)> u_exact = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << x(0)*(1. - x(0))*std::sin(k*x(0))*std::cos(k*x(1)),
          (k*x(0)*(x(0) - 1.)*std::cos(k*x(0)) + (2.*x(0) - 1.)*std::sin(k*x(0)))*std::sin(k*x(1))/k;
    return rv;
  };
  std::function<double(const VectorRd &)> p_exact = [](const VectorRd &x)->double {
    return x(0) + std::cos(k*(x(0)-x(1))) - 0.5 - 2.*(1. - std::cos(k))/(k*k);
  };
  std::function<VectorRd(const VectorRd &)> f = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << -k*k*x(0)*(x(0) - 1.)*std::sin(k*x(0))*std::cos(k*x(1)) - k*std::sin(k*(x(0) - x(1))) + (-k*k*x(0)*(x(0) - 1.)*std::sin(k*x(0)) + 2.*k*x(0)*std::cos(k*x(0)) + 2.*k*(x(0) - 1.)*std::cos(k*x(0)) + 2.*std::sin(k*x(0)))*std::cos(k*x(1)) + 1.,
          2.*k*k*x(0)*x(0)*std::sin(k*x(1))*std::cos(k*x(0)) - 2.*k*k*x(0)*std::sin(k*x(1))*std::cos(k*x(0)) + 8.*k*x(0)*std::sin(k*x(0))*std::sin(k*x(1)) - 4.*k*std::sin(k*x(0))*std::sin(k*x(1)) + k*std::sin(k*(x(0) - x(1))) - 6.*std::sin(k*x(1))*std::cos(k*x(0));
    return rv;
  };
};
#elif SOLN == 1
template<size_t k_int> // Dirichlet
struct Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  
  std::function<VectorRd(const VectorRd &)> u_exact = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << std::sin(k*x(0))*std::sin(k*x(0))*std::sin(k*x(1))*std::cos(k*x(1)),
          -std::sin(k*x(0))*std::cos(k*x(0))*std::sin(k*x(1))*std::sin(k*x(1));
    return rv;
  };
  std::function<double(const VectorRd &)> p_exact = [](const VectorRd &x)->double {
    return x(0) + std::cos(k*(x(0)-x(1))) - 0.5 - 2.*(1. - std::cos(k))/(k*k);
  };
  std::function<VectorRd(const VectorRd &)> f = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << k*k*std::sin(2.*k*x(1)) + k*k*std::sin(2.*k*(x(0) - x(1))) - k*k*std::sin(2.*k*(x(0) + x(1))) - k*std::sin(k*(x(0) - x(1))) + 1.,
          k*(-k*std::sin(2.*k*x(0)) + k*std::sin(2.*k*(x(0) - x(1))) + k*std::sin(2.*k*(x(0) + x(1))) + std::sin(k*(x(0) - x(1))));
    return rv;
  };
};
#elif SOLN == 2 or SOLN == 5
template<size_t k_int> // Neumann, p != 0
struct Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  
  std::function<VectorRd(const VectorRd &)> u_exact = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << (2.*k*x(0) - std::sin(2.*k*x(0)))*0.25*std::sin(k*x(1))*std::sin(k*x(1)) - k*0.125, 
            k*0.125 - (2.*k*x(1) - std::sin(2.*k*x(1)))*0.25*std::sin(k*x(0))*std::sin(k*x(0));
    return rv;
  };
  std::function<double(const VectorRd &)> p_exact = [](const VectorRd &x)->double {
    return sin(k*x(0))*sin(k*x(1))*x(0);
  };
  std::function<VectorRd(const VectorRd &)> f = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << k*k*0.5*(-2.*k*x(0)*std::cos(2.*k*x(1)) - std::sin(2.*k*x(0)) + std::sin(2.*k*(x(0) - x(1))) + std::sin(2.*k*(x(0) + x(1)))) + k*cos(k*x(0))*sin(k*x(1))*x(0) + sin(k*x(0))*sin(k*x(1)),
          k*k*0.5*(2.*k*x(1)*std::cos(2.*k*x(0)) + std::sin(2.*k*x(1)) + std::sin(2.*k*(x(0) - x(1))) - std::sin(2.*k*(x(0) + x(1)))) + k*cos(k*x(1))*sin(k*x(0))*x(0);
    return rv;
  };
};
#elif SOLN == 3
template<size_t k_int> // Neumann, p = 0
struct Sol_struct {
  constexpr static double k = 3.141592653589793238462643383279502884L*k_int;
  
  std::function<VectorRd(const VectorRd &)> u_exact = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << (2.*k*x(0) - std::sin(2.*k*x(0)))*0.25*std::sin(k*x(1))*std::sin(k*x(1)) - k*0.125, 
            k*0.125 - (2.*k*x(1) - std::sin(2.*k*x(1)))*0.25*std::sin(k*x(0))*std::sin(k*x(0));
    return rv;
  };
  std::function<double(const VectorRd &)> p_exact = [](const VectorRd &x)->double {
    return 0.;
  };
  std::function<VectorRd(const VectorRd &)> f = [](const VectorRd &x)->VectorRd {
    VectorRd rv;
    rv << k*k*0.5*(-2.*k*x(0)*std::cos(2.*k*x(1)) - std::sin(2.*k*x(0)) + std::sin(2.*k*(x(0) - x(1))) + std::sin(2.*k*(x(0) + x(1)))),
          k*k*0.5*(2.*k*x(1)*std::cos(2.*k*x(0)) + std::sin(2.*k*x(1)) + std::sin(2.*k*(x(0) - x(1))) - std::sin(2.*k*(x(0) + x(1))));
    return rv;
  };
};
#endif // solution selection

std::function<bool(const VectorRd &)> rightside = [](const VectorRd &x)->bool {return (x(0)+1e-6 > 1.);};

Sol_struct<2> exactsol;

template<size_t degree> 
void validate_Stokes() {
  std::vector<double> meshsize;
  std::vector<double> errorP;
  std::vector<double> errorU;
  std::vector<double> errorT;

 // Iterate over meshes
  for (auto mesh_file : mesh_files) {
    // Build the mesh
    MeshBuilder builder = MeshBuilder(mesh_file);
    std::unique_ptr<Mesh> mesh_ptr = builder.build_the_mesh();
    std::cout << "[main] Mesh size                 " << mesh_ptr->h_max() << std::endl;
    // Store the size of the mesh
    meshsize.emplace_back(mesh_ptr->h_max());

    // Create core 
    StokesProblem stokes(*mesh_ptr,degree);
    std::cout << "[main] StokesProblem constructed" << std::endl;
    
    // Setup boundary conditions
    #if SOLN < 2
    stokes.setup_Harmonics(Harmonics_premade::Pressure);
    stokes.setup_Dirichlet_everywhere(); // stokes.setup_dofsmap() automaticaly called
    #if SOLN == 0
    stokes.interpolate_boundary_value(exactsol.u_exact);
    #endif // SOLN == 0
    #elif SOLN < 5
    stokes.setup_Harmonics(Harmonics_premade::Velocity);
    #elif SOLN == 5
    stokes.setup_Harmonics(Harmonics_premade::None);
    stokes.set_Dirichlet_boundary(rightside);
    stokes.interpolate_boundary_value(exactsol.u_exact);
    #else
    static_assert(false,"Wrong value of SOLN");
    #endif // SOLN switch

    // Create problem and solve
    stokes.assemble_system(exactsol.f);
    stokes.compute();
    Eigen::VectorXd uhunk = stokes.solve();
    std::cout << "[main] System solved" << std::endl;
    #if SOLN < 2 or SOLN == 5
    Eigen::VectorXd uh = stokes.reinsertDirichlet(uhunk);
    #else
    Eigen::VectorXd uh = uhunk;
    #endif

    // Interpolate exact solution
    Eigen::VectorXd Iu = stokes.xnabla().interpolate(exactsol.u_exact) - uh.segment(0,stokes.xnabla().dimension());
    Eigen::VectorXd Ip = stokes.xsl().interpolate(exactsol.p_exact) - uh.segment(stokes.xnabla().dimension(),stokes.xsl().dimension());

    // Write solutions
    if (write_sols) {
      VtuWriter writer(mesh_ptr.get());
      std::cout << "[main] Writing solutions..." << std::flush;
      Eigen::VectorXd uh_vert = get_vertices_values(stokes.xnabla(),uh.head(stokes.xnabla().dimension()));
      Eigen::VectorXd ph_vert = get_vertices_values(stokes.xsl(),uh.segment(stokes.xnabla().dimension(),stokes.xsl().dimension()));
      Eigen::VectorXd u_vert = evaluate_vertices_values(exactsol.u_exact,stokes.stokescore());
      Eigen::VectorXd p_vert = evaluate_vertices_values(exactsol.p_exact,stokes.stokescore());
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<2> > uhx(uh_vert.data(), uh_vert.size()/2);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<2> > uhy(uh_vert.data() + 1, uh_vert.size()/2);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<2> > ux(u_vert.data(), u_vert.size()/2);
      Eigen::Map<Eigen::VectorXd,0,Eigen::InnerStride<2> > uy(u_vert.data() + 1, u_vert.size()/2);
      // write uh
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("uhx.vtu"),uhx,false);
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("uhy.vtu"),uhy,false);
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("ph.vtu"),ph_vert,false);
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("ux.vtu"),ux,false);
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("uy.vtu"),uy,false);
      writer.write_to_vtu(std::string("vtu/") + mesh_file.substr(mesh_file.find_last_of("/\\") + 1) + std::string("p.vtu"),p_vert,false);
      std::cout <<" Done" << std::endl;
    }
    double Errnorm = 0;
    Errnorm += Norm_H1p(stokes.xnabla(),stokes.xvl(),Iu);
    Errnorm += Norm_St(stokes.xnabla(),Iu);
    errorU.emplace_back(Errnorm);
    errorT.emplace_back(Errnorm);
    Errnorm = Norm_St(stokes.xsl(),Ip);
    errorP.emplace_back(Errnorm);
    errorT.back() += Errnorm;
  } // end for meshes
  
  std::cout << "Absolute   ErrorU   ErrorP   ErrorT" << std::endl;
  for (size_t i = 0; i < mesh_files.size();i++) {std::cout<<"          "<<FORMATD(2)<<errorU[i]<<errorP[i]<<errorT[i]<<std::endl;}
  std::cout << "Rate" << std::endl;
  for (size_t i = 1; i < mesh_files.size();i++) {std::cout<<"          "<<FORMATD(2)<<compute_rate(errorU,meshsize,i)<<compute_rate(errorP,meshsize,i)<<compute_rate(errorT,meshsize,i)<<std::endl;}
} // end validate_Stokes 

