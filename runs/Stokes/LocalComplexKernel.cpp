
#include <mesh_builder.hpp>
#include <stokescore.hpp>
#include <xcurlstokes.hpp>
#include <xnabla.hpp>
#include <xvl.hpp>
#include "testfunction.hpp"

#include <parallel_for.hpp>

#define FORMAT(W)                                                       \
  std::setiosflags(std::ios_base::left) << std::setw(W) << std::setfill(' ')

using namespace HArDCore2D;

const std::string mesh_file = "../typ2_meshes/" "hexa1_1.typ2";

Eigen::SparseMatrix<double> assemble_system_xcurl(const XCurlStokes &xcurl);
Eigen::SparseMatrix<double> assemble_system_xnabla(const XCurlStokes &xcurl, const XNabla &xnabla, const XSL &xsl);

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

  Eigen::SparseMatrix<double> SystemCurl = assemble_system_xcurl(xcurl);
  std::cout << "[main] Curl system assembled" << std::endl;
  SystemCurl.makeCompressed();
  Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> solver;
  solver.compute(SystemCurl);
  std::cout << "[main] solver Curl system : " << solver.info() << std::endl;
  std::cout << "Kernel of dimension 1 expected" << std::endl;
  std::cout << "Dimension : " << xcurl.dimension() << " rank : " << solver.rank() << std::endl;

  XSL xsl(stokes_core);
  Eigen::SparseMatrix<double> SystemDiv = assemble_system_xnabla(xcurl,xnabla,xsl); // The system is the mixed formulation for the Hodge-Laplace equations, its kernel is the space of harmonics 1-form.
  std::cout << "[main] Div system assembled" << std::endl;
  SystemDiv.makeCompressed();
  Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<Eigen::SparseMatrix<double>::StorageIndex>> solver2;
  solver2.compute(SystemDiv);
  std::cout << "[main] solver Div system : " << solver2.info() << std::endl;
  std::cout << "Kernel of dimension 0 expected" << std::endl;
  std::cout << "Dimension : " << xcurl.dimension() + xnabla.dimension() << " rank : " << solver2.rank() << std::endl;

  return 0;
}


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

Eigen::SparseMatrix<double> assemble_system_xcurl(const XCurlStokes &xcurl) {
  std::vector<Eigen::MatrixXd> ALoc;
  ALoc.resize(xcurl.mesh().n_cells());
  std::function<void(size_t start, size_t end)> assemble_local = [&xcurl,&ALoc](size_t start, size_t end)->void {
    for (size_t iT = start; iT < end; iT++) {
      Eigen::MatrixXd loc = xcurl.cellOperators(iT).ucurl.transpose()*xcurl.cellOperators(iT).ucurl;
      ALoc[iT] = loc;
    }
  };
  parallel_for(xcurl.mesh().n_cells(),assemble_local,true); // Assemble all local contributions

  std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [&xcurl,&ALoc](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
    for (size_t iT = start; iT < end; iT++) {
      for (size_t i = 0; i < xcurl.dimensionCell(iT); i++) {
        size_t gi = L2Gdofs(xcurl,iT,i);
        for (size_t j = 0; j < xcurl.dimensionCell(iT); j++) {
          size_t gj = L2Gdofs(xcurl,iT,j);
          triplets->emplace_back(gi,gj,ALoc[iT](i,j));
        }
      }
    }
  };

  return parallel_assembly_system(xcurl.mesh().n_cells(),xcurl.dimension(),batch_local_assembly,true).first;
}

Eigen::SparseMatrix<double> assemble_system_xnabla(const XCurlStokes &xcurl, const XNabla &xnabla, const XSL &xsl) {
  std::vector<Eigen::MatrixXd> ALocCC(xnabla.mesh().n_cells());
  std::vector<Eigen::MatrixXd> ALocCD(xnabla.mesh().n_cells());
  std::vector<Eigen::MatrixXd> ALocDC(xnabla.mesh().n_cells());
  std::vector<Eigen::MatrixXd> ALocDD(xnabla.mesh().n_cells());
  std::function<void(size_t start, size_t end)> assemble_local = [&xcurl,&xnabla,&xsl,&ALocCC,&ALocCD,&ALocDC,&ALocDD](size_t start, size_t end)->void {
    for (size_t iT = start; iT < end; iT++) {
      Eigen::MatrixXd locCC = xcurl.computeL2Product(iT);
      ALocCC[iT] = locCC;
      Eigen::MatrixXd locCD = -xcurl.cellOperators(iT).ucurl.transpose()*xnabla.computeL2Product(iT);
      ALocCD[iT] = locCD;
      Eigen::MatrixXd locDC = xnabla.computeL2Product(iT)*xcurl.cellOperators(iT).ucurl;
      ALocDC[iT] = locDC;
      Eigen::MatrixXd locDD = xnabla.cellOperators(iT).divergence.transpose()*xsl.computeL2Product(iT)*xnabla.cellOperators(iT).divergence; 
      ALocDD[iT] = locDD;
    }
  };
  parallel_for(xnabla.mesh().n_cells(),assemble_local,true); // Assemble all local contributions

  std::function<void(size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)> batch_local_assembly = [&xcurl,&xnabla,&ALocCC,&ALocCD,&ALocDC,&ALocDD](size_t start, size_t end, std::list<Eigen::Triplet<double>> * triplets, Eigen::VectorXd * rhs)->void {
    for (size_t iT = start; iT < end; iT++) {
      for (size_t i = 0; i < xcurl.dimensionCell(iT); i++) {
        size_t gi = L2Gdofs(xcurl,iT,i);
        for (size_t j = 0; j < xcurl.dimensionCell(iT); j++) {
          size_t gj = L2Gdofs(xcurl,iT,j);
          triplets->emplace_back(gi,gj,ALocCC[iT](i,j));
        }
        for (size_t j = 0; j < xnabla.dimensionCell(iT); j++) {
          size_t gj = xcurl.dimension() + L2Gdofs(xnabla,iT,j);
          triplets->emplace_back(gi,gj,ALocCD[iT](i,j));
        }
      }
      for (size_t i = 0; i < xnabla.dimensionCell(iT); i++) {
        size_t gi = xcurl.dimension() + L2Gdofs(xnabla,iT,i);
        for (size_t j = 0; j < xcurl.dimensionCell(iT); j++) {
          size_t gj = L2Gdofs(xcurl,iT,j);
          triplets->emplace_back(gi,gj,ALocDC[iT](i,j));
        }
        for (size_t j = 0; j < xnabla.dimensionCell(iT); j++) {
          size_t gj = xcurl.dimension() + L2Gdofs(xnabla,iT,j);
          triplets->emplace_back(gi,gj,ALocDD[iT](i,j));
        }
      }
    }
  };

  return parallel_assembly_system(xnabla.mesh().n_cells(),xcurl.dimension()+xnabla.dimension(),batch_local_assembly,true).first;
}


