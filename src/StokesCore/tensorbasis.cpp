#include "tensorbasis.hpp"

namespace HArDCore2D {

//------------------------------------------------------------------------------
// Basis for Rb^{c,k}(T)
//------------------------------------------------------------------------------

RolybComplBasisCell::RolybComplBasisCell(const Cell &T, size_t degree)
  : m_degree(degree),
    m_xT(T.center_mass()),
    m_hT(T.diam()) {
    // Monomial powers for P^{k-2}(T)
    if (m_degree >= 2){
      m_powers = MonomialPowers<Cell>::compute(m_degree-2);
    } else if (m_degree == 0) {
      std::cout << "Attempting to construct RbckT with degree 0, stopping" << std::endl;
      exit(1);
    }
}

RolybComplBasisCell::FunctionValue RolybComplBasisCell::function(size_t i, const VectorRd &x) const 
{
  VectorRd y = _coordinate_transform(x);
  const VectorZd &powers = m_powers[i];
  RolybComplBasisCell::FunctionValue rv;
  rv << -y(0)*y(1), -y(1)*y(1), y(0)*y(0), y(0)*y(1);
  return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) * rv;
}

RolybComplBasisCell::DivergenceValue RolybComplBasisCell::divergence(size_t i, const VectorRd &x) const 
{
  VectorRd y = _coordinate_transform(x);
  const VectorZd &powers = m_powers[i];
  RolybComplBasisCell::DivergenceValue rv;
  rv << -std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)+1),
      std::pow(y(0), powers(0)+1) * std::pow(y(1), powers(1));
  return rv * (powers(0)+powers(1)+3)/ m_hT;
}

//------------------------------------------------------------------------------
// Basis for Rb^{k}(T)
//------------------------------------------------------------------------------

RolybBasisCell::RolybBasisCell(const Cell &T, size_t degree)
  : m_degree(degree),
    m_xT(T.center_mass()),
    m_hT(T.diam()) {
    // Monomial powers for P^{k}(T)
    m_powers = MonomialPowers<Cell>::compute(m_degree);
}

RolybBasisCell::FunctionValue RolybBasisCell::function(size_t i, const VectorRd &x) const 
{
  i++; // shift i
  VectorRd y = _coordinate_transform(x);
  const VectorZd &powers = m_powers[i];
  RolybBasisCell::FunctionValue rv;
  if (powers(0) == 0) {
    rv << 0.,
      0.,
      powers(1) * std::pow(y(0), powers(0)+1) * std::pow(y(1), powers(1)-1),
      powers(1) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1));
  } else if (powers(1) == 0) {
    rv << powers(0) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)),
      powers(0) * std::pow(y(0), powers(0)-1) * std::pow(y(1), powers(1)+1),
      0.,
      0.;
  } else {
    rv << powers(0) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)),
      powers(0) * std::pow(y(0), powers(0)-1) * std::pow(y(1), powers(1)+1),
      powers(1) * std::pow(y(0), powers(0)+1) * std::pow(y(1), powers(1)-1),
      powers(1) * std::pow(y(0), powers(0)) * std::pow(y(1), powers(1));
  }
  return rv/(1. + powers(0) + powers(1));
}

RolybBasisCell::DivergenceValue RolybBasisCell::divergence(size_t i, const VectorRd &x) const 
{
  i++;
  VectorRd y = _coordinate_transform(x);
  const VectorZd &powers = m_powers[i];
  RolybBasisCell::DivergenceValue rv;
  if (powers(0) == 0) {
    rv << 0.,powers(1)*std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)-1);
  } else if (powers(1) == 0) {
    rv << powers(0)*std::pow(y(0), powers(0)-1) * std::pow(y(1), powers(1)),0.;
  } else {
    rv << powers(0)*std::pow(y(0), powers(0)-1) * std::pow(y(1), powers(1)),
      powers(1)*std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)-1);
  }
  return rv / m_hT;
}

RolybBasisCell::TraceValue RolybBasisCell::trace(size_t i, const VectorRd &x) const 
{
  i++;
  VectorRd y = _coordinate_transform(x);
  const VectorZd &powers = m_powers[i];
  return std::pow(y(0), powers(0)) * std::pow(y(1), powers(1)) *  (powers(0) + powers(1))/(1. + powers(0) + powers(1)) ;
}


} // end of namespace
