#ifndef TESTFUNCTION_HPP
#define TESTFUNCTION_HPP

#include <basis.hpp>
#include <random>
#include <boost/math/differentiation/autodiff.hpp>

// Workarround for https://github.com/boostorg/math/issues/445
template <typename X> X powfx(const X &x, size_t n) 
{X rv = 1; for(size_t i = 0; i < n; i++) {rv *= x;} return rv;}

using namespace boost::math::differentiation;

namespace HArDCore2D {
  
  static std::mt19937 gen(6);

  enum Initialization {
    Zero,
    Default,
    Random
  };

  [[maybe_unused]] static void fill_random_vector (Eigen::VectorXd &inout) {
    std::uniform_real_distribution<> dis(-10.,10.);
    for (long i = 0; i < inout.size();i++) {
      inout[i] = dis(gen);
    }
    return; 
  }

  // Create a global polynomial of total degree k and provide its derivatives
  template <size_t k>
  class PolyTest {
    public:
      PolyTest(Initialization initmode = Initialization::Zero,double _scale = 1.) : scale(_scale) {
        coefficients.resize(nb_monomial,0.);
        switch(initmode) {
          case (Initialization::Random) :
            {
              std::uniform_real_distribution<> dis(1.,3.);
              for (size_t i = 0; i < coefficients.size();i++) {
                coefficients[i] = dis(gen);
              }
            }
            break;
          case (Initialization::Default):
          case (Initialization::Zero):
           ; 
        }
        powers = MonomialPowers<Cell>::compute(k);
      }

      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0) {
        auto const variables = make_ftuple<double, k+2, k+2>(x(0),x(1));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const v = f(X,Y);
        return v.derivative(diffx,diffy);
      }

      std::vector<double> coefficients;
      double scale;
      template <typename X, typename Y> // promote from boost
        promote<X,Y> f(const X &x,const Y &y) {
          promote<X,Y> rv = 0;
          for (size_t i = 0;i < powers.size(); i++) {
            rv += coefficients[i]*powfx(x,powers[i](0))*powfx(y,powers[i](1));
          }
          return scale*rv;
        }
    private:
      const static size_t nb_monomial = (k + 1) * (k + 2) / 2;
      std::vector<VectorZd> powers;
  };

  // Create a global exponential function and provide its derivatives
  template <size_t k>
  class TrigTest {
    public:
      TrigTest(Initialization initmode = Initialization::Zero,double _scale = 1.) : scale(_scale) {
        for (size_t ix = 1; ix < 2;ix++) {
          for (size_t iy = 1; iy < 2;iy++) {
            for (size_t jx = 1; jx < 2;jx++) {
              for (size_t jy = 1; jy < 2;jy++) {
                for (size_t kx = 1; kx < 2;kx++) {
                  for (size_t ky = 1; ky < 2;ky++) {
                    Eigen::VectorXi v(6);
                    v << ix, iy, jx, jy, kx, ky;
                    powers.push_back(v);
                  }
                }
              }
            }
          }
        }
        nb_elements = powers.size();
        coefficients.resize(nb_elements);
        switch(initmode) {
          case (Initialization::Random) :
            {
              std::uniform_real_distribution<> dis(1.,3.);
              for (size_t i = 0; i < coefficients.size(); i++) {
                coefficients[i] = Eigen::Vector2d(dis(gen),dis(gen));
              }
            }
            break;
          case (Initialization::Default):
          case (Initialization::Zero):
            for (size_t i = 0; i < coefficients.size(); i++) {
              coefficients[i] = Eigen::Vector2d::Zero();
            }
        }
      }

      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0) {
        auto const variables = make_ftuple<double, k+2, k+2>(x(0),x(1));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const v = f(X,Y);
        return v.derivative(diffx,diffy);
      }

      std::vector<Eigen::Vector2d> coefficients;
      template <typename X, typename Y> // promote from boost
        promote<X,Y> f(const X &x,const Y &y) {
          promote<X,Y> rv = 0.;
          for (size_t i = 0;i < powers.size(); i++) {
            rv += coefficients[i](0)*cos(coefficients[i](1)*powfx(x,powers[i](0)))*
                  coefficients[i](0)*cos(coefficients[i](1)*powfx(y,powers[i](1))); 
            rv += coefficients[i](0)*sin(coefficients[i](1)*powfx(x,powers[i](2)))*
                  coefficients[i](0)*sin(coefficients[i](1)*powfx(y,powers[i](3)));
            rv += coefficients[i](0)*exp(coefficients[i](1)*powfx(x,powers[i](4)))*
                  coefficients[i](0)*exp(coefficients[i](1)*powfx(y,powers[i](5))); 
          }
          return scale*rv;
        }
        double scale;
    private:
      size_t nb_elements;
      std::vector<Eigen::VectorXi> powers;
  };
  template<typename BaseFamily,size_t k>
  class ZeroXWrapper {
    public:
      ZeroXWrapper( BaseFamily *_base) :base(_base) {;}
      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0) {
        auto const variables = make_ftuple<double, k+2, k+2>(x(0),x(1));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const v = f(X,Y);
        return v.derivative(diffx,diffy);
      }
      template <typename X, typename Y> // promote from boost
      promote<X,Y> f(const X &x, const Y &y) {
        return base->f(x,y)*x*(1. - x);
      }
    private:
      BaseFamily *base;
  };
  template<typename BaseFamily,size_t k>
  class ZeroYWrapper {
    public:
      ZeroYWrapper( BaseFamily *_base) :base(_base) {;}
      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0) {
        auto const variables = make_ftuple<double, k+2, k+2>(x(0),x(1));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const v = f(X,Y);
        return v.derivative(diffx,diffy);
      }
      template <typename X, typename Y> // promote from boost
      promote<X,Y> f(const X &x, const Y &y) {
        return base->f(x,y)*y*(1. - y);
      }
    private:
      BaseFamily *base;
  };
  // Function such that Na f .n = 0 on [0,1]x[0,1]
  template <size_t k>
  class TrigTestdZero {
    public:
      TrigTestdZero(Initialization initmode = Initialization::Zero) {
        switch(initmode) {
          case (Initialization::Random) :
            {
              std::uniform_int_distribution<> dis(1,6);
              for (int i = 0; i < coefficients.size(); i++) {
                coefficients[i] = dis(gen);
              }
            }
            break;
          case (Initialization::Default):
          case (Initialization::Zero):
            for (int i = 0; i < coefficients.size(); i++) {
              coefficients[i] = 0;
            }
        }
      }

      double evaluate(const VectorRd &x, size_t diffx = 0, size_t diffy = 0) {
        auto const variables = make_ftuple<double, k+2, k+2>(x(0),x(1));
        auto const& X = std::get<0>(variables);
        auto const& Y = std::get<1>(variables);
        auto const v = f(X,Y);
        return v.derivative(diffx,diffy);
      }

      Eigen::Vector2i coefficients;
      template <typename X, typename Y> // promote from boost
        promote<X,Y> f(const X &x,const Y &y) {
          return cos(coefficients(0)*boost::math::constants::pi<double>()*x)*cos(coefficients(1)*boost::math::constants::pi<double>()*y);
        }
  };

  [[maybe_unused]] static int m_switch = 0;

  [[maybe_unused]] static class Errandswitch {
    public:
    Errandswitch operator++(int) {Errandswitch temp(*this); m_errcount++; m_switch++; return temp;}
    operator int() const {return m_errcount;}
    private:
      int m_errcount = 0;
    } nb_errors;

  [[maybe_unused]] static struct endls_s {} endls;
    std::ostream& operator<<(std::ostream& out, endls_s) {
    if (m_switch) {
      m_switch = 0;
      return out << "\033[1;31m Unexpected\033[0m" << std::endl;
    } else {
      return out << std::endl;
    }
  }

  static constexpr double threshold = 1e-9;

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

} // end of namespace
#endif

