#include<type_traits>

class A {};
template<typename T> class B {};
template<typename T> class C : public B<T> {};

template<typename T> int lookup(T &x){return 1;}
template<typename T> int lookup(B<T> &x){return 2;}
template<typename T,template<typename> typename S> requires std::is_base_of<B<T>,S<T>>::value int lookup(S<T> &x){return 3;}


int main() {
  A a;
  B<A> b;
  C<A> c;
  return lookup(a)+lookup(b)+lookup(c);
}
