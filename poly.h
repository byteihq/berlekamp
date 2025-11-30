#pragma once

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cassert>

class Poly {
public:
    using value_t = uint64_t;

public:
    Poly() : M(0) {}
    Poly(size_t n, value_t mod) : M(mod), data(n) {}
    Poly(const std::vector<value_t>& v, value_t mod) : M(mod), data(v) {}
    Poly(std::initializer_list<value_t> il, value_t mod) : M(mod), data(il) {}

    Poly(const Poly& rhs) : M(rhs.M), data(rhs.data), is_normalized(rhs.is_normalized) {}
    Poly(Poly&& rhs) noexcept : M(rhs.M), data(std::move(rhs.data)), is_normalized(rhs.is_normalized) {}
    Poly& operator=(Poly&& rhs) noexcept;

    [[msvc::forceinline]] size_t deg() const noexcept {
        assert(!isZero());
        return data.size() - 1;
    }

    [[msvc::forceinline]] bool isZero() const noexcept { return data.empty(); }
    [[msvc::forceinline]] bool isNormalized() const noexcept { return is_normalized; }
    [[msvc::forceinline]] value_t getMod() const noexcept { return M; }

    [[msvc::forceinline]] value_t& operator[](size_t i) { return data[i]; }
    [[msvc::forceinline]] const value_t& operator[](size_t i) const { return data[i]; }

    [[msvc::forceinline]] void push(value_t v) { data.push_back(v); }

    void normalize();
    [[msvc::forceinline]] void setNormalize() { is_normalized = true; }

private:
    value_t M;
    std::vector<value_t> data;
    bool is_normalized{ false };
};

Poly poly_add(const Poly& A, const Poly& B);
Poly poly_sub(const Poly& A, const Poly& B);
Poly poly_scalar_mul(const Poly& A, Poly::value_t k);
Poly poly_mul(const Poly& A, const Poly& B);
std::pair<Poly, Poly> poly_divmod(const Poly& A, const Poly& B);
Poly poly_mod(const Poly& A, const Poly& M);
Poly poly_gcd(Poly A, Poly B);
Poly poly_powmod(const Poly& base, long long exp, const Poly& M);
Poly poly_pth_root(const Poly& A);
std::tuple<int64_t, int64_t, int64_t> egcd(uint64_t a, uint64_t b);
uint64_t inv_mod(uint64_t a, uint64_t P);

inline uint64_t modnorm(uint64_t a, uint64_t P) { return a % P; }
