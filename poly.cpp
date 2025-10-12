#pragma once

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cassert>

inline uint64_t modnorm(uint64_t a, uint64_t P) {
    return a % P;
}

// extended gcd for integers: returns (g, x, y) such that a*x + b*y = g
static std::tuple<uint64_t, uint64_t, uint64_t> egcd(uint64_t a, uint64_t b) {
    if (b == 0) return { a, 1, 0 };
    auto [g, x1, y1] = egcd(b, a % b);
    uint64_t x = y1;
    uint64_t y = x1 - (a / b) * y1;
    return { g, x, y };
}

uint64_t inv_mod(uint64_t a, uint64_t P) {
    a = modnorm(a, P);
    assert(a != 0);

    auto [g, x, y] = egcd(a, P);
    assert(g == 1);

    return modnorm(x, P);
}

class Poly {
public:
    using value_t = uint64_t;

public:
    Poly(size_t n, value_t mod) : M(mod), data(n) {}
    Poly(std::initializer_list<value_t> il, value_t mod) : M(mod), data(il) {}

    void normalize() {
        if (is_normalized)
            return;

        for (size_t i = 0; i < data.size(); ++i)
            data[i] = modnorm(data[i], M);

        data.erase(std::remove(data.begin(), data.end(), 0), data.end());
        is_normalized = true;
    }

    size_t deg() const noexcept {
        assert(!isZero());
        return data.size() - 1;
    }

    bool isZero() const noexcept { return data.empty(); }
    bool isNormalized() const noexcept { return is_normalized; }
    value_t getMod() const noexcept { return M; }

    value_t& operator[](size_t i) { return data[i]; }
    const value_t& operator[](size_t i) const { return data[i]; }

    value_t coeff(size_t i) const {
        assert(i < a.size());
        assert(is_normalized);

        return data[i];
    }

private:
    const value_t M;
    std::vector<value_t> data;
    bool is_normalized{ false };
};

Poly poly_add(const Poly& A, const Poly& B) {
    assert(A.deg() == B.deg());

    Poly C(A.deg() + 1, A.getMod());
    for (size_t i = 0; i < C.deg() + 1; ++i)
        C[i] = A[i] + B[i];

    return C;
}

Poly poly_sub(const Poly& A, const Poly& B) {
    assert(A.deg() == B.deg());

    Poly C(A.deg() + 1, A.getMod());
    for (size_t i = 0; i < C.deg() + 1; ++i)
    {
        if (A[i] < B[i])
        {
            C[i] = (A[i] + A.getMod()) - B[i];
        }
        else
        {
            C[i] = A[i] - B[i];
        }
    }

    return C;
}

Poly poly_scalar_mul(const Poly& A, Poly::value_t k) {
    Poly C(A.deg() + 1, A.getMod());

    for (size_t i = 0; i < C.deg() + 1; ++i)
        C[i] = A[i] * k;

    return C;
}

Poly poly_mul(const Poly& A, const Poly& B) {
    if (A.isZero() || B.isZero()) return Poly();
    size_t na = A.deg(), nb = B.deg();
    Poly C(na + nb + 1);
    for (size_t i = 0; i <= na; ++i) {
        for (size_t j = 0; j <= nb; ++j) {
            int64_t prod = A.coeff(i) * B.coeff(j);
            C.a[i + j] = modnorm(C.a[i + j] + (prod % P));
        }
    }
    C.normalize();
    return C;
}

// polynomial long division: returns quotient Q and remainder R so that A = B*Q + R
std::pair<Poly, Poly> poly_divmod(Poly A, Poly B) {
    A.normalize(); B.normalize();
    if (B.isZero()) throw std::runtime_error("division by zero polynomial");
    if (A.deg() < B.deg()) return { Poly(), A };
    Poly Q(A.deg() - B.deg() + 1);
    Poly R = A;
    poly_obj_t invLeadB = inv_mod(B.coeff(B.deg()));
    while (!R.isZero() && R.deg() >= B.deg()) {
        size_t d = R.deg() - B.deg();
        poly_obj_t coef = modnorm(R.coeff(R.deg()) * invLeadB);
        Q.a[d] = coef;
        // subtract coef * x^d * B from R
        for (size_t i = 0; i <= B.deg(); ++i) {
            size_t idx = i + d;
            R.a[idx] = modnorm(R.a[idx] - coef * B.coeff(i));
        }
        R.normalize();
    }
    Q.normalize(); R.normalize();
    return { Q, R };
}

Poly poly_mod(const Poly& A, const Poly& M) {
    return poly_divmod(A, M).second;
}

Poly poly_gcd(Poly A, Poly B) {
    A.normalize(); B.normalize();
    if (A.isZero()) return B;
    if (B.isZero()) return A;
    while (!B.isZero()) {
        Poly R = poly_divmod(A, B).second;
        A = B; B = R;
    }
    // make monic
    A.normalize();
    if (!A.isZero()) {
        poly_obj_t invLead = inv_mod(A.coeff(A.deg()));
        A = poly_scalar_mul(A, invLead);
    }
    return A;
}

// polynomial exponentiation modulo M
Poly poly_powmod(Poly base, long long exp, const Poly& M) {
    base.normalize();
    Poly res; res.a = { 1 }; // 1
    if (M.isZero()) throw std::runtime_error("modulus polynomial is zero");
    Poly b = base;
    while (exp > 0) {
        if (exp & 1LL) {
            res = poly_mod(poly_mul(res, b), M);
        }
        b = poly_mod(poly_mul(b, b), M);
        exp >>= 1LL;
    }
    res.normalize();
    return res;
}

// p-th root for polynomial when all exponents are multiples of p
Poly poly_pth_root(const Poly& A) {
    if (A.isZero()) return Poly();
    Poly R((A.deg() / P) + 1);
    for (int i = 0; i <= A.deg(); ++i) {
        if (A.coeff(i) != 0 && (i % P) != 0) throw std::runtime_error("pth root: degrees not multiples of p");
        if (i % P == 0) {
            R.a[i / P] = A.coeff(i);
        }
    }
    R.normalize();
    return R;
}
