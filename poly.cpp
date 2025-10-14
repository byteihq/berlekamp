#include "poly.h"

Poly& Poly::operator=(Poly&& rhs) noexcept {
    if (this != &rhs)
    {
        M = rhs.M;
        data = std::move(rhs.data);
        is_normalized = rhs.is_normalized;
    }

    return *this;
}


void Poly::normalize() {
    while (!data.empty() && modnorm(data.back(), M) == 0)
        data.pop_back();

    for (size_t i = 0; i < data.size(); ++i)
        data[i] = modnorm(data[i], M);

    is_normalized = true;
}

// extended gcd for integers: returns (g, x, y) such that a*x + b*y = g
std::tuple<int64_t, int64_t, int64_t> egcd(uint64_t a, uint64_t b) {
    if (b == 0) return { a, 1, 0 };
    auto [g, x1, y1] = egcd(b, a % b);
    int64_t x = y1;
    int64_t y = x1 - (a / b) * y1;
    return { g, x, y };
}

uint64_t inv_mod(uint64_t a, uint64_t P) {
    a = modnorm(a, P);
    assert(a != 0);

    auto [g, x, y] = egcd(a, P);
    assert(g == 1);

    for (; x < 0; x += P);
    return modnorm(x, P);
}

Poly poly_add(const Poly& A, const Poly& B) {
    assert(A.deg() == B.deg());
    assert(A.getMod() == B.getMod());

    Poly C(A.deg() + 1, A.getMod());
    for (size_t i = 0; i < C.deg() + 1; ++i)
        C[i] = A[i] + B[i];

    return C;
}

Poly poly_sub(const Poly& A, const Poly& B) {
    assert(A.deg() == B.deg());
    assert(A.getMod() == B.getMod());

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
    if (A.isZero() || B.isZero())
        return Poly();

    assert(A.getMod() == B.getMod());

    size_t na = A.deg(), nb = B.deg();
    Poly C(na + nb + 1, A.getMod());
    for (size_t i = 0; i <= na; ++i) {
        for (size_t j = 0; j <= nb; ++j) {
            C[i + j] = C[i + j] + modnorm(A[i] * B[j], A.getMod());
        }
    }
    C.normalize();

    return C;
}

// polynomial long division: returns quotient Q and remainder R so that A = B*Q + R
std::pair<Poly, Poly> poly_divmod(const Poly& A, const Poly& B) {
    assert(!B.isZero());
    assert(A.getMod() == B.getMod());

    if (A.deg() < B.deg())
        return { Poly(), A };

    Poly Q(A.deg() - B.deg() + 1, A.getMod());
    Poly R = A;
    Poly::value_t invLeadB = inv_mod(B[B.deg()], B.getMod());

    while (!R.isZero() && R.deg() >= B.deg()) {
        size_t d = R.deg() - B.deg();
        Poly::value_t coef = modnorm(R[R.deg()] * invLeadB, R.getMod());
        Q[d] = coef;

        for (size_t i = 0; i <= B.deg(); ++i) {
            size_t idx = i + d;
            if (const auto rhs = (coef * B[i]) % R.getMod(); R[idx] < rhs)
            {
                R[idx] = modnorm(R.getMod() + R[idx] - rhs, R.getMod());
            }
            else
            {
                R[idx] = modnorm(R[idx] - rhs, R.getMod());
            }
        }
        R.normalize();
    }

    Q.normalize();
    R.normalize();
    return { Q, R };
}

Poly poly_mod(const Poly& A, const Poly& M) {
    return poly_divmod(A, M).second;
}

Poly poly_gcd(Poly A, Poly B) {
    assert(A.isNormalized() && B.isNormalized());

    if (A.isZero())
        return B;
    if (B.isZero())
        return A;

    while (!B.isZero()) {
        Poly R = poly_divmod(A, B).second;
        A = std::move(B);
        B = std::move(R);
    }
    // make monic
    A.normalize();
    if (!A.isZero()) {
        Poly::value_t invLead = inv_mod(A[A.deg()], A.getMod());
        A = poly_scalar_mul(A, invLead);
    }
    return A;
}

// polynomial exponentiation modulo M
Poly poly_powmod(const Poly& base, long long exp, const Poly& M) {
    assert(!M.isZero());

    Poly res{ {1}, M.getMod() };
    Poly b = base;
    while (exp > 0) {
        if (exp & 1LL) {
            res = poly_mod(poly_mul(res, b), M);
        }
        b = poly_mod(poly_mul(b, b), M);
        exp >>= 1LL;
    }

    return res;
}

// p-th root for polynomial when all exponents are multiples of p
Poly poly_pth_root(const Poly& A) {
    if (A.isZero())
        return Poly();

    Poly R((A.deg() / A.getMod()) + 1, A.getMod());
    for (int i = 0; i <= A.deg(); ++i) {
        //if (A.coeff(i) != 0 && (i % P) != 0) throw std::runtime_error("pth root: degrees not multiples of p");
        if (i % R.getMod() == 0) {
            R[i / R.getMod()] = A[i];
        }
    }

    return R;
}
