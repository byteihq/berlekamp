#include <iostream>
#include <vector>
#include <string>
#include <cassert>

using poly_obj_t = uint64_t;

static poly_obj_t P = 2; // глобальный модуль поля (prime)


// Gaussian elimination to compute RREF of matrix (rows x cols) over GF(P)
// it modifies matrix to RREF and returns pivot_col_for_row (size rows) with -1 for zero rows
std::vector<int> matrix_rref(std::vector<std::vector<poly_obj_t>>& M) {
    size_t rows = M.size();
    if (rows == 0) return {};
    size_t cols = M[0].size();
    std::vector<int> pivot_col_for_row(rows, -1);
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c) {
        // find pivot
        int sel = -1;
        for (size_t i = r; i < rows; ++i) if (M[i][c] != 0) { sel = (int)i; break; }
        if (sel == -1) continue;
        swap(M[r], M[sel]);
        poly_obj_t inv = inv_mod(M[r][c]);
        // scale row r to make pivot 1
        for (size_t j = c; j < cols; ++j) M[r][j] = modnorm(M[r][j] * inv);
        // eliminate other rows
        for (size_t i = 0; i < rows; ++i) if (i != r && M[i][c] != 0) {
            poly_obj_t factor = M[i][c];
            for (size_t j = c; j < cols; ++j) {
                M[i][j] = modnorm(M[i][j] - factor * M[r][j]);
            }
        }
        pivot_col_for_row[r] = (int)c;
        ++r;
    }
    return pivot_col_for_row;
}

std::vector<Poly> berlekamp(const Poly& F) {
    std::cout << "[berlekamp] factoring (squarefree) " << F.str() << "\n";
    std::vector<Poly> result;
    if (F.isZero()) return result;
    if (F.deg() <= 0) return result;
    if (F.deg() == 1) { result.push_back(F); return result; }
    size_t n = F.deg();

    // === 1. Построение матрицы Q ===
    std::cout << "[berlekamp] building Q matrix (size " << n << ")...\n";
    Poly X; X.a = { 0,1 }; X.normalize();
    Poly xp = poly_powmod(X, P, F);

    std::vector<std::vector<poly_obj_t>> Q(n, std::vector<poly_obj_t>(n, 0));
    for (size_t j = 0; j < n; ++j) {
        Poly col = poly_powmod(xp, j, F); // x^(p*j) mod F
        for (size_t i = 0; i < col.a.size(); ++i) Q[i][j] = col.coeff(i);
    }

    // Q - I
    for (size_t i = 0; i < n; ++i) Q[i][i] = modnorm(Q[i][i] - 1);

    std::cout << "[berlekamp] matrix (Q - I):\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) std::cout << Q[i][j] << ' ';
        std::cout << '\n';
    }

    // === 2. Базис ядра ===
    auto Qcopy = Q;
    auto piv = matrix_rref(Qcopy);

    std::vector<int> pivot_of_col(n, -1);
    for (int r = 0; r < (int)piv.size(); ++r)
        if (piv[r] != -1) pivot_of_col[piv[r]] = r;

    std::vector<int> free_cols;
    for (int c = 0; c < n; ++c)
        if (pivot_of_col[c] == -1) free_cols.push_back(c);

    std::cout << "[berlekamp] nullspace dimension = " << free_cols.size() << "\n";

    std::vector<Poly> basis;
    for (int idx = 0; idx < (int)free_cols.size(); ++idx) {
        int fc = free_cols[idx];
        std::vector<poly_obj_t> vec(n, 0);
        vec[fc] = 1;
        for (int r = 0; r < (int)piv.size(); ++r) {
            int c = piv[r];
            if (c == -1) continue;
            vec[c] = modnorm(P - Qcopy[r][fc]);
        }
        Poly b; b.a = vec; b.normalize();
        std::cout << "[berlekamp] basis[" << idx << "] = " << b.str() << "\n";
        basis.push_back(b);
    }

    if (basis.empty()) {
        std::cout << "[berlekamp] nullspace trivial -> irreducible: " << F.str() << "\n";
        result.push_back(F);
        return result;
    }
    if ((int)basis.size() == 1 && basis[0].deg() == 0) {
        std::cout << "[berlekamp] nullspace contains only constants -> irreducible\n";
        result.push_back(F);
        return result;
    }

    // === 3. Перебор базиса и констант ===
    for (size_t bi = 0; bi < basis.size(); ++bi) {
        const Poly& v = basis[bi];
        if (v.deg() == 0) continue; // пропускаем константу
        std::cout << "[berlekamp] trying basis vector: " << v.str() << "\n";
        for (int c = 0; c < P; ++c) {
            Poly h = v;
            if (h.a.empty()) h.a = { 0 };
            h.a[0] = modnorm(h.a[0] - c);
            h.normalize();

            Poly g = poly_gcd(F, h);
            if (!g.isZero() && g.deg() >= 1 && g.deg() < F.deg()) {
                std::cout << "[berlekamp] non-trivial factor found: " << g.str()
                    << " (with c=" << c << ")\n";
                Poly f1 = g;
                Poly f2 = poly_divmod(F, f1).first;

                auto r1 = berlekamp(f1);
                auto r2 = berlekamp(f2);
                result.insert(result.end(), r1.begin(), r1.end());
                result.insert(result.end(), r2.begin(), r2.end());
                return result;
            }
        }
    }

    // === 4. Если ничего не нашли ===
    std::cout << "[berlekamp] failed to split -> irreducible: " << F.str() << "\n";
    result.push_back(F);
    return result;
}


// full factorization using squarefree + berlekamp
std::vector<Poly> factor_poly(const Poly& F) {
    std::cout << "[factor] starting full factorization for " << F.str() << "\n";
    if (F.isZero()) return {};

    // сделать коэф при старшей степени 1
    Poly f = F;
    f.normalize();
    // make monic
    if (!f.isZero()) {
        poly_obj_t invLead = inv_mod(f.coeff(f.deg()));
        f = poly_scalar_mul(f, invLead);
    }

    return berlekamp(f);
}

int main() {
    std::cout << "Berlekamp factorization (no external libs)\n";
    std::cout << "Enter prime p, degree n, and n+1 coefficients (highest first):\n";
    if (!(std::cin >> P)) return 0;
    size_t n; std::cin >> n;
    std::vector<poly_obj_t> coeffs(n + 1);
    for (int i = 0; i <= n; ++i) std::cin >> coeffs[i];
    Poly f = poly_make_from_coeffs(coeffs);
    std::cout << "Input polynomial: " << f.str() << " over GF(" << P << ")\n";
    try {
        auto res = factor_poly(f);
        std::cout << "\n=== RESULT: irreducible factors (factor, multiplicity) ===\n";
        for (auto& pr : res) std::cout << "(" << pr.str() << ")\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
    }
    return 0;
}
