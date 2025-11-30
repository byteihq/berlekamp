#include "poly.h"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <string>

#define LOG std::cout

#ifndef NDEBUG
#define DLOG std::cout
#else
struct Dummy {
    constexpr Dummy() {}
    template<typename T>
    inline constexpr Dummy& operator<<(const T&) const { return *this;  }
};
static constexpr Dummy __d{};
#define DLOG __d
#endif

std::string format(const Poly& p)
{
    if (p.isZero())
        return "0";

    std::string s;

    for (size_t i = p.deg(); i > 0; --i) {
        const auto& c = p[i];
        if (c == 0)
            continue;

        if (!s.empty())
            s += " + ";

        s += std::to_string(c);
        s += "*x";

        if (i > 1)
            s += "^" + std::to_string(i);
    }

    if (p[0] != 0)
    {
        if (!s.empty())
            s += " + ";
        s += std::to_string(p[0]);
    }

    return s;
}

// Gaussian elimination to compute RREF of matrix (rows x cols) over GF(P)
// it modifies matrix to RREF and returns pivot_col_for_row (size rows) with -1 for zero rows
std::vector<int> matrix_rref(std::vector<std::vector<Poly::value_t>>& M, Poly::value_t P) {
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
        Poly::value_t inv = inv_mod(M[r][c], P);
        // scale row r to make pivot 1
        for (size_t j = c; j < cols; ++j) M[r][j] = modnorm(M[r][j] * inv, P);
        // eliminate other rows
        for (size_t i = 0; i < rows; ++i) if (i != r && M[i][c] != 0) {
            Poly::value_t factor = M[i][c];
            for (size_t j = c; j < cols; ++j) {
                if (const auto rhs = (factor * M[r][j]) % P; M[i][j] < rhs)
                {
                    M[i][j] = modnorm(P + M[i][j] - rhs, P);
                }
                else
                {
                    M[i][j] = modnorm(M[i][j] - rhs, P);
                }
            }
        }
        pivot_col_for_row[r] = (int)c;
        ++r;
    }
    return pivot_col_for_row;
}

std::vector<Poly> berlekamp(const Poly& F) {
    DLOG << "[berlekamp] factoring (squarefree) " << format(F) << "\n";

    std::vector<Poly> result;
    if (F.isZero()) return result;
    if (F.deg() <= 0) return result;
    if (F.deg() == 1) { result.push_back(F); return result; }
    size_t n = F.deg();

    // === 1. Построение матрицы Q ===
    DLOG << "[berlekamp] building Q matrix (size " << n << ")...\n";
    Poly X({0, 1}, F.getMod());
    
    Poly xp = poly_powmod(X, F.getMod(), F);
    xp.normalize();

    std::vector<std::vector<Poly::value_t>> Q(n, std::vector<Poly::value_t>(n, 0));
    for (size_t j = 0; j < n; ++j) {
        Poly col = poly_powmod(xp, j, F); // x^(p*j) mod F
        for (size_t i = 0; i < col.deg() + 1; ++i)
            Q[i][j] = col[i];
    }

    // Q - I
    for (size_t i = 0; i < n; ++i) Q[i][i] = modnorm(Q[i][i] - 1, F.getMod());

    DLOG << "[berlekamp] matrix (Q - I):\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) DLOG << Q[i][j] << ' ';
        DLOG << '\n';
    }

    // === 2. Базис ядра ===
    auto Qcopy = Q;
    auto piv = matrix_rref(Qcopy, F.getMod());

    std::vector<int> pivot_of_col(n, -1);
    for (int r = 0; r < (int)piv.size(); ++r)
        if (piv[r] != -1) pivot_of_col[piv[r]] = r;

    std::vector<int> free_cols;
    for (int c = 0; c < n; ++c)
        if (pivot_of_col[c] == -1)
            free_cols.push_back(c);

    DLOG << "[berlekamp] nullspace dimension = " << free_cols.size() << "\n";

    std::vector<Poly> basis;
    for (size_t idx = 0; idx < free_cols.size(); ++idx) {
        int fc = free_cols[idx];
        std::vector<Poly::value_t> vec(n, 0);
        vec[fc] = 1;
        for (int r = 0; r < (int)piv.size(); ++r) {
            int c = piv[r];
            if (c == -1) continue;
            vec[c] = modnorm(F.getMod() - Qcopy[r][fc], F.getMod());
        }

        Poly b(vec, F.getMod());
        b.normalize();
        DLOG << "[berlekamp] basis[" << idx << "] = " << format(b) << "\n";
        basis.push_back(std::move(b));
    }

    if (basis.empty()) {
        DLOG << "[berlekamp] nullspace trivial -> irreducible: " << format(F) << "\n";
        result.push_back(F);
        return result;
    }
    if (basis.size() == 1 && basis.front().deg() == 0) {
        DLOG << "[berlekamp] nullspace contains only constants -> irreducible\n";
        result.push_back(F);
        return result;
    }

    // === 3. Перебор базиса и констант ===
    for (size_t bi = 0; bi < basis.size(); ++bi) {
        const Poly& v = basis[bi];
        if (v.deg() == 0) continue; // пропускаем константу
        DLOG << "[berlekamp] trying basis vector: " << format(v) << "\n";

        for (size_t c = 0; c < F.getMod(); ++c) {
            Poly h = v;
            if (h.isZero())
                h.push(F.getMod());

            if (const auto rhs = c % F.getMod(); h[0] < rhs)
            {
                h[0] = modnorm(F.getMod() + h[0] - rhs, F.getMod());
            }
            else
            {
                h[0] = modnorm(h[0] - c, F.getMod());
            }
            h.normalize();

            Poly g = poly_gcd(F, h);
            if (!g.isZero() && g.deg() >= 1 && g.deg() < F.deg()) {
                DLOG << "[berlekamp] non-trivial factor found: " << format(g)
                    << " (with c=" << c << ")\n";
                Poly f1 = g;
                Poly f2 = poly_divmod(F, f1).first;

                auto r1 = berlekamp(f1);
                auto r2 = berlekamp(f2);
                std::transform(r1.begin(), r1.end(), std::back_inserter(result), [](Poly& p) { p.normalize(); return p; });
                std::transform(r2.begin(), r2.end(), std::back_inserter(result), [](Poly& p) { p.normalize(); return p; });
                return result;
            }
        }
    }

    // === 4. Если ничего не нашли ===
    DLOG << "[berlekamp] failed to split -> irreducible: " << format(F) << "\n";
    result.push_back(F);
    return result;
}

// full factorization using squarefree + berlekamp
std::vector<Poly> factor_poly(Poly& F) {
    DLOG << "[factor] starting full factorization for " << format(F) << "\n";

    F.normalize();
    if (F.isZero()) return {};

    Poly::value_t invLead = inv_mod(F[F.deg()], F.getMod());
    F = poly_scalar_mul(F, invLead);
    F.normalize();

    return berlekamp(F);
}

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    if (argc < 2) {
        LOG << "Usage: program <directory_path>\n";
        return 1;
    }

    fs::path dirPath = argv[1];
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        LOG << "Invalid directory path\n";
        return 2;
    }

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file())
            continue;

        std::ifstream file(entry.path());
        if (!file.is_open()) {
            LOG << "Failed to open file: " << entry.path() << "\n";
            continue;
        }

        LOG << "Processing file: " << entry.path() << "\n";

        const auto begin = std::chrono::high_resolution_clock::now();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            std::istringstream iss(line);

            Poly::value_t mod;
            iss >> mod;

            std::vector<Poly::value_t> coeffs;
            Poly::value_t x;
            while (iss >> x) {
                coeffs.push_back(x);
            }

            Poly f(coeffs, mod);
            DLOG << "Input polynomial: over GF(" << mod << ")\n";

            auto res = factor_poly(f);
            DLOG << "\n=== RESULT: irreducible factors (factor, multiplicity) ===\n";
            for (auto& pr : res)
                DLOG << "(" << format(pr) << ")\n";
        }
        const auto end = std::chrono::high_resolution_clock::now();
        LOG << "Time ellapsed " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";
    }

    return 0;
}
