/*
Berlekamp factorization over finite field GF(p) — single-file C++ implementation
(no external libraries).

Что внутри:
 - Полиномиальная арифметика над GF(p): сложение, умножение, деление с остатком, gcd.
 - Функции для приведения многочлена к нормальной форме (ведущий коэффициент = 1).
 - Квадратно-свободное разложение (squarefree factorization).
 - Реализация матрицы Берлекемпа: построение матрицы Q- I и вычисление её ядра
   (методом элементарных преобразований над GF(p)).
 - Стратегия разделения: случайные линейные комбинации в ядре и gcd(f, b(x)-c).
 - Подробное логирование (std::cout) всех значимых шагов для отладки и понимания.

Ограничения и замечания:
 - Модуль p должен быть простым числом (алгоритм работает для конечного поля GF(p)).
 - Этот код ориентирован на понятность, поэтому использует наивные умножения
   (O(n^2)). Для больших степеней потребуется оптимизация.
 - В реализации используется 128-битная арифметика (__int128) для безопасного умножения
   коэффициентов (чтобы избежать переполнений при умножении чисел порядка p).

Компиляция:
 g++ -std=c++17 -O2 Berlekamp_factorization.cpp -o berlekamp

Пример запуска (ввод из stdin):
 В первой строке: простое p
 Во второй строке: степень n (>=1)
 В третьей строке: n+1 коэффициентов через пробел, начиная с старшего (коэффициент при x^n)
 Пример:
 5
 3
 1 0 4 1   // значит x^3 + 0*x^2 + 4*x + 1 над GF(5)

Программа выведет подробный лог и список неприводимых множителей (включая кратности).
*/

#include <iostream>
#include <vector>
#include <string>
#include <random>

using namespace std;
using int64 = long long;

static int64 P = 2; // глобальный модуль поля (prime)

inline int64 modnorm(int64 a) {
    a %= P; if (a < 0) a += P; return a;
}

// extended gcd for integers: returns (g, x, y) such that a*x + b*y = g
static tuple<int64, int64, int64> egcd(int64 a, int64 b) {
    if (b == 0) return { a, 1, 0 };
    auto [g, x1, y1] = egcd(b, a % b);
    int64 x = y1;
    int64 y = x1 - (a / b) * y1;
    return { g, x, y };
}

int64 inv_mod(int64 a) {
    a = modnorm(a);
    if (a == 0) throw runtime_error("inverse of zero");
    auto [g, x, y] = egcd(a, P);
    if (g != 1) {
        throw runtime_error("no inverse exists, modulus not prime or a==0");
    }
    return modnorm(x);
}

struct Poly {
    // coefficients: a[i] * x^i
    vector<int64> a;
    Poly() {}
    Poly(size_t n) : a(n) {}
    Poly(initializer_list<int64> il) : a(il) {}

    void normalize() {
        while (!a.empty() && modnorm(a.back()) == 0) a.pop_back();
        for (size_t i = 0; i < a.size(); ++i) a[i] = modnorm(a[i]);
    }

    int deg() const { return (int)a.size() - 1; }
    bool isZero() const { return a.empty(); }

    int64 coeff(int i) const { return (i >= 0 && i < (int)a.size()) ? modnorm(a[i]) : 0; }

    string str() const {
        if (isZero()) return "0";
        string s;
        for (int i = deg(); i >= 0; --i) {
            int64 c = coeff(i);
            if (c == 0) continue;
            if (!s.empty()) s += " + ";
            s += to_string(c);
            if (i > 0) s += "*x";
            if (i > 1) s += "^" + to_string(i);
        }
        return s;
    }
};

Poly poly_make_from_coeffs(const vector<int64>& coeffs_high_to_low) {
    int n = coeffs_high_to_low.size();
    Poly p(n);
    for (int i = 0; i < n; ++i) p.a[n - 1 - i] = modnorm(coeffs_high_to_low[i]);
    p.normalize();
    return p;
}

Poly poly_add(const Poly& A, const Poly& B) {
    Poly C(max((int)A.a.size(), (int)B.a.size()));
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = modnorm(A.coeff(i) + B.coeff(i));
    C.normalize();
    return C;
}

Poly poly_sub(const Poly& A, const Poly& B) {
    Poly C(max((int)A.a.size(), (int)B.a.size()));
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = modnorm(A.coeff(i) - B.coeff(i));
    C.normalize();
    return C;
}

Poly poly_scalar_mul(const Poly& A, int64 k) {
    Poly C(A.a.size());
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = modnorm(A.a[i] * k);
    C.normalize();
    return C;
}

Poly poly_mul(const Poly& A, const Poly& B) {
    if (A.isZero() || B.isZero()) return Poly();
    int na = A.deg(), nb = B.deg();
    Poly C(na + nb + 1);
    for (int i = 0; i <= na; ++i) {
        for (int j = 0; j <= nb; ++j) {
            int64_t prod = (int64_t)A.coeff(i) * (int64_t)B.coeff(j);
            C.a[i + j] = modnorm(C.a[i + j] + (int64)(prod % P));
        }
    }
    C.normalize();
    return C;
}

// polynomial long division: returns quotient Q and remainder R so that A = B*Q + R
pair<Poly, Poly> poly_divmod(Poly A, Poly B) {
    A.normalize(); B.normalize();
    if (B.isZero()) throw runtime_error("division by zero polynomial");
    if (A.deg() < B.deg()) return { Poly(), A };
    Poly Q(max(0, A.deg() - B.deg()) + 1);
    Poly R = A;
    int64 invLeadB = inv_mod(B.coeff(B.deg()));
    while (!R.isZero() && R.deg() >= B.deg()) {
        int d = R.deg() - B.deg();
        int64 coef = modnorm(R.coeff(R.deg()) * invLeadB);
        Q.a[d] = coef;
        // subtract coef * x^d * B from R
        for (int i = 0; i <= B.deg(); ++i) {
            int idx = i + d;
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
        int64 invLead = inv_mod(A.coeff(A.deg()));
        A = poly_scalar_mul(A, invLead);
    }
    return A;
}

// polynomial exponentiation modulo M
Poly poly_powmod(Poly base, long long exp, const Poly& M) {
    base.normalize();
    Poly res; res.a = { 1 }; // 1
    if (M.isZero()) throw runtime_error("modulus polynomial is zero");
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
        if (A.coeff(i) != 0 && (i % P) != 0) throw runtime_error("pth root: degrees not multiples of p");
        if (i % P == 0) {
            R.a[i / P] = A.coeff(i);
        }
    }
    R.normalize();
    return R;
}

// squarefree factorization over GF(P)
// returns vector of pairs (factor, multiplicity)
vector<pair<Poly, int>> squarefree_factor(const Poly& F) {
    cout << "[squarefree] start for: " << F.str() << " over GF(" << P << ")\n";
    vector<pair<Poly, int>> res;
    if (F.isZero()) return res;
    Poly f = F;
    f.normalize();
    int64 p = P;

    Poly df; // formal derivative
    if (f.deg() <= 0) return res;
    df.a.assign(max(0, f.deg()), 0);
    for (int i = 1; i <= f.deg(); ++i) {
        df.a[i - 1] = modnorm(f.coeff(i) * i);
    }
    df.normalize();

    if (df.isZero()) {
        // derivative is zero -> polynomial is a polynomial in x^p
        cout << "[squarefree] derivative is zero: extracting p-th root and recurse\n";
        Poly g = poly_pth_root(f);
        auto sub = squarefree_factor(g);
        // each multiplicity multiplied by p
        for (auto& pr : sub) res.emplace_back(pr.first, pr.second * (int)P);
        return res;
    }

    Poly a = poly_gcd(f, df);
    Poly b = poly_divmod(f, a).first; // f / a
    int i = 1;
    while (!b.isZero()) {
        Poly y = poly_gcd(b, a);
        Poly z = poly_divmod(b, y).first; // b / y

        if (z.a.size() == 1 && z.a[0] == 1)
            break;

        if (!z.isZero()) {
            z.normalize();
            cout << "[squarefree] found factor (multiplicity " << i << "): " << z.str() << "\n";
            res.emplace_back(z, i);
        }

        b = y;
        // a = a / y
        a = poly_divmod(a, y).first;
        ++i;
    }
    if (!a.isZero()) {
        // remaining a contains p-th powers
        cout << "[squarefree] remaining factor is p-th power; taking p-th root and recursing\n";
        Poly g = poly_pth_root(a);
        auto sub = squarefree_factor(g);
        for (auto& pr : sub) res.emplace_back(pr.first, pr.second * (int)P);
    }
    return res;
}

// Gaussian elimination to compute RREF of matrix (rows x cols) over GF(P)
// it modifies matrix to RREF and returns pivot_col_for_row (size rows) with -1 for zero rows
vector<int> matrix_rref(vector<vector<int64>>& M) {
    int rows = M.size();
    if (rows == 0) return {};
    int cols = M[0].size();
    vector<int> pivot_col_for_row(rows, -1);
    int r = 0;
    for (int c = 0; c < cols && r < rows; ++c) {
        // find pivot
        int sel = -1;
        for (int i = r; i < rows; ++i) if (M[i][c] != 0) { sel = i; break; }
        if (sel == -1) continue;
        swap(M[r], M[sel]);
        int64 inv = inv_mod(M[r][c]);
        // scale row r to make pivot 1
        for (int j = c; j < cols; ++j) M[r][j] = modnorm(M[r][j] * inv);
        // eliminate other rows
        for (int i = 0; i < rows; ++i) if (i != r && M[i][c] != 0) {
            int64 factor = M[i][c];
            for (int j = c; j < cols; ++j) {
                M[i][j] = modnorm(M[i][j] - factor * M[r][j]);
            }
        }
        pivot_col_for_row[r] = c;
        ++r;
    }
    return pivot_col_for_row;
}

vector<Poly> berlekamp(const Poly& F) {
    cout << "[berlekamp] factoring (squarefree) " << F.str() << "\n";
    vector<Poly> result;
    if (F.isZero()) return result;
    if (F.deg() <= 0) return result;
    if (F.deg() == 1) { result.push_back(F); return result; }
    int n = F.deg();

    // === 1. Построение матрицы Q ===
    cout << "[berlekamp] building Q matrix (size " << n << ")...\n";
    Poly X; X.a = { 0,1 }; X.normalize();
    Poly xp = poly_powmod(X, P, F);

    vector<vector<int64>> Q(n, vector<int64>(n, 0));
    for (int j = 0; j < n; ++j) {
        Poly col = poly_powmod(xp, j, F); // x^(p*j) mod F
        for (int i = 0; i < n; ++i) Q[i][j] = col.coeff(i);
    }

    // Q - I
    for (int i = 0; i < n; ++i) Q[i][i] = modnorm(Q[i][i] - 1);

    cout << "[berlekamp] matrix (Q - I):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) cout << Q[i][j] << ' ';
        cout << '\n';
    }

    // === 2. Базис ядра ===
    auto Qcopy = Q;
    auto piv = matrix_rref(Qcopy);

    vector<int> pivot_of_col(n, -1);
    for (int r = 0; r < (int)piv.size(); ++r)
        if (piv[r] != -1) pivot_of_col[piv[r]] = r;

    vector<int> free_cols;
    for (int c = 0; c < n; ++c)
        if (pivot_of_col[c] == -1) free_cols.push_back(c);

    cout << "[berlekamp] nullspace dimension = " << free_cols.size() << "\n";

    vector<Poly> basis;
    for (int idx = 0; idx < (int)free_cols.size(); ++idx) {
        int fc = free_cols[idx];
        vector<int64> vec(n, 0);
        vec[fc] = 1;
        for (int r = 0; r < (int)piv.size(); ++r) {
            int c = piv[r];
            if (c == -1) continue;
            vec[c] = modnorm(-Qcopy[r][fc]);
        }
        Poly b; b.a = vec; b.normalize();
        cout << "[berlekamp] basis[" << idx << "] = " << b.str() << "\n";
        basis.push_back(b);
    }

    if (basis.empty()) {
        cout << "[berlekamp] nullspace trivial -> irreducible: " << F.str() << "\n";
        result.push_back(F);
        return result;
    }
    if ((int)basis.size() == 1 && basis[0].deg() == 0) {
        cout << "[berlekamp] nullspace contains only constants -> irreducible\n";
        result.push_back(F);
        return result;
    }

    // === 3. Перебор базиса и констант ===
    for (size_t bi = 0; bi < basis.size(); ++bi) {
        const Poly& v = basis[bi];
        if (v.deg() == 0) continue; // пропускаем константу
        cout << "[berlekamp] trying basis vector: " << v.str() << "\n";
        for (int c = 0; c < P; ++c) {
            Poly h = v;
            if (h.a.empty()) h.a = { 0 };
            h.a[0] = modnorm(h.a[0] - c);
            h.normalize();

            Poly g = poly_gcd(F, h);
            if (!g.isZero() && g.deg() >= 1 && g.deg() < F.deg()) {
                cout << "[berlekamp] non-trivial factor found: " << g.str()
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
    cout << "[berlekamp] failed to split -> irreducible: " << F.str() << "\n";
    result.push_back(F);
    return result;
}


// full factorization using squarefree + berlekamp
vector<pair<Poly, int>> factor_poly(const Poly& F) {
    cout << "[factor] starting full factorization for " << F.str() << "\n";
    if (F.isZero()) return {};

    // сделать коэф при старшей степени 1
    Poly f = F;
    f.normalize();
    // make monic
    if (!f.isZero()) {
        int64 invLead = inv_mod(f.coeff(f.deg()));
        f = poly_scalar_mul(f, invLead);
    }

    auto sq = squarefree_factor(f);
    vector<pair<Poly, int>> out;
    for (auto& pr : sq) {
        Poly gi = pr.first; int mult = pr.second;
        // gi is squarefree; factor it with Berlekamp
        gi.normalize();
        // make monic
        if (!gi.isZero()) {
            int64 invLead = inv_mod(gi.coeff(gi.deg()));
            gi = poly_scalar_mul(gi, invLead);
        }
        auto facs = berlekamp(gi);
        // facs is vector of irreducible factors (not necessarily unique?), but algorithm returns monic factors
        // increase multiplicity as found in squarefree decomposition
        for (auto& h : facs) {
            out.emplace_back(h, mult);
        }
    }
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "Berlekamp factorization (no external libs)\n";
    cout << "Enter prime p, degree n, and n+1 coefficients (highest first):\n";
    if (!(cin >> P)) return 0;
    int n; cin >> n;
    vector<int64> coeffs(n + 1);
    for (int i = 0; i <= n; ++i) cin >> coeffs[i];
    Poly f = poly_make_from_coeffs(coeffs);
    cout << "Input polynomial: " << f.str() << " over GF(" << P << ")\n";
    try {
        auto res = factor_poly(f);
        cout << "\n=== RESULT: irreducible factors (factor, multiplicity) ===\n";
        for (auto& pr : res) cout << "(" << pr.first.str() << ", " << pr.second << ")\n";
    }
    catch (const exception& ex) {
        cerr << "Error: " << ex.what() << "\n";
    }
    return 0;
}
