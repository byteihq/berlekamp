#include "poly.h"
#include <CL/cl.h>
#if defined(__has_include)
#  if __has_include(<CL/cl.h>)
#    include <CL/cl.h>
#    define HAVE_OPENCL 1
#  elif __has_include(<OpenCL/opencl.h>)
#    include <OpenCL/opencl.h>
#    define HAVE_OPENCL 1
#  else
#    define HAVE_OPENCL 0
#  endif
#else
#  define HAVE_OPENCL 0
#endif

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <algorithm>

// Copy of helper from berlekamp.cpp
static std::string formatPoly(const Poly& p)
{
    if (p.isZero()) return "0";
    std::string s;
    for (size_t i = p.deg(); i > 0; --i) {
        auto c = p[i];
        if (c == 0) continue;
        if (!s.empty()) s += " + ";
        s += std::to_string(c);
        s += "*x";
        if (i > 1) s += "^" + std::to_string(i);
    }
    if (p[0] != 0) {
        if (!s.empty()) s += " + ";
        s += std::to_string(p[0]);
    }
    return s;
}

// Gaussian elimination (same as berlekamp.cpp)
static std::vector<int> matrix_rref(std::vector<std::vector<Poly::value_t>>& M, Poly::value_t P) {
    size_t rows = M.size();
    if (rows == 0) return {};
    size_t cols = M[0].size();
    std::vector<int> pivot_col_for_row(rows, -1);
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c) {
        int sel = -1;
        for (size_t i = r; i < rows; ++i) if (M[i][c] != 0) { sel = (int)i; break; }
        if (sel == -1) continue;
        swap(M[r], M[sel]);
        Poly::value_t inv = inv_mod(M[r][c], P);
        for (size_t j = c; j < cols; ++j) M[r][j] = modnorm(M[r][j] * inv, P);
        for (size_t i = 0; i < rows; ++i) if (i != r && M[i][c] != 0) {
            Poly::value_t factor = M[i][c];
            for (size_t j = c; j < cols; ++j) {
                if (const auto rhs = (factor * M[r][j]) % P; M[i][j] < rhs)
                    M[i][j] = modnorm(P + M[i][j] - rhs, P);
                else
                    M[i][j] = modnorm(M[i][j] - rhs, P);
            }
        }
        pivot_col_for_row[r] = (int)c;
        ++r;
    }
    return pivot_col_for_row;
}

#if HAVE_OPENCL
// Kernel: MAXN will be replaced by host if needed. Keep conservative limit.
static const char* kernelTemplate = R"CLC(
#define MAXN %d
__kernel void compute_columns(__global const ulong* Fcoeffs, const uint n, const ulong P, __global const ulong* xp, __global ulong* out) {
    const uint j = get_global_id(0);
    if (j >= n) return;

    ulong res[MAXN];
    ulong base[MAXN];
    ulong temp[2*MAXN];

    for (uint i = 0; i < n; ++i) { res[i] = 0UL; base[i] = xp[i] % P; }
    for (uint i = n; i < MAXN; ++i) res[i] = 0UL;
    res[0] = 1UL;

    uint exp = j;
    while (exp > 0) {
        if (exp & 1u) {
            // res = res * base
            for (uint t = 0; t < 2*n; ++t) temp[t] = 0UL;
            for (uint u = 0; u < n; ++u) {
                if (res[u] == 0UL) continue;
                for (uint v = 0; v < n; ++v) {
                    if (base[v] == 0UL) continue;
                    ulong prod = (res[u] * base[v]) % P;
                    uint idx = u + v;
                    temp[idx] = (temp[idx] + prod) % P;
                }
            }
            // reduce
            for (int k = (int)(2*n - 2); k >= (int)n; --k) {
                ulong coef = temp[k] % P;
                if (coef == 0UL) continue;
                for (uint t = 1; t <= n; ++t) {
                    ulong sub = (coef * Fcoeffs[n - t]) % P;
                    if (temp[k - t] < sub) temp[k - t] = (P + temp[k - t] - sub) % P;
                    else temp[k - t] = (temp[k - t] - sub) % P;
                }
                temp[k] = 0UL;
            }
            for (uint i = 0; i < n; ++i) res[i] = temp[i] % P;
        }
        // base = base * base
        {
            for (uint t = 0; t < 2*n; ++t) temp[t] = 0UL;
            for (uint u = 0; u < n; ++u) {
                if (base[u] == 0UL) continue;
                for (uint v = 0; v < n; ++v) {
                    if (base[v] == 0UL) continue;
                    ulong prod = (base[u] * base[v]) % P;
                    uint idx = u + v;
                    temp[idx] = (temp[idx] + prod) % P;
                }
            }
            for (int k = (int)(2*n - 2); k >= (int)n; --k) {
                ulong coef = temp[k] % P;
                if (coef == 0UL) continue;
                for (uint t = 1; t <= n; ++t) {
                    ulong sub = (coef * Fcoeffs[n - t]) % P;
                    if (temp[k - t] < sub) temp[k - t] = (P + temp[k - t] - sub) % P;
                    else temp[k - t] = (temp[k - t] - sub) % P;
                }
                temp[k] = 0UL;
            }
            for (uint i = 0; i < n; ++i) base[i] = temp[i] % P;
        }
        exp >>= 1;
    }

    for (uint i = 0; i < n; ++i) out[i*n + j] = res[i] % P;
}
)CLC";
#endif

// Build Q on CPU (fallback)
static void build_Q_cpu(const Poly& F, std::vector<std::vector<Poly::value_t>>& Q) {
    size_t n = F.deg();
    Q.assign(n, std::vector<Poly::value_t>(n, 0));
    Poly X({ 0,1 }, F.getMod());
    Poly xp = poly_powmod(X, F.getMod(), F);
    xp.normalize();
    for (size_t j = 0; j < n; ++j) {
        Poly col = poly_powmod(xp, j, F);
        for (size_t i = 0; i < col.deg() + 1; ++i) Q[i][j] = col[i];
    }
}

#if HAVE_OPENCL
// Try compute Q on GPU using OpenCL
static bool build_Q_gpu(const Poly& F, std::vector<std::vector<Poly::value_t>>& Q) {
    size_t n = F.deg();
    if (n == 0) return false;
    using vt = Poly::value_t;
    vt P = F.getMod();

    // prepare arrays
    std::vector<cl_ulong> Fcoeffs(n), xp_coeffs(n);
    for (size_t i = 0; i < n; ++i) {
        Fcoeffs[i] = (i < F.deg() + 1) ? (cl_ulong)F[i] : 0;
    }
    Poly X({ 0,1 }, P);
    Poly xp = poly_powmod(X, P, F);
    xp.normalize();
    for (size_t i = 0; i < n; ++i) xp_coeffs[i] = (i < xp.deg() + 1) ? (cl_ulong)xp[i] : 0;

    cl_int err;
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) return false;
    std::vector<cl_platform_id> platforms(platformCount);
    if ((err = clGetPlatformIDs(platformCount, platforms.data(), nullptr)) != CL_SUCCESS) return false;
    cl_platform_id platform = platforms[0];

    cl_uint deviceCount = 0;
    if ((err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount)) != CL_SUCCESS || deviceCount == 0) return false;
    std::vector<cl_device_id> devices(deviceCount);
    if ((err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr)) != CL_SUCCESS) return false;
    cl_device_id device = devices[0];

    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) return false;

    cl_command_queue queue = nullptr;
#if defined(CL_VERSION_2_0)
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    if (!queue || err != CL_SUCCESS) { clReleaseContext(context); return false; }

    // build kernel
    int MAXN = (int)std::max<size_t>(256, n);
    // cap MAXN to reasonable size to avoid huge kernels
    if (MAXN > 1024) MAXN = 1024;
    // Create source string by replacing first "%d" with MAXN (avoid printf on template with "%")
    std::string srcStr(kernelTemplate);
    auto pos = srcStr.find("%d");
    if (pos != std::string::npos) srcStr.replace(pos, 2, std::to_string(MAXN));
    const char* src = srcStr.c_str();
    size_t srcLen = srcStr.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    if (!program || err != CL_SUCCESS) { clReleaseCommandQueue(queue); clReleaseContext(context); return false; }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "OpenCL build log:\n" << log << "\n";
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "compute_columns", &err);
    if (!kernel || err != CL_SUCCESS) { clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context); return false; }

    // Declare objects that have non-trivial constructors here to avoid MSVC C2362
    cl_uint cln = 0;
    cl_ulong clP = 0;
    size_t global = 0;
    std::vector<cl_ulong> out_flat;

    size_t bytes_n = n * sizeof(cl_ulong);
    size_t bytes_nn = n * n * sizeof(cl_ulong);
    cl_mem bufF = nullptr;
    cl_mem bufxp = nullptr;
    cl_mem bufOut = nullptr;
    bufF = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_n, Fcoeffs.data(), &err);
    if (err != CL_SUCCESS) goto cl_error;
    bufxp = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_n, xp_coeffs.data(), &err);
    if (err != CL_SUCCESS) goto cl_error;
    bufOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_nn, nullptr, &err);
    if (err != CL_SUCCESS) goto cl_error;

    cln = (cl_uint)n;
    clP = (cl_ulong)P;
    if ((err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufF)) != CL_SUCCESS) goto cl_error;
    if ((err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &cln)) != CL_SUCCESS) goto cl_error;
    if ((err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &clP)) != CL_SUCCESS) goto cl_error;
    if ((err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufxp)) != CL_SUCCESS) goto cl_error;
    if ((err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufOut)) != CL_SUCCESS) goto cl_error;

    global = n;
    if ((err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr)) != CL_SUCCESS) goto cl_error;
    clFinish(queue);

    out_flat.assign(n * n, 0);
    if ((err = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, bytes_nn, out_flat.data(), 0, nullptr, nullptr)) != CL_SUCCESS) goto cl_error;

    Q.assign(n, std::vector<Poly::value_t>(n, 0));
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Q[i][j] = (Poly::value_t)(out_flat[i * n + j] % P);

    clReleaseMemObject(bufF); clReleaseMemObject(bufxp); clReleaseMemObject(bufOut);
    clReleaseKernel(kernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
    return true;

cl_error:
    std::cerr << "[OpenCL] error code: " << err << "\n";
    if (bufF) clReleaseMemObject(bufF);
    if (bufxp) clReleaseMemObject(bufxp);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    return false;
}
#endif // HAVE_OPENCL

static std::vector<Poly> berlekamp(const Poly& F) {
    std::cout << "[berlekamp] factoring (squarefree) " << formatPoly(F) << "\n";
    std::vector<Poly> result;
    if (F.isZero()) return result;
    if (F.deg() <= 0) return result;
    if (F.deg() == 1) { result.push_back(F); return result; }
    size_t n = F.deg();

    std::cout << "[berlekamp] building Q matrix (size " << n << ")...\n";
    std::vector<std::vector<Poly::value_t>> Q;
#if HAVE_OPENCL
    bool ok = build_Q_gpu(F, Q);
    if (!ok) {
        std::cerr << "[parallel] OpenCL path failed, using CPU fallback\n";
        build_Q_cpu(F, Q);
    }
    else {
        std::cout << "[parallel] Q computed on GPU\n";
    }
#else
    build_Q_cpu(F, Q);
#endif

    for (size_t i = 0; i < n; ++i) Q[i][i] = modnorm(Q[i][i] - 1, F.getMod());

    std::cout << "[berlekamp] matrix (Q - I):\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) std::cout << Q[i][j] << ' ';
        std::cout << '\n';
    }

    auto Qcopy = Q;
    auto piv = matrix_rref(Qcopy, F.getMod());

    std::vector<int> pivot_of_col(n, -1);
    for (int r = 0; r < (int)piv.size(); ++r) if (piv[r] != -1) pivot_of_col[piv[r]] = r;
    std::vector<int> free_cols;
    for (int c = 0; c < (int)n; ++c) if (pivot_of_col[c] == -1) free_cols.push_back(c);

    std::cout << "[berlekamp] nullspace dimension = " << free_cols.size() << "\n";

    std::vector<Poly> basis;
    for (size_t idx = 0; idx < free_cols.size(); ++idx) {
        int fc = free_cols[idx];
        std::vector<Poly::value_t> vec(n, 0);
        vec[fc] = 1;
        for (int r = 0; r < (int)piv.size(); ++r) {
            int c = piv[r]; if (c == -1) continue;
            vec[c] = modnorm(F.getMod() - Qcopy[r][fc], F.getMod());
        }
        Poly b(vec, F.getMod()); b.normalize();
        std::cout << "[berlekamp] basis[" << idx << "] = " << formatPoly(b) << "\n";
        basis.push_back(std::move(b));
    }

    if (basis.empty()) { std::cout << "[berlekamp] nullspace trivial -> irreducible: " << formatPoly(F) << "\n"; result.push_back(F); return result; }
    if (basis.size() == 1 && basis.front().deg() == 0) { std::cout << "[berlekamp] nullspace contains only constants -> irreducible\n"; result.push_back(F); return result; }

    for (size_t bi = 0; bi < basis.size(); ++bi) {
        const Poly& v = basis[bi]; if (v.deg() == 0) continue;
        std::cout << "[berlekamp] trying basis vector: " << formatPoly(v) << "\n";
        for (size_t c = 0; c < F.getMod(); ++c) {
            Poly h = v; if (h.isZero()) h.push(F.getMod());
            if (const auto rhs = c % F.getMod(); h[0] < rhs) h[0] = modnorm(F.getMod() + h[0] - rhs, F.getMod()); else h[0] = modnorm(h[0] - c, F.getMod());
            h.normalize();
            Poly g = poly_gcd(F, h);
            if (!g.isZero() && g.deg() >= 1 && g.deg() < F.deg()) {
                std::cout << "[berlekamp] non-trivial factor found: " << formatPoly(g) << " (with c=" << c << ")\n";
                Poly f1 = g; Poly f2 = poly_divmod(F, f1).first;
                auto r1 = berlekamp(f1); auto r2 = berlekamp(f2);
                result.insert(result.end(), r1.begin(), r1.end()); result.insert(result.end(), r2.begin(), r2.end());
                return result;
            }
        }
    }

    std::cout << "[berlekamp] failed to split -> irreducible: " << formatPoly(F) << "\n";
    result.push_back(F);
    return result;
}

static std::vector<Poly> factor_poly(Poly& F) {
    F.normalize(); if (F.isZero()) return {};
    Poly::value_t invLead = inv_mod(F[F.deg()], F.getMod());
    F = poly_scalar_mul(F, invLead); F.normalize();
    return berlekamp(F);
}

int main() {
    std::cout << "Berlekamp factorization (parallel)\n";
    std::cout << "Enter prime p, degree n, and n+1 coefficients (highest first):\n";
    uint64_t mod; if (!(std::cin >> mod)) return 0;
    size_t n; std::cin >> n;
    std::vector<Poly::value_t> coeffs(n + 1);
    for (int i = (int)n; i >= 0; --i) std::cin >> coeffs[i];
    Poly f(coeffs, mod);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = factor_poly(f);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== RESULT: irreducible factors ===\n";
    for (auto& pr : res) std::cout << "(" << formatPoly(pr) << ")\n";

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Elapsed time: " << ms << " ms\n";
    return 0;
}