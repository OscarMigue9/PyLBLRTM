extern "C" __global__
void dtau_kernel(const double* __restrict__ nu,   int npts,
                 const double* __restrict__ nu0,  int nlin,
                 const double* __restrict__ S,
                 const double* __restrict__ gam,
                 const double* __restrict__ alfd,
                 const double* __restrict__ alfv,
                 const double* __restrict__ Yv,
                 const int*    __restrict__ isco2,
                 const double* __restrict__ Pl,
                 int nlay, double CUT,
                 double* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = nlay * npts;
    if (tid >= tot) return;
    int L = tid / npts;
    int j = tid - L * npts;
    const double v   = nu[j];
    const double P   = Pl[L];
    const double C2c = CUT * CUT;
    const double S2  = 0.8325546111576977;
    const double ISP = 0.5641895835477563;
    const double IPI = 0.3183098861837907;
    const size_t off = (size_t)L * nlin;
    double acc = 0.0;
    for (int i = 0; i < nlin; ++i) {
        double d = v - nu0[i];
        if (d < -CUT || d > CUT) continue;
        double g = gam[off + i];
        double a = alfd[off + i];
        double x = S2 * d / a;
        double y = S2 * g / a;
        double ax = fabs(x), s = ax + y;

        double Tr = y, Ti = -x, wr;
        if (s >= 15.0) {

            double ur = Tr*Tr - Ti*Ti + 0.5, ui = 2.0*Tr*Ti;
            double nr = Tr*0.5641896, ni = Ti*0.5641896;
            double den = ur*ur + ui*ui;
            wr = (nr*ur + ni*ui) / den;
        } else if (s >= 5.5) {
            double ur = Tr*Tr - Ti*Ti, ui = 2.0*Tr*Ti;
            double pr = 1.410474 + ur*0.5641896, pi = ui*0.5641896;
            double nr = Tr*pr - Ti*pi, ni = Tr*pi + Ti*pr;
            double qr = 0.75 + 3.0*ur + (ur*ur - ui*ui);
            double qi = 3.0*ui + 2.0*ur*ui;
            double den = qr*qr + qi*qi;
            wr = (nr*qr + ni*qi) / den;
        } else if (y >= 0.195*ax - 0.176) {

            const double NC[4] = {3.778987, 11.96482, 20.20933, 16.4955};
            const double DC[5] = {6.699398, 21.69274, 39.27121, 38.82363, 16.4955};
            double nr = 0.5642236, ni = 0.0, qr = 1.0, qi = 0.0, tr, ti;
            for (int k = 0; k < 4; ++k) {
                tr = nr*Tr - ni*Ti; ti = nr*Ti + ni*Tr; nr = tr + NC[k]; ni = ti;
            }
            for (int k = 0; k < 5; ++k) {
                tr = qr*Tr - qi*Ti; ti = qr*Ti + qi*Tr; qr = tr + DC[k]; qi = ti;
            }
            double den = qr*qr + qi*qi;
            wr = (nr*qr + ni*qi) / den;
        } else {


            const double NC[6] = {1.320522, 35.76683, 219.0313, 1540.787,
                                  3321.9905, 36183.31};
            const double DC[7] = {1.841439, 61.57037, 364.2191, 2186.181,
                                  9022.228, 24322.84, 32066.6};
            double ur = Tr*Tr - Ti*Ti, ui = 2.0*Tr*Ti;
            double nr = 0.56419, ni = 0.0, dr = 1.0, di = 0.0, tr, ti;
            for (int k = 0; k < 6; ++k) {
                tr = nr*ur - ni*ui; ti = nr*ui + ni*ur;
                nr = NC[k] - tr;   ni = -ti;
            }
            double mr = Tr*nr - Ti*ni, mi = Tr*ni + Ti*nr;
            for (int k = 0; k < 7; ++k) {
                tr = dr*ur - di*ui; ti = dr*ui + di*ur;
                dr = DC[k] - tr;   di = -ti;
            }
            double eu = exp(ur), er = eu*cos(ui);
            double den = dr*dr + di*di;
            wr = er - (mr*dr + mi*di) / den;
        }
        double V  = wr * (ISP * S2 / a);
        double LC = IPI * g / (C2c + g*g);
        double f;
        if (isco2[i]) {
            f = V - (2.0 - d*d/C2c) * LC;
            double Y = Yv[off + i];
            if (Y != 0.0) f *= (1.0 + d * Y * P / alfv[off + i]);
        } else {
            f = V - LC;
        }
        acc += S[off + i] * f;
    }
    out[(size_t)L * npts + j] += acc;
}
