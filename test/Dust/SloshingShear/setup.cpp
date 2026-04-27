#include "idefix.hpp"
#include "setup.hpp"
#include <complex>

static real omega;
static real shear;
static real psi;
static real cs;
real epsilon;
real chi;
real tauGlob;
real pi=M_PI;

#define  FILENAME    "timevol.dat"

//#define STRATIFIED
void PressureGradient(Hydro *hydro, const real t, const real dt) {
  auto Uc = hydro->Uc;
  auto Vc = hydro->Vc;
  DataBlock *data = hydro->data;
  real eps = epsilon;
  idefix_for("MySourceTerm",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                // Radial pressure gradient
                  Uc(MX1,k,j,i) += eps*Vc(RHO,k,j,i)*dt;
              });
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real omegaLocal=omega;
  real shearLocal =shear;

  idefix_for("BodyForce",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = -2.0*omegaLocal*shearLocal*x(i);
                force(JDIR,k,j,i) = ZERO_F;
                force(KDIR,k,j,i) = ZERO_F;

      });


  idfx::popRegion();
}

void InducedWarpFlow(Hydro *hydro, const real t, const real dt) {
  auto Uc = hydro->Uc;
  auto Vc = hydro->Vc;
  DataBlock *data = hydro->data;
  IdefixArray1D<real> z = data->x[KDIR];

  real omegaLocal=omega;
  real psiLocal = psi;

  idefix_for("MySourceTerm",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                  Uc(MX1,k,j,i) += psiLocal*omegaLocal*omegaLocal*z(k)*sin(omegaLocal*t)*dt;
              });
}

void InducedWarp(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("InducedWarp");
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real omegaLocal=omega;
  real psiLocal = psi;

  idefix_for("InducedWarp",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = psiLocal*omegaLocal*omegaLocal*z(k)*sin(omegaLocal*t);
                force(JDIR,k,j,i) = ZERO_F;
                force(KDIR,k,j,i) = ZERO_F;

      });


  idfx::popRegion();

}

void nonReflective(DataBlock& data, int dir, BoundarySide side, real t) {
  idfx::pushRegion("nonReflective");
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray4D<real> Uc = data.dust[0]->Vc;

  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  real shearLocal = shear;
  real chilocal = chi;


  int nxi = data.np_int[IDIR];
  int nxj = data.np_int[JDIR];
  int nxk = data.np_int[KDIR];

  const int ighost = data.nghost[IDIR];
  const int jghost = data.nghost[JDIR];
  const int kghost = data.nghost[KDIR];
  //DataBlockHost d(data);

  
  data.hydro->boundary->BoundaryFor("nonReflective", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);
        }
        else{
          Vc(VX1,k,j,i) = 4.142938*z(k)*cos(t);
          Vc(VX2,k,j,i) = -2.1162*z(k)*sin(t)+shearLocal*x(i);
        }
        });
  data.dust[0]->boundary->BoundaryFor("nonReflective", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        Uc(RHO,k,j,i) = Uc(RHO,kref,jref,iref);
        Uc(VX3,k,j,i) = Uc(VX3,kref,jref,iref);
        if(kref == k){
          Uc(VX1,k,j,i) = Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = Uc(VX2,kref,jref,iref);
        }
        else{
          Uc(VX1,k,j,i) = 4.160947*z(k)*cos(t);
          Uc(VX2,k,j,i) = -2.1067767*z(k)*sin(t)+shearLocal*x(i);
        }


        });
}

void ApplyBoundaryReversePeriodic(DataBlock *data, IdefixArray4D<real> Vc, int dir, BoundarySide side) {
  idfx::pushRegion("reversePeriodic");
  IdefixArray1D<real> x = data->x[IDIR];
  real shearLocal = shear;
  int nxi = data->np_int[IDIR];
  int nxj = data->np_int[JDIR];
  int nxk = data->np_int[KDIR];
  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

    data->hydro->boundary->BoundaryFor("reversePeriodic", dir, side,
      KOKKOS_LAMBDA (int k, int j, int i) {
        int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);

        }
        else{
          Vc(VX1,k,j,i) = -Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = -Vc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }
        });
}

void reversePeriodicGas(Hydro *hydro, int dir, BoundarySide side, real t) {
  ApplyBoundaryReversePeriodic(hydro->data, hydro->Vc, dir, side);
}

void reversePeriodicDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  ApplyBoundaryReversePeriodic(dust->data, dust->Vc, dir, side);
}


void reversePeriodic(DataBlock& data, int dir, BoundarySide side, real t) {
  idfx::pushRegion("reversePeriodic");
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray4D<real> Uc = data.dust[0]->Vc;

  IdefixArray1D<real> x = data.x[IDIR];
  real shearLocal = shear;


  int nxi = data.np_int[IDIR];
  int nxj = data.np_int[JDIR];
  int nxk = data.np_int[KDIR];

  const int ighost = data.nghost[IDIR];
  const int jghost = data.nghost[JDIR];
  const int kghost = data.nghost[KDIR];
  //DataBlockHost d(data);

  data.hydro->boundary->BoundaryFor("reversePeriodic", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);

        }
        else{
          Vc(VX1,k,j,i) = -Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = -Vc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }
        });
  data.dust[0]->boundary->BoundaryFor("reversePeriodic", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Uc(RHO,k,j,i) = Uc(RHO,kref,jref,iref);
        Uc(VX3,k,j,i) = Uc(VX3,kref,jref,iref);
        if(kref == k){
          Uc(VX1,k,j,i) = Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = Uc(VX2,kref,jref,iref);
        }
        else{
          Uc(VX1,k,j,i) = -Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = -Uc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }


        });
}



// Analyse data to produce an output
void Analysis(DataBlock & data) {


// Mirror data on Host
  DataBlockHost d(data);
  // Sync it
  d.SyncFromDevice();
  real rho = d.dustVc[0](RHO,0,0,0);
  real KE = 0.0;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        KE = KE + d.Vc(RHO,k,j,i)*d.Vc(VX3,k,j,i)*d.Vc(VX3,k,j,i);
        if(rho < d.dustVc[0](RHO,k,j,i)) {
          rho=d.dustVc[0](RHO,k,j,i);
        }
      }
    }
  }
  #ifdef WITH_MPI
  real rho_max;
  real KE_total;
  MPI_Reduce(&rho, &rho_max, 1, realMPI, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&KE, &KE_total, 1, realMPI, MPI_SUM, 0, MPI_COMM_WORLD);
  #endif
  if(idfx::prank == 0) {
  std::ofstream f;
  f.open(FILENAME,std::ios::app);
  f.precision(10);
  #ifdef WITH_MPI
  f << std::scientific << data.t << "	" << rho_max << "	" << 0.5/d.np_tot[KDIR]*1.2084/d.np_tot[IDIR]*3.14159/d.np_tot[JDIR]*KE_total << " " << std::endl;
  #else
  f << std::scientific << data.t << "	" << rho << "	" << 0.5/d.np_tot[KDIR]*1.2084/d.np_tot[IDIR]*3.14159/d.np_tot[JDIR]*KE << " " << std::endl;
  #endif
  f.close();
  }

}

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Get rotation rate along vertical axis
  omega=input.Get<real>("Hydro","rotation",0);
  shear=input.Get<real>("Hydro","shearingBox",0);
  psi=input.Get<real>("Hydro","warpMagnitude",0);
  cs=input.Get<real>("Hydro","csiso",1);


  tauGlob = input.Get<real>("Dust","drag",1);
  epsilon = input.Get<real>("Setup","epsilon",0);
  chi = input.Get<real>("Setup","chi",0);
  //data.hydro->EnrollUserSourceTerm(&PressureGradient);
  //data.hydro->EnrollUserSourceTerm(&InducedWarpFlow);

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
  output.EnrollAnalysis(&Analysis);
  //data.hydro->EnrollUserDefBoundary(&reversePeriodicGas);
  //data.dust[0]->EnrollUserDefBoundary(&reversePeriodicDust);
  //data.hydro->EnrollUserDefBoundary(nonReflective);
  //data.dust[0]->EnrollUserDefBoundary(nonReflective);

}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    using cplx = std::complex<real>;

    real taus = tauGlob * omega;
    real Lx   = data.mygrid->xend[IDIR] - data.mygrid->xbeg[IDIR];
    real Ly   = data.mygrid->xend[JDIR] - data.mygrid->xbeg[JDIR];
    real Lz   = data.mygrid->xend[KDIR] - data.mygrid->xbeg[KDIR];
    real Kb   = 2.*pi / Lz;

    // perturbation parameters
    real Kx   = 2*pi/Lx*3;
    real pert = 0.0001;
    real omega1 = -0.37309439207695594*omega;
    real omega2 = omega1+omega;

    // Mode frequencies (indices 1 and 2; index 0 unused)
    real w_n[3] = {0., omega1, omega2};

    // Per-mode precomputed complex eigenmode amplitudes and normalisations
    cplx ug[3], vg[3], wg[3], hg[3];
    cplx ud[3], vd[3], wd[3], hd[3];
    real norm_n[3], Kbn[3];

    for (int n = 1; n <= 2; n++) {
        real wn  = w_n[n];
        Kbn[n]   = Kb * n;

        // Gas dispersion bracket  D_gn = wn^2 - (Kb*n)^2
        real Dgn = wn*wn - Kbn[n]*Kbn[n];

        // Gas normalisation: A_n = pert / (wn * D_gn)
        // Chosen so that Re[A * u_gn * e^{i*phi}] = -pert*sin(phi)
        norm_n[n] = pert / (wn * Dgn);

        // Gas eigenmode components
        cplx ugn(0.,       wn * Dgn);           // i * wn * D_gn
        cplx vgn(0.5*Dgn,  0.);
        cplx wgn(0.,       Kx * Kbn[n] * wn);  // i * Kx * Kb*n * wn
        cplx hgn(0.,       Kx * wn * wn);       // i * Kx * wn^2

        ug[n] = ugn;
        vg[n] = vgn;
        wg[n] = wgn;
        hg[n] = hgn;

        // Complex denominator: chi_n = (1 - i*wn*taus)^2 + taus^2
        cplx chin = std::pow(cplx(1., -wn*taus), 2) + taus*taus;

        // Shared sub-expression for vertical coupling
        cplx denom_w = cplx(0.,1.) + taus*wn;

        // Dust eigenmode
        ud[n] = (ugn + 2.*taus*vgn - cplx(0.,1.)*taus*ugn*wn) / chin;
        vd[n] = (-2.*vgn + taus*(ugn + cplx(0.,2.)*vgn*wn)) / (-2.*chin);
        wd[n] = cplx(0.,1.) * wgn / denom_w;

        // Dust enthalpy
        hd[n] = (cplx(0.,1.)/wn) * (Kbn[n]*wgn/denom_w
                 - Kx*(cplx(0.,1.)*(ugn + 2.*taus*vgn) + taus*ugn*wn) / chin);
    }

    for (int k = 0; k < d.np_tot[KDIR]; k++) {
        for (int j = 0; j < d.np_tot[JDIR]; j++) {
            for (int i = 0; i < d.np_tot[IDIR]; i++) {
                real x = d.x[IDIR](i);
                real z = d.x[KDIR](k);

                real phi1 = Kx*x + Kbn[1]*z;
                real phi2 = Kx*x + Kbn[2]*z;

                // ── Gas ──────────────────────────────────────────────────
                // Density (exponential for positivity)
                d.Vc(RHO,k,j,i) = 1.0
                    * std::exp(norm_n[1]*(hg[1].real()*std::cos(phi1) - hg[1].imag()*std::sin(phi1)))
                    * std::exp(norm_n[2]*(hg[2].real()*std::cos(phi2) - hg[2].imag()*std::sin(phi2)));

                // VX1 (u)
                d.Vc(VX1,k,j,i) = psi*cs*std::sin(Kb*z)
                    + norm_n[1]*(ug[1].real()*std::cos(phi1) - ug[1].imag()*std::sin(phi1))
                    + norm_n[2]*(ug[2].real()*std::cos(phi2) - ug[2].imag()*std::sin(phi2));

                // VX2 (v)
                d.Vc(VX2,k,j,i) = shear*x
                    + norm_n[1]*(vg[1].real()*std::cos(phi1) - vg[1].imag()*std::sin(phi1))
                    + norm_n[2]*(vg[2].real()*std::cos(phi2) - vg[2].imag()*std::sin(phi2));

                // VX3 (w)
                d.Vc(VX3,k,j,i) = 0.
                    + norm_n[1]*(wg[1].real()*std::cos(phi1) - wg[1].imag()*std::sin(phi1))
                    + norm_n[2]*(wg[2].real()*std::cos(phi2) - wg[2].imag()*std::sin(phi2));

                // ── Dust ─────────────────────────────────────────────────
                // For any complex eigenmode amplitude E, the perturbation is:
                //   A_n * Re[E * e^{i*phi}] = A_n * (E_r*cos(phi) - E_i*sin(phi))

                // Density (exponential for positivity, consistent with gas)
                d.dustVc[0](RHO,k,j,i) = chi
                    * std::exp(norm_n[1]*(hd[1].real()*std::cos(phi1) - hd[1].imag()*std::sin(phi1)))
                    * std::exp(norm_n[2]*(hd[2].real()*std::cos(phi2) - hd[2].imag()*std::sin(phi2)));

                // VX1 (u_d)
                d.dustVc[0](VX1,k,j,i) = psi*cs*std::sin(Kb*z)
                    + norm_n[1]*(ud[1].real()*std::cos(phi1) - ud[1].imag()*std::sin(phi1))
                    + norm_n[2]*(ud[2].real()*std::cos(phi2) - ud[2].imag()*std::sin(phi2));

                // VX2 (v_d)
                d.dustVc[0](VX2,k,j,i) = shear*x
                    + norm_n[1]*(vd[1].real()*std::cos(phi1) - vd[1].imag()*std::sin(phi1))
                    + norm_n[2]*(vd[2].real()*std::cos(phi2) - vd[2].imag()*std::sin(phi2));

                // VX3 (w_d)
                d.dustVc[0](VX3,k,j,i) = 0.
                    + norm_n[1]*(wd[1].real()*std::cos(phi1) - wd[1].imag()*std::sin(phi1))
                    + norm_n[2]*(wd[2].real()*std::cos(phi2) - wd[2].imag()*std::sin(phi2));
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock & data) {
}
