// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// #if !MAGNETIC_FIELDS_ENABLED
// #error "This problem generator requires magnetic fields"
// #endif


// declare metric definition
void MyCoordinateSystem(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);


// enroll MyCoordinateSystem
void Mesh::InitUserMeshData(ParameterInput *pin){

  EnrollUserMetric(MyCoordinateSystem);
}



void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // constants
  Real gm1 = peos->GetGamma() - 1.0;
  Real sigma = pin->GetReal("problem", "sigma");
  Real dsigma = pin->GetReal("problem", "dsigma");
  Real beta = pin->GetReal("problem", "beta");
  Real L = pin->GetReal("problem", "L");
  Real B0 = std::sqrt(2*sigma);
  Real dB = std::sqrt(2*dsigma);
  Real Ptot = sigma*(beta + 1);

  // Initialize density, momentum, and magnetic field
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

        // density
        phydro->u(IDN,k,j,i) = 1.0;

        // momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        //magnetic field
        pfield->b.x1f(k,j,i) = B0*std::tanh(std::sin(TWO_PI*pcoord->x2f(j)) / L) + dB*std::sin(TWO_PI*pcoord->x2f(j));
        pfield->b.x2f(k,j,i) = dB*std::sin(TWO_PI*pcoord->x1f(i));
        pfield->b.x3f(k,j,i) = 0.0;
      }
    }
  }

  // initialize total energy
  Real Pmag;
  if (NON_BAROTROPIC_EOS) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Pmag = 0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
               SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
               SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));
          phydro->u(IEN,k,j,i) = Ptot - Pmag;
        }
      }
    }
  }

  return;
}


void MyCoordinateSystem(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3) {

  // Extract inputs
  Real R = pin->GetReal("coord", "R");

  // Intermediate
  Real sx = std::sin(TWO_PI*x1);
  Real cx = std::cos(TWO_PI*x1);

  // Set covariant components
  g(I00) = -1.0;
  g(I01) = 0.0;
  g(I02) = 0.0;
  g(I03) = 0.0;
  g(I11) = SQR(R);
  g(I12) = 0.0;
  g(I13) = 0.0;
  g(I22) = SQR(R)*SQR(sx);
  g(I23) = 0.0;
  g(I33) = 1.0;

  // Set contravariant components
  g_inv(I00) = -1.0;
  g_inv(I01) = 0.0;
  g_inv(I02) = 0.0;
  g_inv(I03) = 0.0;
  g_inv(I11) = 1 / SQR(R);
  g_inv(I12) = 0.0;
  g_inv(I13) = 0.0;
  g_inv(I22) = 1 / (SQR(R)*SQR(sx));
  g_inv(I23) = 0.0;
  g_inv(I33) = 1.0;

  // Set x-derivatives of covariant components
  dg_dx1(I00) = 0.0;
  dg_dx1(I01) = 0.0;
  dg_dx1(I02) = 0.0;
  dg_dx1(I03) = 0.0;
  dg_dx1(I11) = 0.0;
  dg_dx1(I12) = 0.0;
  dg_dx1(I13) = 0.0;
  dg_dx1(I22) = 2*TWO_PI*SQR(R)*sx*cx;
  dg_dx1(I23) = 0.0;
  dg_dx1(I33) = 0.0;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = 0.0;
  dg_dx2(I01) = 0.0;
  dg_dx2(I02) = 0.0;
  dg_dx2(I03) = 0.0;
  dg_dx2(I11) = 0.0;
  dg_dx2(I12) = 0.0;
  dg_dx2(I13) = 0.0;
  dg_dx2(I22) = 0.0;
  dg_dx2(I23) = 0.0;
  dg_dx2(I33) = 0.0;

  // Set z-derivatives of covariant components
  for (int n = 0; n < NMETRIC; ++n) {
    dg_dx3(n) = 0.0;
  }
  return;
}
