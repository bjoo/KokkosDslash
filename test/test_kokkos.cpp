#include "kokkos_dslash_config.h"

#include "gtest/gtest.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

//#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"

#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

#if 0
TEST(TestKokkos, TestLatticeInitialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;


}
#endif

#if 0
TEST(TestKokkos, TestSpinorInitialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	KokkosCBFineSpinor<MGComplex<float>,4> cb_spinor_e(info, EVEN);
	KokkosCBFineSpinor<MGComplex<float>,4> cb_spinor_o(info, ODD);

	KokkosCBFineGaugeField<MGComplex<float>> gauge_field_even(info, EVEN);
	KokkosCBFineGaugeField<MGComplex<float>> gauge_field_odd(info,ODD);
 }
#endif


#if 0
TEST(TestKokkos, TestQDPCBSpinorImportExport)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	LatticeFermion qdp_out;
	LatticeFermion qdp_in;

	gaussian(qdp_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosCBFineSpinor<MGComplex<REAL>,4>  kokkos_spinor_e(info, EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,4>  kokkos_spinor_o(info, ODD);
	{
		qdp_out = zero;
		// Import Checkerboard, by checkerboard
		QDPLatticeFermionToKokkosCBSpinor<REAL,LatticeFermion>(qdp_in, kokkos_spinor_e);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion<REAL,LatticeFermion>(kokkos_spinor_e,qdp_out);
		qdp_out[rb[0]] -= qdp_in;

		// Elements of QDP_out should now be zero.
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		qdp_out = zero;
		QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_spinor_o);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_o,qdp_out);

		qdp_out[rb[1]] -= qdp_in;
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

}
#endif

#if 0
TEST(TestKokkos, TestQDPCBHalfSpinorImportExport)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	LatticeHalfFermion qdp_out;
	LatticeHalfFermion qdp_in;

	gaussian(qdp_in);

	LatticeInfo info(latdims,2,3,NodeInfo());
	KokkosCBFineSpinor<MGComplex<REAL>,2>  kokkos_hspinor_e(info, EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,2>  kokkos_hspinor_o(info, ODD);
	{
		qdp_out = zero;
		// Import Checkerboard, by checkerboard
		QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_e);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_e,qdp_out);
		qdp_out[rb[0]] -= qdp_in;

		// Elements of QDP_out should now be zero.
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		qdp_out = zero;
		QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_o);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_o,qdp_out);

		qdp_out[rb[1]] -= qdp_in;
		double norm_diff = toDouble(sqrt(norm2(qdp_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

}
#endif

#if 0
TEST(TestKokkos, TestQDPCBSpinorImportExportVec)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	multi1d<LatticeFermionF> qdp_out(8);
	multi1d<LatticeFermionF> qdp_in(8);

	for(int v=0; v < 8; ++v) {
		gaussian(qdp_in[v]);
	}

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosCBFineSpinorVec<REAL32,8>  kokkos_spinor_e(info, EVEN);
	KokkosCBFineSpinorVec<REAL32,8>  kokkos_spinor_o(info, ODD);
	{
		for(int v=0; v < 8; ++v) {
			qdp_out[v] = zero;
		}

		// Import Checkerboard, by checkerboard
		QDPLatticeFermionToKokkosCBSpinor<REAL32,8,LatticeFermionF>(qdp_in, kokkos_spinor_e);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion<REAL32,8,LatticeFermionF>(kokkos_spinor_e,qdp_out);

		for(int v=0; v < 8; ++v) {
			qdp_out[v][rb[0]] -= qdp_in[v];

			// Elements of QDP_out should now be zero.
			double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
			MasterLog(INFO, "v=%d norm_diff = %lf", v, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

	{
		for(int v=0; v < 8; ++v) {
				qdp_out[v] = zero;
			}

		QDPLatticeFermionToKokkosCBSpinor<REAL32,8,LatticeFermionF>(qdp_in, kokkos_spinor_o);
		// Export back out
		KokkosCBSpinorToQDPLatticeFermion<REAL32,8,LatticeFermionF>(kokkos_spinor_o,qdp_out);

		for(int v=0; v < 8; ++v) {
			qdp_out[v][rb[1]] -= qdp_in[v];
			double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
			MasterLog(INFO, "v=%d norm_diff = %lf", v, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}
#endif


#if 0
TEST(TestKokkos, TestQDPCBHalfSpinorImportExportVec)
{
	IndexArray latdims={{4,6,8,10}};
	initQDPXXLattice(latdims);

	multi1d<LatticeHalfFermionF> qdp_out(8);
	multi1d<LatticeHalfFermionF> qdp_in(8);

	for(int v=0; v < 8; ++v) {
		gaussian(qdp_in[v]);
	}

	LatticeInfo info(latdims,2,3,NodeInfo());
	KokkosCBFineHalfSpinorVec<REAL32,8>  kokkos_hspinor_e(info, EVEN);
	KokkosCBFineHalfSpinorVec<REAL32,8>  kokkos_hspinor_o(info, ODD);
	{
		for(int v=0; v < 8; ++v) {
			qdp_out[v] = zero;
		}
		// Import Checkerboard, by checkerboard
		QDPLatticeHalfFermionToKokkosCBSpinor2<REAL32,8,LatticeHalfFermionF>(qdp_in, kokkos_hspinor_e);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion<REAL32,8,LatticeHalfFermionF>(kokkos_hspinor_e,qdp_out);

		for(int v=0; v < 8; ++v) {
			qdp_out[v][rb[0]] -= qdp_in[v];


		// Elements of QDP_out should now be zero.
			double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
			MasterLog(INFO, "v=%d norm_diff = %lf", v,norm_diff);

			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

	{
		for(int v=0; v < 8; ++v ) {
			qdp_out[v] = zero;
		}

		QDPLatticeHalfFermionToKokkosCBSpinor2<REAL32,8,LatticeHalfFermionF>(qdp_in, kokkos_hspinor_o);
		// Export back out
		KokkosCBSpinor2ToQDPLatticeHalfFermion<REAL32,8,LatticeHalfFermionF>(kokkos_hspinor_o,qdp_out);

		for(int v=0; v < 8; ++v) {
			qdp_out[v][rb[1]] -= qdp_in[v];

			double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
			MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}
#endif


#if 1
TEST(TestKokkos, TestSpinProject)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	LatticeFermion qdp_in;
	LatticeHalfFermion qdp_out;
	LatticeHalfFermion kokkos_out;


	gaussian(qdp_in);
	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_in(info,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,2> kokkos_hspinor_out(hinfo,EVEN);

	QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_in);
	
	{
	  // sign = -1 dir = 0
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,-1);
	  qdp_out[rb[0]] = spinProjectDir0Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,0,-1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = -1 dir = 1
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,-1);
	  qdp_out[rb[0]] = spinProjectDir1Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,1,-1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = -1 dir = 2
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,-1);
	  qdp_out[rb[0]] = spinProjectDir2Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,2,-1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = -1 dir = 3
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,-1);
	  qdp_out[rb[0]] = spinProjectDir3Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,3,-1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = 1 dir = 0
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,1);
	  qdp_out[rb[0]] = spinProjectDir0Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,0,1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = 1 dir = 1
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,1);
	  qdp_out[rb[0]] = spinProjectDir1Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,1,1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = 1 dir = 2
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,1);
	  qdp_out[rb[0]] = spinProjectDir2Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,2,1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  // sign = 1 dir = 3
	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,+1);
	  qdp_out[rb[0]] = spinProjectDir3Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosProjectLattice<MGComplex<REAL>,MGComplex<REAL>,3,1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  qdp_out[rb[0]] -= kokkos_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

}
#endif

#if !defined( MG_USE_HIP ) && ! defined( MG_USE_OPENMP_TARGET)
TEST(TestKokkos, TestSpinProjectVec)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	static const int V = 8;
	multi1d<LatticeFermionF> qdp_in(V);
	multi1d<LatticeHalfFermionF> qdp_out(V); 
	multi1d<LatticeHalfFermionF>  kokkos_out(V);


	for(int v=0; v < V; ++ v) gaussian(qdp_in[v]);

	// These will all use SIMDComplex
	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,4> kokkos_in(info,EVEN);
	KokkosCBFineSpinor<SIMDComplex<REAL32,V>,2> kokkos_hspinor_out(hinfo,EVEN);

	QDPLatticeFermionToKokkosCBSpinor(qdp_in, kokkos_in);

	{
	  // sign = -1 dir = 0

	  MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,-1);
	  for(int v=0; v < V; ++v) {
		  qdp_out[v][rb[0]] = spinProjectDir0Minus(qdp_in[v]);
		  qdp_out[v][rb[1]] = zero;
	  }

	  KokkosProjectLattice<SIMDComplex<REAL32,V>,ThreadSIMDComplex<REAL32,V>,0,-1>(kokkos_in,kokkos_hspinor_out);
	  KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
	  for(int v=0; v < V; ++v) { 
	    qdp_out[v][rb[0]] -= kokkos_out[v];
	    double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	    MasterLog(INFO, "v=%d norm_diff = %lf", v,norm_diff);
	    ASSERT_LT( norm_diff, 1.0e-5);
	  }
	}


}
#endif

#if 1
TEST(TestKokkos, TestSpinRecons)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	LatticeHalfFermion qdp_in;
	LatticeFermion     qdp_out;
	LatticeFermion     kokkos_out;


	gaussian(qdp_in);

	KokkosCBFineSpinor<MGComplex<REAL>,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_spinor_out(info,EVEN);

	QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_in);
	
	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, -1);
	  qdp_out[rb[0]] = spinReconstructDir0Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,0,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, -1);
	  qdp_out[rb[0]] = spinReconstructDir1Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,1,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, -1);
	  qdp_out[rb[0]] = spinReconstructDir2Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,2,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, -1);
	  qdp_out[rb[0]] = spinReconstructDir3Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,3,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}


	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, +1);
	  qdp_out[rb[0]] = spinReconstructDir0Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,0,1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, +1);
	  qdp_out[rb[0]] = spinReconstructDir1Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,1,1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, +1);
	  qdp_out[rb[0]] = spinReconstructDir2Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,2,1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, +1);
	  qdp_out[rb[0]] = spinReconstructDir3Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  KokkosReconsLattice<MGComplex<REAL>,MGComplex<REAL>,3,1>(kokkos_hspinor_in,kokkos_spinor_out);
	  KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	  qdp_out[rb[0]] -= kokkos_out;
	  
	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

}
#endif


#if !defined MG_USE_HIP && !defined(MG_USE_OPENMP_TARGET)
TEST(TestKokkos, TestSpinReconsVec)
{
	IndexArray latdims={{4,2,2,4}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	static const int V=8;

	multi1d<LatticeHalfFermion> qdp_in(V);
	multi1d<LatticeFermion>     qdp_out(V);
	multi1d<LatticeFermion>     kokkos_out(V);

	for(int v=0; v < V; ++v) { 
	  gaussian(qdp_in[v]);
	}

	KokkosCBFineSpinor<SIMDComplex<REAL,V>,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<SIMDComplex<REAL,V>,4> kokkos_spinor_out(info,EVEN);

	QDPLatticeHalfFermionToKokkosCBSpinor2(qdp_in, kokkos_hspinor_in);
	
	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, -1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir0Minus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,0,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}

{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, -1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir1Minus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,1,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, -1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir2Minus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,2,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, -1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir3Minus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,3,-1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}



	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, 1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir0Plus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,0,1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}

{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, 1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir1Plus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,1,1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, 1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir2Plus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,2,1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
 {
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, 1);
	    for(int v=0; v < V; ++v) {
	      qdp_out[v][rb[0]] = spinReconstructDir3Plus(qdp_in[v]);
	      qdp_out[v][rb[1]] = zero;
	    }

	    KokkosReconsLattice<SIMDComplex<REAL,V>,ThreadSIMDComplex<REAL,V>,3,1>(kokkos_hspinor_in,kokkos_spinor_out);
	    KokkosCBSpinorToQDPLatticeFermion(kokkos_spinor_out,kokkos_out);

	    for(int v=0; v < V; ++v) { 
	      qdp_out[v][rb[0]] -= kokkos_out[v];
	
	      double norm_diff = toDouble(sqrt(norm2(qdp_out[v])));
	      MasterLog(INFO, "v=%d norm_diff = %lf",v, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}


}
#endif


#if 0
TEST(TestKokkos, TestQDPCBGaugeFIeldImportExport)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	multi1d<LatticeColorMatrix> gauge_out;  // Basic uninitialized

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosCBFineGaugeField<MGComplex<REAL>>  kokkos_gauge_e(info, EVEN);
	KokkosCBFineGaugeField<MGComplex<REAL>>  kokkos_gauge_o(info, ODD);
	{
		// Import Checkerboard, by checkerboard
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);
		// Export back out
		KokkosCBGaugeFieldToQDPGaugeField(kokkos_gauge_e, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu])[rb[0]] -= gauge_in[mu];

			// In this test, the copy back initialized gauge_out
			// so its non-checkerboard piece is junk
			// Take norm over only the checkerboard of interest
			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu], rb[0])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

		{
		// gauge out is now allocated, so zero it
		for(int mu=0; mu < n_dim; ++mu) {
			gauge_out[mu] = zero;
		}
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_o);
		// Export back out
		KokkosCBGaugeFieldToQDPGaugeField(kokkos_gauge_o, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu])[rb[1]] -= gauge_in[mu];
			// Other checkerboard was zeroed initially.

			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}
#endif


#if 0
TEST(TestKokkos, TestQDPGaugeFIeldImportExport)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);

	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	multi1d<LatticeColorMatrix> gauge_out;  // Basic uninitialized

	LatticeInfo info(latdims,4,3,NodeInfo());
	KokkosFineGaugeField<MGComplex<REAL>>  kokkos_gauge(info);
	{
		// Import Checkerboard, by checkerboard
		QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
		// Export back out
		KokkosGaugeFieldToQDPGaugeField(kokkos_gauge, gauge_out);

		for(int mu=0; mu < n_dim; ++mu) {
			(gauge_out[mu]) -= gauge_in[mu];

			// In this test, the copy back initialized gauge_out
			// so its non-checkerboard piece is junk
			// Take norm over only the checkerboard of interest
			double norm_diff = toDouble(sqrt(norm2(gauge_out[mu])));
			MasterLog(INFO, "norm_diff[%d] = %lf", mu, norm_diff);
			ASSERT_LT( norm_diff, 1.0e-5);
		}
	}

}
#endif


#if 1
TEST(TestKokkos, TestMultHalfSpinor)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	LatticeHalfFermion psi_in;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	// Type for Gauge -- scalar
	using GT = MGComplex<REAL>;

	// Type for Spinor -- scalar
	using ST = MGComplex<REAL>;

	// Type for ThreadSpecific spinor - scalar
	using TST = MGComplex<REAL>;

	KokkosCBFineSpinor<ST,2> kokkos_hspinor_in(hinfo,EVEN);
	KokkosCBFineSpinor<ST,2> kokkos_hspinor_out(hinfo,EVEN);
	KokkosCBFineGaugeField<GT>  kokkos_gauge_e(info, EVEN);


	// Import Gauge Field
	QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);

	LatticeHalfFermion psi_out, kokkos_out;
	MasterLog(INFO, "Testing MV");
	{
		psi_out[rb[0]] = gauge_in[0]*psi_in;
		psi_out[rb[1]] = zero;


		// Import Gauge Field
		QDPGaugeFieldToKokkosCBGaugeField(gauge_in, kokkos_gauge_e);
		// Import input halfspinor
		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);


		KokkosMVLattice<GT,ST,TST>(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;
		double norm_diff = toDouble(sqrt(norm2(psi_out)));

		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	// ********* TEST ADJOINT ****************
	{

		psi_out[rb[0]] = adj(gauge_in[0])*psi_in;
		psi_out[rb[1]] = zero;

		// Import input halfspinor -- Gauge still imported

		QDPLatticeHalfFermionToKokkosCBSpinor2(psi_in, kokkos_hspinor_in);
		KokkosHVLattice<GT,ST,TST>(kokkos_gauge_e, kokkos_hspinor_in, 0, kokkos_hspinor_out);

		// Export result HalfFermion
		KokkosCBSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		psi_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(psi_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}
}

#endif

#if defined(MG_FLAT_PARALLEL_DSLASH)
TEST(TestKokkos, TestDslash)
{
	IndexArray latdims={{16,16,16,16}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_spinor_even(info,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_spinor_odd(info,ODD);
	KokkosFineGaugeField<MGComplex<REAL>>  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);


	int per_team = 2;
	KokkosDslash<MGComplex<REAL>,MGComplex<REAL>,MGComplex<REAL>> D(info,per_team);
	MasterLog(INFO, "per_team=%d", per_team);

	LatticeFermion psi_out = zero;
	LatticeFermion  kokkos_out=zero;
	for(int cb=0; cb < 2; ++cb) {
	  KokkosCBFineSpinor<MGComplex<REAL>,4>& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
	  KokkosCBFineSpinor<MGComplex<REAL>,4>& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;
	  
	    for(int isign=-1; isign < 2; isign+=2) {
	      
	      // In the Host
	      psi_out = zero;
	      
	      // Target cb=1 for now.
	      dslash(psi_out,gauge_in,psi_in,isign,cb);
	      
	      QDPLatticeFermionToKokkosCBSpinor(psi_in, in_spinor);
	      
	      
	      D(in_spinor,kokkos_gauge,out_spinor,isign);
	      
	      kokkos_out = zero;
	      KokkosCBSpinorToQDPLatticeFermion(out_spinor, kokkos_out);
	      
	      // Check Diff on Odd
	      psi_out[rb[cb]] -= kokkos_out;
	      double norm_diff = toDouble(sqrt(norm2(psi_out,rb[cb])));
	      
	      MasterLog(INFO, "sites_per_team=%d norm_diff = %lf", per_team, norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
	
}
#endif

#ifdef MG_KOKKOS_USE_MDRANGE
TEST(TestKokkos, TestDslashMDRange)
{
	IndexArray latdims={{32,32,32,32}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_spinor_even(info,EVEN);
	KokkosCBFineSpinor<MGComplex<REAL>,4> kokkos_spinor_odd(info,ODD);
	KokkosFineGaugeField<MGComplex<REAL>>  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
	int per_team=32; // Arbitraty
	KokkosDslash<MGComplex<REAL>,MGComplex<REAL>,MGComplex<REAL>> D(info,per_team);

	IndexArray blockings[6] = { { 1,1,1,1 },
                                    { 2,2,2,4 },
                                    { 4,4,1,2 },
                                    { 4,2,8,4 },
                                    { 8,4,1,4 },
                                    { 16,4,1,1} };

	for(int b=0; b<6; ++b) {
	  IndexType bx = blockings[b][0];
	  IndexType by = blockings[b][1];
	  IndexType bz = blockings[b][2];
	  IndexType bt = blockings[b][3];

	  if (bx*by*bz*bt > 256 ) continue;


	LatticeFermion psi_out = zero;
	LatticeFermion  kokkos_out=zero;
	for(int cb=0; cb < 2; ++cb) {
	  KokkosCBFineSpinor<MGComplex<REAL>,4>& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
	  KokkosCBFineSpinor<MGComplex<REAL>,4>& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;
	  
	    for(int isign=-1; isign < 2; isign+=2) {
	      
	      // In the Host
	      psi_out = zero;
	      
	      // Target cb=1 for now.
	      dslash(psi_out,gauge_in,psi_in,isign,cb);
	      
	      QDPLatticeFermionToKokkosCBSpinor(psi_in, in_spinor);
	      
	      MasterLog(INFO, "D with blocking=(%d,%d,%d,%d),",bx,by,bz,bt);
	      D(in_spinor,kokkos_gauge,out_spinor,isign,{bx,by,bz,bt});
	      
	      kokkos_out = zero;
	      KokkosCBSpinorToQDPLatticeFermion(out_spinor, kokkos_out);
	      
	      // Check Diff on Odd
	      psi_out[rb[cb]] -= kokkos_out;
	      double norm_diff = toDouble(sqrt(norm2(psi_out,rb[cb])));
	      
	      MasterLog(INFO, "norm_diff = %lf", norm_diff);
	      ASSERT_LT( norm_diff, 1.0e-5);
	    }
	}
	}// blockings
}
#endif

#if !defined(MG_USE_HIP) && !defined(MG_USE_OPENMP_TARGET)

TEST(TestKokkos, TestDslashVec)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	static const int VLen=8;

	multi1d<LatticeFermion> psi_in(VLen);
	for(int v=0; v < VLen; ++v) {
		gaussian(psi_in[v]);
	}

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	KokkosCBFineSpinor<SIMDComplex<REAL,VLen>,4> kokkos_spinor_even(info,EVEN);
	KokkosCBFineSpinor<SIMDComplex<REAL,VLen>,4> kokkos_spinor_odd(info,ODD);
	KokkosFineGaugeField<MGComplex<REAL>>  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);


	for(int per_team=2; per_team < 4; per_team*=2) { 
	  KokkosDslash<MGComplex<REAL>,SIMDComplex<REAL,VLen>,ThreadSIMDComplex<REAL,VLen>> D(info,per_team);

	  multi1d<LatticeFermion> psi_out(VLen);
	  multi1d<LatticeFermion> kokkos_out(VLen);
	  for(int cb=0; cb < 2; ++cb) {
	    KokkosCBFineSpinor<SIMDComplex<REAL,VLen>,4>& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
	    KokkosCBFineSpinor<SIMDComplex<REAL,VLen>,4>& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;
	    
	    for(int isign=-1; isign < 2; isign+=2) {
	      
	      // In the Host
	      for(int v=0; v < VLen; ++v) {
		psi_out[v] = zero;
		
		kokkos_out[v] = zero;
		
		// Prep reference data
		dslash(psi_out[v],gauge_in,psi_in[v],isign,cb);
	      }
	      
	      // Import
	      QDPLatticeFermionToKokkosCBSpinor(psi_in, in_spinor);
	      
	      // Vector Dslash
	      D(in_spinor,kokkos_gauge,out_spinor,isign);
	      
	      // Export
	      KokkosCBSpinorToQDPLatticeFermion<>(out_spinor, kokkos_out);
	      MasterLog(INFO, "Sites Per Team=%d", per_team);
	      for(int v=0; v < VLen; ++v) {
		// Check Diff on Odd
		psi_out[v][rb[cb]] -= kokkos_out[v];
		double norm_diff = toDouble(sqrt(norm2(psi_out[v],rb[cb])));
		
		MasterLog(INFO, "\t v=%d norm_diff = %lf", v,norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	      }
	    }
	  }
	}
}
#endif

