/*
 * kokkos_matvec.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_MATVEC_H_
#define TEST_KOKKOS_MATVEC_H_

#include "kokkos_defaults.h"
#include "kokkos_constants.h"
#include "kokkos_types.h"
#include "kokkos_ops.h"

namespace MG
{


	template<typename T>
	KOKKOS_INLINE_FUNCTION
	void mult_u_halfspinor(const GaugeSiteView<T>& u,
			const HalfSpinorSiteView<T>& v_in,
			HalfSpinorSiteView<T>& v_out)
	{

	  for(int row=0; row < 3; ++row) {
	    for(int spin = 0; spin < 2; ++spin) {
	      ComplexZero(v_out(row,spin));
	    }

	    for(int col=0; col < 3; ++col) {
	      for(int spin = 0; spin < 2; ++spin) {
					//v_out(row,spin) += u(row,col)*v_in(col);
					// complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
					//                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
					//
		ComplexCMadd(v_out(row,spin), u(row,col), v_in(col,spin));
	      }
	    }
	  }


	}


	template<typename T>
	KOKKOS_INLINE_FUNCTION
	void mult_adj_u_halfspinor(const GaugeSiteView<T>& u,
			const HalfSpinorSiteView<T>& v_in,
			HalfSpinorSiteView<T>& v_out)
	{
	  for(int row=0; row < 3; ++row) {
	    for(int spin = 0; spin < 2; ++spin) {
	      ComplexZero(v_out(row,spin));
	    }
	  }

	  for(int col=0; col < 3; ++col) {
	    for(int row=0; row < 3; ++row) {
	      //v_out(row,spin) += u(row,col)*v_in(col);
	      // complex mul:   real_part: a(K_RE)*b(K_RE)-a(K_IM)*b(K_IM)
	      //                imag_part: a(K_RE)*b(K_IM) + a(K_IM)*b(K_RE);
	      //
	      for(int spin=0; spin < 2; ++spin) {
		ComplexConjMadd(v_out(row,spin), u(col,row), v_in(col,spin));
	      }
	    }
	  }
	}

	template<typename GT, typename ST, typename TST>
	void KokkosMVLattice(const KokkosCBFineGaugeField<GT>& u_in,
			const KokkosCBFineSpinor<ST,2>& hspinor_in,
			int dir,
			     const KokkosCBFineSpinor<ST,2>& hspinor_out, int _sites_per_team = 2)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<ST> hspinor_in_view = hspinor_in.GetData();
		GaugeView<GT> u = u_in.GetData();
		HalfSpinorView<ST> hspinor_out_view = hspinor_out.GetData();

// #ifndef MG_FLAT_PARALLEL_DSLASH
#if 0
		const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {
#else
	Kokkos::parallel_for(num_sites, KOKKOS_LAMBDA(const int i) { 
#endif
				// Site local workspace...
				HalfSpinorSiteView<TST> site_in;
				GaugeSiteView<TST> u_site;
				for(int col=0; col <3; ++col) {
					for(int spin=0; spin < 2; ++spin) {
						Load(site_in(col,spin), hspinor_in_view(i,col,spin));
					}
				}

			        load<TST,GT>(u_site,u,i,dir);
				HalfSpinorSiteView<TST> site_out;
				mult_u_halfspinor<TST>(u_site, site_in, site_out);

				// Write out
				for(int col=0; col < 3; ++col) {
					for(int spin=0; spin < 2; ++spin ) {
						Store(hspinor_out_view(i,col,spin),site_out(col,spin));
					}
				}
		});
//#ifndef MG_FLAT_PARALLEL_DSLASH
#if 0
		  });
#endif

	}



	template<typename GT, typename ST, typename TST>
	void KokkosHVLattice(const KokkosCBFineGaugeField<GT>& u_in,
				  const KokkosCBFineSpinor<ST,2>& hspinor_in,
				  int dir,
			     const KokkosCBFineSpinor<ST,2>& hspinor_out,
			     int _sites_per_team = 2)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<ST> hspinor_in_view = hspinor_in.GetData();
		HalfSpinorView<ST> hspinor_out_view = hspinor_out.GetData();
		GaugeView<GT> u = u_in.GetData();

//#ifndef MG_FLAT_PARALLEL_DSLASH
#if 0
		const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {
#else
		Kokkos::parallel_for(num_sites,KOKKOS_LAMBDA(const int i) { 
#endif
			// Site local workspace...
			HalfSpinorSiteView<TST> site_in;
			for(int col=0; col <3; ++col) {
				for(int spin=0; spin < 2; ++spin) {
					Load(site_in(col,spin), hspinor_in_view(i,col,spin));
				}
			}
			GaugeSiteView<TST> u_site;
			load<TST,GT>(u_site,u,i,dir);
			HalfSpinorSiteView<TST> site_out;
			mult_adj_u_halfspinor<TST>(u_site, site_in, site_out);

			// Write out
			for(int col=0; col < 3; ++col) {
				for(int spin=0; spin < 2; ++spin ) {
					Store(hspinor_out_view(i,col,spin), site_out(col,spin));
				}
			}
		});
// #ifndef MG_FLAT_PARALLEL_DSLASH
#if 0
	});
#endif

	}

}


#endif /* TEST_KOKKOS_MATVEC_H_ */
