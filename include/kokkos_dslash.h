/*
 * kokkos_dslash.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_DSLASH_H_
#define TEST_KOKKOS_KOKKOS_DSLASH_H_
#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "kokkos_defaults.h"
#include "kokkos_types.h"
#include "kokkos_spinproj.h"
#include "kokkos_matvec.h"
#include "kokkos_traits.h"
#include "kokkos_dslash_config.h"
namespace MG {




  // Try an N-dimensional threading policy for cache blocking
typedef Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<4,Kokkos::Experimental::Iterate::Left,Kokkos::Experimental::Iterate::Left>> t_policy;

enum DirIdx { T_MINUS=0, Z_MINUS=1, Y_MINUS=2, X_MINUS=3, X_PLUS=4, Y_PLUS=5, Z_PLUS=6, T_PLUS=7 };


#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)

   void ComputeSiteTable(int _n_xh, int _n_x, int _n_y, int _n_z, int _n_t,  Kokkos::View<int*[2][8],NeighLayout, MemorySpace> _table) {
		int num_sites =  _n_xh*_n_y*_n_z*_n_t;
			Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,num_sites), KOKKOS_LAMBDA(int site) {
		        for(int target_cb=0; target_cb < 2; ++target_cb) {
			     // Break down site index into xcb, y,z and t
			     IndexType tmp_yzt = site / _n_xh;
			     IndexType xcb = site - _n_xh * tmp_yzt;
			     IndexType tmp_zt = tmp_yzt / _n_y;
			     IndexType y = tmp_yzt - _n_y * tmp_zt;
			     IndexType t = tmp_zt / _n_z;
			     IndexType z = tmp_zt - _n_z * t;

			     // Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
			     IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

			     if( t > 0 ) {
			       _table(site,target_cb,T_MINUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1)));
			     }
			     else {
			       _table(site,target_cb,T_MINUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
			     }

			     if( z > 0 ) {
			       _table(site,target_cb,Z_MINUS) = xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,Z_MINUS) = xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
			     }

			     if( y > 0 ) {
			       _table(site,target_cb,Y_MINUS) = xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,Y_MINUS) = xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));
			     }

			     if ( x > 0 ) {
			       _table(site,target_cb,X_MINUS)= ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,X_MINUS)= ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
			     }

			     if ( x < _n_x - 1 ) {
			       _table(site,target_cb,X_PLUS) = ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,X_PLUS) = 0 + _n_xh*(y + _n_y*(z + _n_z*t));
			     }

			     if( y < _n_y-1 ) {
			       _table(site,target_cb,Y_PLUS) = xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,Y_PLUS) = xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
			     }

			     if( z < _n_z-1 ) {
			       _table(site,target_cb,Z_PLUS) = xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t));
			     }
			     else {
			       _table(site,target_cb,Z_PLUS) = xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
			     }

			     if( t < _n_t-1 ) {
			       _table(site,target_cb,T_PLUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1)));
			     }
			     else {
			       _table(site,target_cb,T_PLUS) = xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
			     }
			    } // target CB
		        });

	}
#endif

class SiteTable {
public:


	  SiteTable(int n_xh,
		    int n_y,
		    int n_z,
		    int n_t) : 
	 _n_x(2*n_xh),
	 _n_xh(n_xh),
	 _n_y(n_y),
	 _n_z(n_z),
	 _n_t(n_t) {

#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	   _table = Kokkos::View<int*[2][8],NeighLayout,MemorySpace>("table", n_xh*n_y*n_z*n_t);
	   ComputeSiteTable(n_xh, 2*n_xh, n_y, n_z, n_t, _table);
#endif
	}

	  KOKKOS_FORCEINLINE_FUNCTION
	  	int  coords_to_idx(const int& xcb, const int& y, const int& z, const int& t) const
	  	{
	  	  return xcb+_n_xh*(y + _n_y*(z + _n_z*t));
	  	}

#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	KOKKOS_INLINE_FUNCTION
	int NeighborTMinus(int site, int target_cb) const {
		return _table(site,target_cb,T_MINUS);
	}

	KOKKOS_INLINE_FUNCTION
	int NeighborTPlus(int site, int target_cb) const {
		return _table(site,target_cb,T_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborZMinus(int site, int target_cb) const {
		return _table(site,target_cb,Z_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborZPlus(int site, int target_cb) const {
		return _table(site,target_cb,Z_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborYMinus(int site, int target_cb) const {
		return _table(site,target_cb,Y_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborYPlus(int site, int target_cb) const {
		return _table(site,target_cb,Y_PLUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborXMinus(int site, int target_cb) const {
		return _table(site,target_cb,X_MINUS);
	}
	KOKKOS_INLINE_FUNCTION
	int NeighborXPlus(int site, int target_cb) const {
		return _table(site,target_cb,X_PLUS);
	}
#else



	KOKKOS_INLINE_FUNCTION
	int NeighborTMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( t > 0 ) ? xcb + _n_xh*(y + _n_y*(z + _n_z*(t-1))) : xcb + _n_xh*(y + _n_y*(z + _n_z*(_n_t-1)));
	}



	KOKKOS_INLINE_FUNCTION
	int NeighborZMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( z > 0 ) ? xcb + _n_xh*(y + _n_y*((z-1) + _n_z*t)) : xcb + _n_xh*(y + _n_y*((_n_z-1) + _n_z*t));
	}


	KOKKOS_INLINE_FUNCTION
	int NeighborYMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  ( y > 0 ) ? xcb + _n_xh*((y-1) + _n_y*(z + _n_z*t)) : xcb + _n_xh*((_n_y-1) + _n_y*(z + _n_z*t));

	}


	KOKKOS_INLINE_FUNCTION
	int NeighborXMinus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

		return  (x > 0) ? ((x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t)) : ((_n_x-1)/2) + _n_xh*(y + _n_y*(z + _n_z*t));
	}


	KOKKOS_INLINE_FUNCTION
	int NeighborXPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);

		return  (x < _n_x - 1) ? ((x+1)/2)  + _n_xh*(y + _n_y*(z + _n_z*t)) : 0 + _n_xh*(y + _n_y*(z + _n_z*t));
	}



	KOKKOS_INLINE_FUNCTION
	int NeighborYPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;



		return  (y < _n_y - 1) ? xcb + _n_xh*((y+1) + _n_y*(z + _n_z*t)) : xcb + _n_xh*(0 + _n_y*(z + _n_z*t));
	}


	KOKKOS_INLINE_FUNCTION
	int NeighborZPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;

		return  (z < _n_z - 1) ? xcb + _n_xh*(y + _n_y*((z+1) + _n_z*t)) : xcb + _n_xh*(y + _n_y*(0 + _n_z*t));
	}


	KOKKOS_INLINE_FUNCTION
	int NeighborTPlus(int site, int target_cb) const {
		// Break down site index into xcb, y,z and t
		IndexType tmp_yzt = site / _n_xh;
		IndexType xcb = site - _n_xh * tmp_yzt;
		IndexType tmp_zt = tmp_yzt / _n_y;
		IndexType y = tmp_yzt - _n_y * tmp_zt;
		IndexType t = tmp_zt / _n_z;
		IndexType z = tmp_zt - _n_z * t;


		return  (t < _n_t - 1) ? xcb + _n_xh*(y + _n_y*(z + _n_z*(t+1))) : xcb + _n_xh*(y + _n_y*(z + _n_z*(0)));
	}
#endif


	KOKKOS_INLINE_FUNCTION
	  SiteTable( const SiteTable& st):
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table(st._table),
#endif
	  _n_x(st._n_x),
	  _n_xh(st._n_xh),
	  _n_y(st._n_y),
	  _n_z(st._n_z),
	  _n_t(st._n_t) {}

	KOKKOS_INLINE_FUNCTION
	  SiteTable& operator=(const  SiteTable& st) {
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	  _table = st._table;
#endif
	  _n_x = st._n_x;
	  _n_xh = st._n_xh;
	  _n_y = st._n_y;
	  _n_z = st._n_z;
	  _n_t = st._n_t;

	  return *this;
	}

private:
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
	Kokkos::View<int*[2][8], NeighLayout,MemorySpace > _table;
#endif
       int _n_x;
       int _n_xh;
       int _n_y;
       int _n_z;
       int _n_t;

};




 template<typename GT, typename ST, typename TST, const int isign, const int target_cb>
   struct DslashFunctor { 

     SpinorView<ST> s_in;
     GaugeView<GT> g_in_src_cb;
     GaugeView<GT> g_in_target_cb;
     SpinorView<ST> s_out;
     int num_sites;
     int sites_per_team;
     SiteTable neigh_table;

#if defined (MG_FLAT_PARALLEL_DSLASH)
     KOKKOS_FORCEINLINE_FUNCTION
     void operator()( int site ) const {

#elif defined (MG_KOKKOS_USE_MDRANGE)
     KOKKOS_FORCEINLINE_FUNCTION
     void operator()(const int& xcb, const int& y, const int& z, const int& t) const
     {

       int site = neigh_table.coords_to_idx(xcb,y,z,t);

#else
     KOKKOS_FORCEINLINE_FUNCTION
     void operator()(const TeamHandle& team) const {
		    const int start_idx = team.league_rank()*sites_per_team;
		    const int end_idx = start_idx + sites_per_team  < num_sites ? start_idx + sites_per_team : num_sites;

		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int site) {
#endif

		     // Warning: GCC Alignment Attribute!
		    // Site Sum: Not a true Kokkos View
			SpinorSiteView<TST> res_sum;// __attribute__((aligned(64)));
			
		    // Temporaries: Not a true Kokkos View
			HalfSpinorSiteView<TST> proj_res; // __attribute__((aligned(64)));
			HalfSpinorSiteView<TST> mult_proj_res; // __attribute__((aligned(64)));
		    

		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
			ComplexZero(res_sum(color,spin));
		      }
		    }
			
		    // T - minus
		    KokkosProjectDir3<ST,TST,isign>(s_in, proj_res,neigh_table.NeighborTMinus(site,target_cb));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table.NeighborTMinus(site,target_cb),3);
		    KokkosRecons23Dir3<TST,isign>(mult_proj_res,res_sum);

		    // Z - minus
		    KokkosProjectDir2<ST,TST,isign>(s_in, proj_res,neigh_table.NeighborZMinus(site,target_cb));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table.NeighborZMinus(site,target_cb),2);
		    KokkosRecons23Dir2<TST,isign>(mult_proj_res,res_sum);
		    
		    // Y - minus
		    KokkosProjectDir1<ST,TST,isign>(s_in, proj_res,neigh_table.NeighborYMinus(site,target_cb));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table.NeighborYMinus(site,target_cb),1);
		    KokkosRecons23Dir1<TST,isign>(mult_proj_res,res_sum);


		    // X - minus
		    KokkosProjectDir0<ST,TST,isign>(s_in, proj_res,neigh_table.NeighborXMinus(site,target_cb));
		    mult_adj_u_halfspinor<GT,TST>(g_in_src_cb,proj_res,mult_proj_res,neigh_table.NeighborXMinus(site,target_cb),0);
		    KokkosRecons23Dir0<TST,isign>(mult_proj_res,res_sum);

		    
		    // X - plus
		    KokkosProjectDir0<ST,TST,-isign>(s_in,proj_res,neigh_table.NeighborXPlus(site,target_cb));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,0);
		    KokkosRecons23Dir0<TST,-isign>(mult_proj_res, res_sum);

		    
		    // Y - plus
		    KokkosProjectDir1<ST,TST,-isign>(s_in,proj_res,neigh_table.NeighborYPlus(site,target_cb));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,1);
		    KokkosRecons23Dir1<TST,-isign>(mult_proj_res, res_sum);


		    // Z - plus
		    KokkosProjectDir2<ST,TST,-isign>(s_in,proj_res,neigh_table.NeighborZPlus(site,target_cb));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,2);
		    KokkosRecons23Dir2<TST,-isign>(mult_proj_res, res_sum);

		    
		    // T - plus
		    KokkosProjectDir3<ST,TST,-isign>(s_in,proj_res,neigh_table.NeighborTPlus(site,target_cb));
		    mult_u_halfspinor<GT,TST>(g_in_target_cb,proj_res,mult_proj_res,site,3);
		    KokkosRecons23Dir3<TST,-isign>(mult_proj_res, res_sum);

		    // Stream out spinor
 		    for(int color=0; color < 3; ++color) {
		      for(int spin=0; spin < 4; ++spin) {
		    	  Stream(s_out(site,color,spin),res_sum(color,spin));
		      }
		    }
#if !defined (MG_FLAT_PARALLEL_DSLASH) && !defined(MG_KOKKOS_USE_MDRANGE)
	      });
#endif 
      }

   };

template<typename GT, typename ST, typename TST>
class KokkosDslash {
 public:
	const LatticeInfo& _info;

	SiteTable _neigh_table;
	const int _sites_per_team;
public:

KokkosDslash(const LatticeInfo& info, int sites_per_team=1) : _info(info),
	  _neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]),
	  _sites_per_team(sites_per_team)
	  {}

#ifndef MG_KOKKOS_USE_MDRANGE	
	void operator()(const KokkosCBFineSpinor<ST,4>& fine_in,
		      const KokkosFineGaugeField<GT>& gauge_in,
		      KokkosCBFineSpinor<ST,4>& fine_out,
		      int plus_minus) const
	{
	  int source_cb = fine_in.GetCB();
	  int target_cb = (source_cb == EVEN) ? ODD : EVEN;
	  const SpinorView<ST>& s_in = fine_in.GetData();
	  const GaugeView<GT>& g_in_src_cb = (gauge_in(source_cb)).GetData();
	  const GaugeView<GT>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
	  SpinorView<ST>& s_out = fine_out.GetData();
	  const int num_sites = _info.GetNumCBSites();

#if !defined (MG_FLAT_PARALLEL_DSLASH)
	  ThreadExecPolicy policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
#endif
	  if( plus_minus == 1 ) {
	    if (target_cb == 0 ) {
	      DslashFunctor<GT,ST,TST,1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team,_neigh_table};

#if defined (MG_FLAT_PARALLEL_DSLASH)
	Kokkos::parallel_for(num_sites,f);
#else
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
#endif
	    }
	    else {
	      DslashFunctor<GT,ST,TST,1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};

#if defined (MG_FLAT_PARALLEL_DSLASH)
	Kokkos::parallel_for(num_sites,f);
#else
	     Kokkos::parallel_for(policy, f); // Outer Lambda 
#endif
	    }
	  }
	  else {
	    if( target_cb == 0 ) { 
	      DslashFunctor<GT,ST,TST,-1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};
#if defined (MG_FLAT_PARALLEL_DSLASH)
	Kokkos::parallel_for(num_sites,f);
#else
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
#endif
	    }
	    else {
	      DslashFunctor<GT,ST,TST,-1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table };
#if defined (MG_FLAT_PARALLEL_DSLASH)
	Kokkos::parallel_for(num_sites,f);
#else
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
#endif
	    }
	  }
	  
	}

#else
	void operator()(const KokkosCBFineSpinor<ST,4>& fine_in,
		      const KokkosFineGaugeField<GT>& gauge_in,
		      KokkosCBFineSpinor<ST,4>& fine_out,
			int plus_minus, const IndexArray& blocks) const
	{
	  int source_cb = fine_in.GetCB();
	  int target_cb = (source_cb == EVEN) ? ODD : EVEN;
	  const SpinorView<ST>& s_in = fine_in.GetData();
	  const GaugeView<GT>& g_in_src_cb = (gauge_in(source_cb)).GetData();
	  const GaugeView<GT>&  g_in_target_cb = (gauge_in(target_cb)).GetData();
	  SpinorView<ST>& s_out = fine_out.GetData();
	  IndexArray cb_latdims = _info.GetCBLatticeDimensions();
	  const int num_sites = _info.GetNumCBSites();
	  MDPolicy policy({0,0,0,0},
			  {cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]},
			  {blocks[0],blocks[1],blocks[2],blocks[3]}
			  );

	  if( plus_minus == 1 ) {
	    if (target_cb == 0 ) {
	      DslashFunctor<GT,ST,TST,1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team,_neigh_table};


	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	    else {
	      DslashFunctor<GT,ST,TST,1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};

	     Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	  }
	  else {
	    if( target_cb == 0 ) { 
	      DslashFunctor<GT,ST,TST,-1,0> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table};
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	    else {
	      DslashFunctor<GT,ST,TST,-1,1> f = {s_in, g_in_src_cb, g_in_target_cb, s_out,
	    		  num_sites, _sites_per_team, _neigh_table };
	      Kokkos::parallel_for(policy, f); // Outer Lambda 
	    }
	  }
	  
	}
#endif
};




};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
