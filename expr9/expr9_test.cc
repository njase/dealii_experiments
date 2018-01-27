//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
//			 - compare values, gradients and hessians
// This is an experiment to validate my implementation by using existing implementation of dealii RT elements
//so that functional bugs are removed
//The code base is adapted from matrix_vector_stokes.cc and changed to RT
/////////////////

#include "tests.h"

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

#include "create_mesh.h"

#include <iostream>
#include <complex>
#include <vector>


//More basic tests
//Comparison of general output of FEValues (MatrixFree)
// and FEEvaluationGen+MatrixFree(for RT)


std::ofstream logfile("output");

void unit_cell_mesh (Triangulation<3> &tria,
                  const double scale_grid = 1.)
{
  const unsigned int dim = 3;
  std::vector<Point<dim> > points (8);

  // build the mesh layer by layer from points

  // 1. cube cell
  points[0] = Point<dim> (0,0,0);
  points[1] = Point<dim> (0,1.,0);
  points[2] = Point<dim> (0,0,1);
  points[3] = Point<dim> (0,1.,1);
  points[4] = Point<dim> (1.,0,0);
  points[5] = Point<dim> (1.,1.,0);
  points[6] = Point<dim> (1.,0,1);
  points[7] = Point<dim> (1.,1.,1);

  if (scale_grid != 1.)
    for (unsigned int i=0; i<points.size(); ++i)
      points[i] *= scale_grid;

  // connect the points to cells
  std::vector<CellData<dim> > cells(1);

   cells[0].vertices[0] = 0;
   cells[0].vertices[1] = 4;
   cells[0].vertices[2] = 1;
   cells[0].vertices[3] = 5;
   cells[0].vertices[4] = 2;
   cells[0].vertices[5] = 6;
   cells[0].vertices[6] = 3;
   cells[0].vertices[7] = 7;
   cells[0].material_id = 0;

  tria.create_triangulation (points, cells, SubCellData());
}


void unit_cell_mesh (Triangulation<2> &tria,
                  const double scale_grid = 1.)
{
  const unsigned int dim = 2;

  std::vector<Point<dim> > points (4);

  // 1. cube cell
  points[0] = Point<dim> (0, 0);
  points[1] = Point<dim> (0, 1);
  points[2] = Point<dim> (1,0);
  points[3] = Point<dim> (1,1);


  std::vector<CellData<dim> > cells(1);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 2;
  cells[0].vertices[2] = 1;
  cells[0].vertices[3] = 3;

  tria.create_triangulation (points, cells, SubCellData());
}

template <int dim, int degree_p, typename VectorType>
class MatrixFreeTest
{
public:
  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
  typedef double Number;
  const VectorizedArray<Number> *values_quad_new_impl = nullptr;

  MatrixFreeTest(const MatrixFree<dim,Number> &data_in, bool check_with_scalar=false):
	  	  	  data (data_in), n_q_points_1d(degree_p+2)
  {
	  deallog<<"Is MatrixFree using primitive element? = "<<data.is_primitive()<<std::endl;
	  deallog<<"n_array_elements on this machine = "<<n_array_elements<<std::endl;
	  deallog<<"n_q_points_1d = "<<n_q_points_1d<<std::endl;
	  deallog<<"No of (physical cells, macro cells) = ("<<data.n_physical_cells()<<", "<<data.n_macro_cells()<<")"<<std::endl;

  };

  void
  local_apply_vector (const MatrixFree<dim,Number> &data,
               VectorType          &dst,
               const VectorType    &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    typedef VectorizedArray<Number> vector_t;
    FEEvaluationGen<FE_RaviartThomas<dim>,degree_p+2,dim,degree_p+1,Number> velocity (data, 0);
    FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p,Number> pressure (data, 1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (false,true,false);
        //Debug
        values_quad_new_impl = velocity.begin_values();
      }

    //std::cout<<"Loop was internally run from "<<cell_range.first<<" to "<<cell_range.second<<std::endl;

}

  void vmult (VectorType &dst,
              const VectorType &src) const
  {
    if (data.is_primitive())
    {
    	//data.cell_loop (&MatrixFreeTest<dim,degree_p,VectorType>::local_apply_vector,
        //            this, dst, src);
    	std::cout<<"Primitive element not supported in this test"<<std::endl;
    }
    else
    {
    		data.cell_loop (&MatrixFreeTest<dim,degree_p,VectorType>::local_apply_vector,
                    this, dst, src);
    }

  };

private:
  const MatrixFree<dim,Number> &data;
  const int n_q_points_1d;
  const int n_array_elements = VectorizedArray<Number>::n_array_elements;
};


template <int dim, int fe_degree>
void test ()
{
	  Triangulation<dim>   triangulation;
	  unit_cell_mesh(triangulation);

	  std::cout<<"No of active cells from triangulation = "<<triangulation.n_active_cells()<<std::endl;

	  FE_RaviartThomas<dim> fe_u(fe_degree);
	  DoFHandler<dim>      dof_handler_u (triangulation);

	  MatrixFree<dim,double> mf_data(true);

	  ConstraintMatrix     constraints;


	  BlockVector<double> src_dofs, mf_res_vec, old_res_vec;

	  dof_handler_u.distribute_dofs (fe_u);

	  int n_u = dof_handler_u.n_dofs();

	  constraints.close ();

	  src_dofs.reinit (1);
	  src_dofs.block(0).reinit (n_u);
	  src_dofs.collect_sizes ();

	  mf_res_vec.reinit(src_dofs);
	  old_res_vec.reinit(src_dofs);


	  // fill some random values for actual dofss
	  for (unsigned int i=0; i<1; ++i)
	    for (unsigned int j=0; j<src_dofs.block(i).size(); ++j)
	      {
	        //const double val = -1. + 2.*random_value<double>();
	    	//debug, set all to 1
	    	const double val = 1.0;
	        src_dofs.block(i)(j) = val;
	      }

	  QGauss<dim>   quadrature_formula(fe_degree+2);
	  FullMatrix<double> N_matrix(n_u,quadrature_formula.size());
	  double values_quad_old_impl[n_u];


	  //Evaluate only on unit cell
	  //Non MF implementation
	  {
		//first test without C matrix, later use RT and use C matrix

		const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
				dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_u);

		//Evaluate basis functions on these point
		std::vector<Tensor<1,dim>> poly_values(n_u);
		std::vector<Tensor<2,dim>> unused2;
		std::vector<Tensor<3,dim>> unused3;
		std::vector<Tensor<4,dim>> unused4;
		std::vector<Tensor<5,dim>> unused5;

    	for (unsigned int q=0; q<quadrature_formula.size(); ++q)
    	{
   			Point<dim> p = quadrature_formula.get_points()[q];
			fe_poly->poly_space.compute(p,poly_values,unused2,unused3,unused4,unused5);
			for (int i=0;i<n_u;i++)
					N_matrix(i,q) = poly_values[i][0]; //Only for component 0 , debug FIXME
		}

    	//N_matrix.Tvmult(old_res_vec(0),src_dofs(0));
	  }


	  //MF implementation
	  {
	    std::vector<const DoFHandler<dim>*> dofs;
	    dofs.push_back(&dof_handler_u);
	    ConstraintMatrix dummy_constraints;
	    dummy_constraints.close();
	    std::vector<const ConstraintMatrix *> constraints;
	    constraints.push_back (&dummy_constraints);
	    QGauss<1> quad(fe_degree+2);
	    mf_data.reinit (dofs, constraints, quad,
	                    typename MatrixFree<dim>::AdditionalData
	                    (MatrixFree<dim>::AdditionalData::none));

		  //Convert moment dofs to nodal dofs for RT tensor product
		  //fe_u.inverse_node_matrix.vmult(src_vec.block(0),system_rhs.block(0));

		  typedef  BlockVector<double> VectorType;
		  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
		  //mf.vmult(mf_res_vec, src_dofs);

		  int n_eval_elements = n_u;
	  }

	  //mf.values_quad_new_impl is now available = mf_res_vec






#if 0 //open later
	  //Debug
	  // Verification
	  double error = 0., tol=1e-10;
	  bool result = true;

	  for (unsigned int i=0; i<2; ++i)
	    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
	    {
	      error += std::fabs (solution.block(i)(j)-dst_vec(i)(j));
	      if (error > tol)
	    		result = false;
	    }
	  double relative = solution.block(0).l1_norm();
	  deallog << "  Verification fe degree " << fe_degree  <<  ": "
	          << error/relative << std::endl << std::endl;


	  std::cout<<" Final result : "<<((result==true)?"pass ": "fail ")<<std::endl<<std::endl;
#endif
}


int main ()
{
  deallog.attach(logfile);

  //deallog << std::setprecision (3);

  {
    //deallog << std::endl << "Test with doubles" << std::endl << std::endl;
    //deallog.push("2d");
    test<2,1>();
    //test<2,2>();
    //test<2,3>();
    //test<2,4>();
    //deallog.pop();
    //deallog.push("3d");
    //test<3,1>();
    //test<3,2>();
    //deallog.pop();
  }
  deallog.detach();
}
