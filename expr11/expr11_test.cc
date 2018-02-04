//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
//			 - compare values, gradients and hessians
// This is an experiment to validate my implementation by using existing implementation of dealii RT elements
//so that functional bugs are removed
//The code base is adapted from matrix_vector_stokes.cc and changed to RT
// Step by step tests: Step 1
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

#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/evaluation_kernels.h>


#include "create_mesh.h"

#include <iostream>
#include <complex>
#include <vector>

using namespace dealii;

//More basic tests
//Comparison of general output of FEValues (MatrixFree)
// and FEEvaluationGen+MatrixFree(for RT)
//sibling test: expr_4 - test2.cc and matrix_vector_stokes.cc


void test_mesh (Triangulation<2> &tria,
                  const double scale_grid = 1.)
{
  const unsigned int dim = 2;

#if 0
  std::vector<Point<dim> > points (12);

  // build the mesh layer by layer from points

  // 1. cube cell
  points[0] = Point<dim> (0, 0);
  points[1] = Point<dim> (0, 1);
  points[2] = Point<dim> (1,0);
  points[3] = Point<dim> (1,1);

  // 2. rectangular cell
  points[4] = Point<dim> (3., 0);
  points[5] = Point<dim> (3., 1);

  // 3. parallelogram cell
  points[6] = Point<dim> (5., 1.);
  points[7] = Point<dim> (5., 2.);

  // almost square cell (but trapezoidal by
  // 1e-8)
  points[8] = Point<dim> (6., 1.);
  points[9] = Point<dim> (6., 2.+1e-8);

  // apparently trapezoidal cell
  points[10] = Point<dim> (7., 1.4);
  points[11] = Point<dim> (7.5, numbers::PI);

  if (scale_grid != 1.)
    for (unsigned int i=0; i<points.size(); ++i)
      points[i] *= scale_grid;


  // connect the points to cells
  std::vector<CellData<dim> > cells(5);
  for (unsigned int i=0; i<5; ++i)
    {
      cells[i].vertices[0] = 0+2*i;
      cells[i].vertices[1] = 2+2*i;
      cells[i].vertices[2] = 1+2*i;
      cells[i].vertices[3] = 3+2*i;
      cells[i].material_id = 0;
    }
#endif

  std::vector<Point<dim> > points (4);

#if 0
  // 3. parallelogram cell
  points[0] = Point<dim> (3., 0);
  points[1] = Point<dim> (3., 1);
  points[2] = Point<dim> (5., 1.);
  points[3] = Point<dim> (5., 2.);
#endif

  //Lets first confirm on unit cell
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

VectorizedArray<double> *gradients_mf;

template <int dim, int degree_p, typename VectorType>
class MatrixFreeTest
{
public:
  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
  typedef double Number;

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
    FEEvaluationGen<FE_RaviartThomas<dim>,degree_p+2,dim,degree_p,Number> velocity (data, 0);
    //FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p,Number> pressure (data, 1);
    FEEvaluation<dim,degree_p,degree_p+2,1,Number> pressure (data, 1); //For scalar elements, use orig FEEvaluation


    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (false,true,false);
        //pressure.reinit (cell);
        //pressure.read_dof_values (src.block(1));
        //pressure.evaluate (true,false,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            //Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
            //vector_t pres = pressure.get_value(q);
            //vector_t div = -trace(grad_u);
            //pressure.submit_value   (div, q);

            // subtract p * I
            //for (unsigned int d=0; d<dim; ++d)
            //  grad_u[d][d] -= pres;

            //velocity.submit_gradient(grad_u, q);
          }

        gradients_mf = velocity.begin_gradients();

        //velocity.integrate (false,true);
        //velocity.distribute_local_to_global (dst.block(0));
        //pressure.integrate (true,false);
        //pressure.distribute_local_to_global (dst.block(1));
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


template <int dim, int fe_degree, int n_components=dim>
void test ()
{
#if 0
	constexpr int fe_deg_x2 = 2;
	constexpr int fe_deg_y2 = 1;
	constexpr int n_q_points_1d = fe_degree+2;
	using VecArr = VectorizedArray<double>;
	VecArr N0[9] = {{0.6872,0.6872}, {0,0}, {-0.0872,-0.0872},
				 {0.3999,0.3999}, {1,1}, {0.3999,0.3999},
				 {-0.0872,-0.0872},{0,0}, {0.6872,0.6872}
				 };
	VecArr N1[6] = {{0.8872,0.8872}, {0.5,0.5}, {0.11270,0.11270},
				 {0.11270,0.11270}, {0.5,0.5}, {0.8872,0.8872}
				 };
	VecArr D0[9] = {
			{-2.5491,-2.5491}, {-1,-1}, {0.5491,0.5491},
			{3.0983,3.0983}, {0,0}, {-3.098,-3.098},
			{-0.5491,-0.5491}, {1,1}, {2.5491,2.5491}
	};

	VecArr D1[6] = {
			{-1,-1}, {-1,-1}, {-1,-1},
			{1,1}, {1,1}, {1,1}
	};

	VecArr values_dofs[2][6] = {
				{{1,1},{2,2},{3,3},{4,4},{5,5},{6,6}},
				{{7,7},{8,8},{9,9},{10,10},{11,11},{12,12}}
	};
	VecArr temp1[100];
	VecArr gradients_quad[2][2][9];

	//c=0
	// grad x
	dealii::internal::apply_anisotropic<dim,fe_deg_x2,n_q_points_1d,VecArr,0,true,false,fe_deg_y2>(D0,values_dofs[0], temp1);
	dealii::internal::apply_anisotropic<dim,fe_deg_y2,n_q_points_1d,VecArr,1,true,false,fe_deg_x2>(N1,temp1,gradients_quad[0][0]);

	dealii::internal::apply_anisotropic<dim, fe_deg_x2, n_q_points_1d,VecArr,0,true,false,fe_deg_y2>(N0,values_dofs[0], temp1);
	dealii::internal::apply_anisotropic<dim,fe_deg_y2,n_q_points_1d,VecArr,1,true,false,fe_deg_x2>(D1,temp1,gradients_quad[0][1]);

	//c=1
	constexpr int fe_deg_x2_1 = 1;
	constexpr int fe_deg_y2_1 = 2;
	dealii::internal::apply_anisotropic<dim,fe_deg_x2_1,n_q_points_1d,VecArr,0,true,false,fe_deg_y2_1>(D1,values_dofs[1], temp1);
	dealii::internal::apply_anisotropic<dim,fe_deg_y2_1,n_q_points_1d,VecArr,1,true,false,fe_deg_x2_1>(N0,temp1,gradients_quad[1][0]);

	dealii::internal::apply_anisotropic<dim, fe_deg_x2_1, n_q_points_1d,VecArr,0,true,false,fe_deg_y2_1>(N1,values_dofs[1], temp1);
	dealii::internal::apply_anisotropic<dim,fe_deg_y2_1,n_q_points_1d,VecArr,1,true,false,fe_deg_x2_1>(D0,temp1,gradients_quad[1][1]);


	  std::cout<<"Results from test gradient eval are"<<std::endl;

	  for (int c=0; c<n_components; c++)
	  {
		  std::cout<<"=====Component = "<<c<<std::endl;
		  for (int d=0; d<dim; d++)
		  {
			  std::cout<<"==dim = "<<d<<"    ";
			  for (int q=0; q<9; q++)
			  {
				  std::cout<<std::setw(10)<<gradients_quad[c][d][q][0];
			  }
			  std::cout<<std::endl;
		  }
		  std::cout<<std::endl;
	  }
	  std::cout<<std::endl;


	return;
#endif
	///////////////////
	  Triangulation<dim>   triangulation;
	  test_mesh(triangulation);
#if 0
	  create_mesh (triangulation);
	  if (fe_degree == 1)
	    triangulation.refine_global (4-dim);
	  else
	    triangulation.refine_global (3-dim);
#endif

	  std::cout<<"No of active cells from triangulation = "<<triangulation.n_active_cells()<<std::endl;

	  FE_RaviartThomas<dim> fe_u(fe_degree);
	  FE_Q<dim>            fe_p (fe_degree);
	  FESystem<dim>        fe (fe_u, 1, fe_p, 1);
	  DoFHandler<dim>      dof_handler_u (triangulation);
	  DoFHandler<dim>      dof_handler_p (triangulation);
	  DoFHandler<dim>      dof_handler (triangulation);

	  MatrixFree<dim,double> mf_data(true);

	  ConstraintMatrix     constraints;

	  BlockSparsityPattern      sparsity_pattern;
	  BlockSparseMatrix<double> system_matrix;

	  BlockVector<double> solution;
	  BlockVector<double> system_rhs;
	  BlockVector<double> src_vec, dst_vec;

	  dof_handler.distribute_dofs (fe);
	  dof_handler_u.distribute_dofs (fe_u);
	  dof_handler_p.distribute_dofs (fe_p);
	  DoFRenumbering::component_wise (dof_handler);

	  int n_u = dof_handler_u.n_dofs();
	  int n_p = dof_handler_p.n_dofs();

	  constraints.close ();


	  BlockDynamicSparsityPattern dsp(2, 2);
	  dsp.block(0, 0).reinit (n_u, n_u);
	  dsp.block(1, 0).reinit (n_p, n_u);
	  dsp.block(0, 1).reinit (n_u, n_p);
	  dsp.block(1, 1).reinit (n_p, n_p);
	  dsp.collect_sizes ();
	  DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
	  sparsity_pattern.copy_from(dsp);
	  system_matrix.reinit (sparsity_pattern);


	  //#5666: All components of velocity are treated as one block, pressure is treated as another block
	  //all components of velocity together
	  system_rhs.reinit (2);
	  system_rhs.block(0).reinit (n_u);
	  system_rhs.block(1).reinit (n_p);
	  system_rhs.collect_sizes ();

	  solution.reinit (system_rhs);

	  src_vec.reinit(system_rhs);
	  dst_vec.reinit(system_rhs);

	  // this is from step-22
	  //{

	    QGauss<dim>   quadrature_formula(fe_degree+2);

	    FEValues<dim> fe_values (fe, quadrature_formula,
	                             update_values    |
	                             update_JxW_values |
	                             update_gradients| update_inverse_jacobians);

	    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
	    const unsigned int   n_q_points      = quadrature_formula.size();

	    //Matrix of gradient vectors
	    std::vector<FullMatrix<double>> rt_test_phi_grad_u_matrix(dim,FullMatrix<double>(n_u,n_q_points));
	    std::vector<FullMatrix<double>> poly_test_phi_grad_u_matrix(dim,FullMatrix<double>(n_u,n_q_points));
	    int test_c = 0; //Test component number
	    Vector<double> test_system_rhs(n_u);
 	    Vector<double> test_grad_evaluation_results(n_q_points);
 	    const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
				dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_u);
		//Evaluate basis functions on these point
		std::vector<Tensor<1,dim>> unused1;
		std::vector<Tensor<2,dim>> poly_grads(n_u);
		std::vector<Tensor<3,dim>> unused3;
		std::vector<Tensor<4,dim>> unused4;
		std::vector<Tensor<5,dim>> unused5;

	    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

	    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	    const FEValuesExtractors::Vector velocities (0);
	    const FEValuesExtractors::Scalar pressure (dim);

	    std::vector<Tensor<2,dim> > phi_grad_u (dofs_per_cell);
	    std::vector<double>         div_phi_u  (dofs_per_cell);
	    std::vector<double>         phi_p      (dofs_per_cell);

	    typename DoFHandler<dim>::active_cell_iterator
	    cell = dof_handler.begin_active(),
	    endc = dof_handler.end();
	    for (; cell!=endc; ++cell)
	      {
	        fe_values.reinit (cell);
	        local_matrix = 0;

	        for (unsigned int q=0; q<n_q_points; ++q)
	          {
	        	Point<dim> p = quadrature_formula.get_points()[q];
	        	fe_poly->poly_space.compute(p,unused1,poly_grads,unused3,unused4,unused5);
	            for (unsigned int k=0; k<n_u/*dofs_per_cell*/; ++k)
	              {
	                phi_grad_u[k] = fe_values[velocities].gradient(k, q);
	                //div_phi_u[k]  = fe_values[velocities].divergence (k, q);
	                //phi_p[k]      = fe_values[pressure].value (k, q);

	                for (int d=0; d<dim; d++)
	                {
	                	//rt_test_phi_grad_u_matrix[d](k,q) = phi_grad_u[k][test_c][d];
	                	/// The below line gives result in correct order. but above line does not
	                	rt_test_phi_grad_u_matrix[d](k,q) = fe_u.shape_grad_component(k,p,test_c)[d];
	                	poly_test_phi_grad_u_matrix[d](k,q) = poly_grads[k][test_c][d];
	                }
	              }

#if 0
	            for (unsigned int i=0; i<dofs_per_cell; ++i)
	              {
	                for (unsigned int j=0; j<=i; ++j)
	                  {
	                    local_matrix(i,j) += (scalar_product(phi_grad_u[i], phi_grad_u[j])
	                                          - div_phi_u[i] * phi_p[j]
	                                          - phi_p[i] * div_phi_u[j])
	                                         * fe_values.JxW(q);


	                  }
	              }
#endif
	          }
#if 0
	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          for (unsigned int j=i+1; j<dofs_per_cell; ++j)
	            local_matrix(i,j) = local_matrix(j,i);

	        cell->get_dof_indices (local_dof_indices);
	        constraints.distribute_local_to_global (local_matrix,
	                                                local_dof_indices,
	                                                system_matrix);
#endif
	      }
	  //}

	  // first system_rhs with random numbers
	    float t = 1.0f;
	  for (unsigned int i=0; i<2; ++i)
	    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
	      {
	    	//all zeros
	        const double val = -1. + 2.*random_value<double>();
	        system_rhs.block(i)(j) = t++; //val;
	        if (i==0)
	        	test_system_rhs[j] = val;
	      }

#if 0
	  system_matrix.vmult (solution, system_rhs);
#endif

	  //Extend it to all dimensions later after some tests
	  //test_phi_grad_u_matrix[0].vmult(test_grad_evaluation_results,test_system_rhs);

	  // setup matrix-free structure
	  {
	    std::vector<const DoFHandler<dim>*> dofs;
	    dofs.push_back(&dof_handler_u);
	    dofs.push_back(&dof_handler_p);
	    ConstraintMatrix dummy_constraints;
	    dummy_constraints.close();
	    std::vector<const ConstraintMatrix *> constraints;
	    constraints.push_back (&dummy_constraints);
	    constraints.push_back (&dummy_constraints);
	    QGauss<1> quad(fe_degree+2);
	    mf_data.reinit (dofs, constraints, quad,
	                    typename MatrixFree<dim>::AdditionalData
	                    (MatrixFree<dim>::AdditionalData::none));
	  }


	  //Convert moment dofs to nodal dofs for RT tensor product
	  //fe_u.inverse_node_matrix.vmult(src_vec.block(0),system_rhs.block(0));
	  src_vec = system_rhs;
	  //src_vec.block(0)[0] = 1.0;

	  typedef  BlockVector<double> VectorType;
	  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
	  mf.vmult(dst_vec, src_vec);


	  std::cout<<"Input src_vector is "<<std::endl;
	  for (int i=0; i<n_u; i++)
	  {
		  std::cout<<std::setw(10)<<src_vec.block(0)[i];
	  }

	  std::cout<<std::endl;

	  std::cout<<"Results from MF gradient eval are"<<std::endl;

	  for (int c=0; c<n_components; c++)
	  {
		  std::cout<<"=====Component = "<<c<<std::endl;
		  for (int d=0; d<dim; d++)
		  {
			  std::cout<<"==dim = "<<d<<"    ";
			  for (int q=0; q<n_q_points; q++)
			  {
				  std::cout<<std::setw(10)<<gradients_mf[c*(dim*n_q_points)+d*n_q_points+q][0];
			  }
			  std::cout<<std::endl;
		  }
		  std::cout<<std::endl;
	  }
	  std::cout<<std::endl;


#if 0 //open later
	  //Debug
	  std::ofstream logfile("output");
	  deallog.attach(logfile);

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
	  deallog.detach();

	  std::cout<<" Final result : "<<((result==true)?"pass ": "fail ")<<std::endl<<std::endl;
#endif
		std::cout<<"Component no = "<<test_c<<"  ========================"<<std::endl;

#if 0 //This is ok, so close now
		std::cout<<"Matrix from poly RT d = 0 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<poly_test_phi_grad_u_matrix[0](i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;

		std::cout<<"Matrix from poly RT d = 1 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<poly_test_phi_grad_u_matrix[1](i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;
#endif
#if 0
		act = C.phihat
		C = X^{-T}
		i have X^{-1}

		act = Tra(inv node).phihat
		inv node.Tmmult(act,phihat)
#endif

#if 0 //This match but row is pushed down , to debug next : I think this is due to component wise ordering
		//..dealii somehow orders, should not matter for me.
		FullMatrix<double> temp(n_u,n_q_points);
		fe_u.inverse_node_matrix.Tmmult(temp,poly_test_phi_grad_u_matrix[0]);

		std::cout<<"Matrix from Processed poly RT d = 0 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<temp(i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;


		fe_u.inverse_node_matrix.Tmmult(temp,poly_test_phi_grad_u_matrix[1]);

		std::cout<<"Matrix from Processed poly RT d = 1 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<temp(i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;



		std::cout<<"Matrix from RT d = 0 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<rt_test_phi_grad_u_matrix[0](i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;

		std::cout<<"Matrix from RT d = 1 is"<<std::endl;
		for (unsigned int i=0; i<n_u; i++)
		{
			for (unsigned int q=0; q<n_q_points; q++)
			{
				std::cout <<std::setw(12)<<rt_test_phi_grad_u_matrix[1](i,q);
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl<<std::endl;
#endif
}


int main ()
{
  //deallog.attach(logfile);

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
}
