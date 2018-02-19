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



template <int dim>
void test_mesh (Triangulation<2> &tria,
                  const double scale_grid = 1.)
{
  std::vector<Point<dim> > points (4);

#if 0
  //Lets first confirm on unit cell
  // 1. cube cell
  points[0] = Point<dim> (0, 0);
  points[1] = Point<dim> (0, 1);
  points[2] = Point<dim> (1,0);
  points[3] = Point<dim> (1,1);
#endif

//#if 0
  // 2. rectangular cell
  points[0] = Point<dim> (1,0);;
  points[1] = Point<dim> (1,1);
  points[2] = Point<dim> (3., 0);
  points[3] = Point<dim> (3., 1);
//#endif

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
    //FEEvaluation<dim,degree_p,degree_p+2,1,Number> pressure (data, 1); //For scalar elements, use orig FEEvaluation


    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (false,true,false);
        //pressure.reinit (cell);
        //pressure.read_dof_values (src.block(1));
        //pressure.evaluate (true,false,false);

        //gradients_mf = velocity.begin_gradients(); //unit cell gradients

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
            //std::cout<<"gradient for first component are "<<grad_u[0][0][0]<<" and "<<grad_u[0][1][0]<<std::endl;
            //std::cout<<"gradient for second component are "<<grad_u[1][0][0]<<" and "<<grad_u[1][1][0]<<std::endl;

            //vector_t pres = pressure.get_value(q);
            //vector_t div = -trace(grad_u);
            //pressure.submit_value   (div, q);

            // subtract p * I
            //for (unsigned int d=0; d<dim; ++d)
            //  grad_u[d][d] -= pres;

            velocity.submit_gradient(grad_u, q);
            //std::cout<<"gradient for first component are "<<grad_u[0][0][0]<<" and "<<grad_u[0][1][0]<<std::endl;
            //std::cout<<"gradient for second component are "<<grad_u[1][0][0]<<" and "<<grad_u[1][1][0]<<std::endl;
          }
        //gradients_mf = velocity.begin_gradients(); //real cell gradients


        velocity.integrate (false,true);
        velocity.distribute_local_to_global (dst.block(0));
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
	///////////////////
	  Triangulation<dim>   triangulation;
	  test_mesh<dim>(triangulation);

	  std::cout<<"No of active cells from triangulation = "<<triangulation.n_active_cells()<<std::endl;

	  FE_RaviartThomas<dim> fe_u(fe_degree);
	  FESystem<dim>        fe (fe_u, 1);
	  DoFHandler<dim>      dof_handler_u (triangulation);
	  DoFHandler<dim>      dof_handler (triangulation);

	  MatrixFree<dim,double> mf_data(true);

	  ConstraintMatrix     constraints;

	  BlockSparsityPattern      sparsity_pattern;
	  BlockSparseMatrix<double> system_matrix;

	  BlockVector<double> solution;
	  BlockVector<double> system_rhs;
	  BlockVector<double> dst_vec;

	  dof_handler.distribute_dofs (fe);
	  dof_handler_u.distribute_dofs (fe_u);
	  DoFRenumbering::component_wise (dof_handler);

	  int n_u = dof_handler_u.n_dofs();

	  constraints.close ();


	  BlockDynamicSparsityPattern dsp(1, 1);

	  dsp.block(0, 0).reinit (n_u, n_u);
	  dsp.collect_sizes ();
	  DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
	  sparsity_pattern.copy_from(dsp);
	  system_matrix.reinit (sparsity_pattern);


	  //#5666: All components of velocity are treated as one block, pressure is treated as another block
	  //all components of velocity together
	  system_rhs.reinit (1);
	  system_rhs.block(0).reinit (n_u);
	  system_rhs.collect_sizes ();

	  solution.reinit (system_rhs);

	  dst_vec.reinit(system_rhs);


	  // this is from step-22

	    QGauss<dim>   quadrature_formula(fe_degree+2);

	    FEValues<dim> fe_values (fe, quadrature_formula,
	                             update_values    |
	                             update_JxW_values |
	                             update_gradients| update_inverse_jacobians);

	    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
	    const unsigned int   n_q_points      = quadrature_formula.size();

	    //Matrix of gradient vectors
	    std::vector<FullMatrix<double>> rt_test_phi_grad_u_matrix(dim,FullMatrix<double>(n_u,n_q_points));

	    int test_c = 1; //Test component number

	    Vector<double> test_system_rhs(n_u);
 	    Vector<double> test_grad_evaluation_results(n_q_points);

	    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

	    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	    const FEValuesExtractors::Vector velocities (0);

	    std::vector<Tensor<2,dim> > phi_grad_u (dofs_per_cell);

	    typename DoFHandler<dim>::active_cell_iterator
	    cell = dof_handler.begin_active(),
	    endc = dof_handler.end();
	    for (; cell!=endc; ++cell)
	      {
	    	std::cout<<"Cell fill "<<std::endl;
	        fe_values.reinit (cell);
	        local_matrix = 0;

#if 0
	        ////Debug
	        std::cout<<"inverse Jac is "<<std::endl;
	        for (int i=0; i<n_q_points; i++)
	        {
	        	auto j = fe_values.inverse_jacobian(i);
	        	std::cout<<j[0][0]<<", "<<j[0][1]<<", "<<j[1][0]<<", "<<j[1][1];
	        	std::cout<<std::endl;
	        }

	        std::cout<<std::endl;

	        return;
	        ////
#endif

	        for (unsigned int q=0; q<n_q_points; ++q)
	          {
	        	//Point<dim> p = quadrature_formula.get_points()[q];
	        	for (unsigned int k=0; k<dofs_per_cell; ++k)
	              {
	                phi_grad_u[k] = fe_values[velocities].gradient(k, q);
            		//div_phi_u[k]  = fe_values[velocities].divergence (k, q);
	                //phi_p[k]      = fe_values[pressure].value (k, q);

            		for (int d=0; d<dim; d++)
            		{
            		   		rt_test_phi_grad_u_matrix[d](k,q) = phi_grad_u[k][test_c][d];
            		}

	              }

//#if 0
	            for (unsigned int i=0; i<dofs_per_cell; ++i)
	              {
	                for (unsigned int j=0; j<=i; ++j)
	                  {
	                    //local_matrix(i,j) += (scalar_product(phi_grad_u[i], phi_grad_u[j])
	                    //                      - div_phi_u[i] * phi_p[j]
	                    //                      - phi_p[i] * div_phi_u[j])
	                    //                     * fe_values.JxW(q);
	                    local_matrix(i,j) += scalar_product(phi_grad_u[i], phi_grad_u[j])
	                                         * fe_values.JxW(q);

	                    //local_matrix(i,j) += scalar_product(phi_grad_u[i], phi_grad_u[j])
	                    //	                                         ;

	                  }
	              }
//#endif
	          }

	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          for (unsigned int j=i+1; j<dofs_per_cell; ++j)
	            local_matrix(i,j) = local_matrix(j,i);

	        #if 0
	        	  std::cout<<"Local matrix from dealii is"<<std::endl;
	              for (unsigned int i=0; i<dofs_per_cell; ++i)
	                {
	                  for (unsigned int j=0; j<dofs_per_cell; ++j)
	                    {
	        				double val = local_matrix(i,j);
	        				if (std::fabs(val) < 10e-6) val = 0;
	        				std::cout <<std::setw(5)<<val;
	                    }
	                  std::cout<<std::endl;
	                }
	              std::cout<<std::endl;

	              //return;
	        #endif

	        cell->get_dof_indices (local_dof_indices);
	        constraints.distribute_local_to_global (local_matrix,
	                                                local_dof_indices,
	                                                system_matrix);
	      }

#if 0
	    std::cout<<"test component = "<<test_c<<std::endl;
		for (int d=0; d<dim; d++)
		{
			std::cout<<"direction = "<<d<<"  ========================"<<std::endl;
			std::cout<<"D matrix from FEValues is"<<std::endl;
			for (unsigned int i=0; i<n_u; i++)
			{
				for (unsigned int q=0; q<n_q_points; q++)
				{
					double val = rt_test_phi_grad_u_matrix[d](i,q);
					if (std::fabs(val) < 10e-6) val = 0;
					std::cout <<std::setw(15)<<val;
				}
				std::cout<<std::endl;
			}
			std::cout<<std::endl<<std::endl;
		}
#endif



	  // first system_rhs with random numbers
	    float t = 1.0f;
	  for (unsigned int i=0; i<1; ++i)
	    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
	      {
	    	//all zeros
	        const double val = -1. + 2.*random_value<double>();
	        system_rhs.block(i)(j) = t++; //val;
	        if (i==0)
	        	test_system_rhs[j] = system_rhs.block(i)(j);
	      }

//#if 0
	  std::cout<<"Input to  dealii is "<<std::endl;
	  for (unsigned int i=0; i<1; ++i)
	  {
		  std::cout<<"Block = "<<i<<std::endl;
	    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
	      {
	    	std::cout<<std::setw(10)<<system_rhs.block(i)(j);
	      }
	  }
	  std::cout<<std::endl;
//#endif

#if 0
	  std::cout<<"System matrix from dealii is"<<std::endl;
	              for (unsigned int i=0; i<dofs_per_cell; ++i)
	                {
	                  for (unsigned int j=0; j<dofs_per_cell; ++j)
	                    {
	        				double val = system_matrix(i,j);
	        				if (std::fabs(val) < 10e-6) val = 0;
	        				std::cout <<std::setw(5)<<val;
	                    }
	                  std::cout<<std::endl;
	                }
	              std::cout<<std::endl;

#endif

//#if 0
	  system_matrix.vmult (solution, system_rhs);

	  std::cout<<"Solution vector using dealii is "<<std::endl;
	  for (unsigned int i=0; i<1; ++i)
	  {
		  std::cout<<"Block = "<<i<<std::endl;
	    for (unsigned int j=0; j<solution.block(i).size(); ++j)
	      {
	    	std::cout<<std::setw(10)<<solution.block(i)(j);
	      }
	  }
	  std::cout<<std::endl;
//#endif



#if 0
	  std::cout<<"Results from dealii RT gradient eval for component = "<<test_c<<" are"<<std::endl;
	  for (int d=0; d<dim; d++)
	  {
		  rt_test_phi_grad_u_matrix[d].Tvmult(test_grad_evaluation_results,test_system_rhs);
		  std::cout<<"==dim = "<<d<<"    ";
		  for (int q=0; q<n_q_points; q++)
		  {
			  std::cout<<std::setw(10)<<test_grad_evaluation_results[q];
		  }
		  std::cout<<std::endl;
	  }
#endif

	  // setup matrix-free structure
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
	  }


	  typedef  BlockVector<double> VectorType;
	  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
	  mf.vmult(dst_vec, system_rhs);

//#if 0
	  std::cout<<"Solution vector using MF is "<<std::endl;
	  for (unsigned int i=0; i<1; ++i)
	  {
		  std::cout<<"Block = "<<i<<std::endl;
	    for (unsigned int j=0; j<dst_vec.block(i).size(); ++j)
	      {
	    	std::cout<<std::setw(10)<<dst_vec.block(i)(j);
	      }
	  }
	  std::cout<<std::endl;
//#endif


#if 0
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
#endif

#if 0
	  std::cout<<"W is "<<std::endl;
	  for (int i=0; i<n_q_points; i++)
	  {
		  std::cout<<std::setw(10)<<quadrature_formula.weight(i);
	  }
	  std::cout<<std::endl;

	  std::cout<<"JxW is "<<std::endl;
	  for (int i=0; i<n_q_points; i++)
	  {
		  std::cout<<std::setw(10)<<fe_values.JxW(i);
	  }

	  std::cout<<std::endl;
#endif

}

#if 0
#define __UT__

template <int dim, int fe_degree, int n_q_points_1d, typename Number,
int direction, bool dof_to_quad, bool add, int inter_dim>
inline
void
apply_anisotropic (const Number *shape_data,
           const Number in [],
           Number       out [])
  {
    AssertIndexRange (direction, dim);
    const int mm     = dof_to_quad ? (fe_degree+1) : n_q_points_1d,
              nn     = dof_to_quad ? n_q_points_1d : (fe_degree+1);


    //const int n_blocks1 = (dim > 1 ? (direction > 0 ? nn : mm) : 1);
    const int n_blocks1 = (dim > 1 ? inter_dim : 1);
    const int n_blocks2 = (dim > 2 ? (direction > 1 ? nn : mm) : 1); //FIXME for dim=3
    //const int stride    = Utilities::fixed_int_power<nn,direction>::value;
    const int stride    = Utilities::fixed_int_power<inter_dim,direction>::value; //FIXME for dim=3

#ifdef __UT__
    printf("\n inside apply_anisotropic, (mm,nn, inter_dim) = (%d,%d,%d)", mm,nn,inter_dim);
    printf("\n b_blocks(1,2) = (%d,%d), Stride = %d", n_blocks1, n_blocks2,stride);
#endif

    for (int i2=0; i2<n_blocks2; ++i2)
      {
        for (int i1=0; i1<n_blocks1; ++i1)
          {
            for (int col=0; col<nn; ++col)
              {
                Number val0;
                if (dof_to_quad == true)
                  val0 = shape_data[col];
                else
                  val0 = shape_data[col*n_q_points_1d];
                Number res0 = val0 * in[0];
#ifdef __UT__
                printf("\n val0 = %f in[0] = %f res0 = %f", val0,in[0], res0);
#endif
                for (int ind=1; ind<mm; ++ind)
                  {
                    if (dof_to_quad == true)
                      val0 = shape_data[ind*n_q_points_1d+col];
                    else
                      val0 = shape_data[col*n_q_points_1d+ind];
                    res0 += val0 * in[stride*ind];
#ifdef __UT__
                    printf("\n val0 = %f, Reading from index = %d a value in[index] = %f and calculating res0 = %f",val0,stride*ind,in[stride*ind],res0);
#endif
                  }
                if (add == false)
                  out[stride*col]  = res0;
                else
                  out[stride*col] += res0;
#ifdef __UT__
                printf("\n Storing res0 = %f at index = %d", res0,(stride*col));
#endif
              }

            // increment: in regular case, just go to the next point in
            // x-direction. If we are at the end of one chunk in x-dir, need
            // to jump over to the next layer in z-direction
            switch (direction)
              {
              case 0:
                in += mm;
                out += nn;
                break;
              case 1:
              case 2:
                ++in;
                ++out;
                break;
              default:
                Assert (false, ExcNotImplemented());
              }
          }
        if (direction == 1) //FIXME for dim=3
          {
            in += nn*(mm-1);
            out += nn*(nn-1);
          }
      }
  }
#endif

int main ()
{
#if 0
	constexpr int fe_degree = 2;
	constexpr int n_q_points_1d = fe_degree+1;
	constexpr int dir = 1;
	constexpr bool dof_to_quad = true;


	double out[15];
	double temp[20];
	double A[6] = {1,2,3,4,5,6};
	double B[20] = {7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
#endif

#if 0 //tr(A)*U*B is ok
	double in[8] = {20,21,22,23,24,25,26,27};
	apply_anisotropic<2, 1, 3, double, 0, dof_to_quad, false, 4>(A,in,temp);
	apply_anisotropic<2, 3, 5, double, 1, dof_to_quad, false, 3>(B,temp,out);
	std::cout<<"product is  "<<std::endl;
	for (unsigned int i=0;i<15; ++i)
		std::cout<<std::setw(10)<<out[i];
	std::cout<<std::endl;
#endif

#if 0 //A*U*tr(B) is ok
	double in[15] = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34};
	apply_anisotropic<2, 1, 3, double, 0, false, false, 5>(A,in,temp);
	apply_anisotropic<2, 3, 5, double, 1, false, false, 2>(B,temp,out);
	std::cout<<"product is  "<<std::endl;
	for (unsigned int i=0;i<8; ++i)
		std::cout<<std::setw(10)<<out[i];
	std::cout<<std::endl;
#endif

#if 0 //A*U*B is ok
	double in[12] = {20,21,22,23,24,25,26,27,28,29,30,31};
	apply_anisotropic<2, 1, 3, double, 0, false, false, 4>(A,in,temp);
	apply_anisotropic<2, 3, 5, double, 1, true, false, 2>(B,temp,out);
	for (unsigned int i=0;i<10; ++i)
		std::cout<<std::setw(10)<<out[i];
	std::cout<<std::endl;
#endif

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
