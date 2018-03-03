/*
 * this case tests the correctness of the implementation of matrix free
 * operators used with Raviart Thomas Elements. The operators chosen are
 * from Darcy porous media flow problem with unit permeability tensor.
 * The results of matrix-vector products are compared with the result of
 * deal.II sparse matrix. No hanging nodes and no other constraints.
 */
//This test should be added to dealii tests

#include "tests.h"

std::ofstream logfile("output");

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>


#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>


  using namespace dealii;

  template <int dim, int degree_p, typename VectorType>
  class MF_MixedLaplace
  {
  public:
    typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
    typedef double Number;

    static constexpr unsigned int  n_q_points_1d = degree_p+2;

    MF_MixedLaplace(const MatrixFree<dim,Number> &data_in):
    	data (data_in)
    {};

    //We use unit permeability tensor in these tests

    void
    local_apply_opA (const MatrixFree<dim,Number> &data,
                 VectorType          &dst,
                 const VectorType    &src,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
      typedef VectorizedArray<Number> vector_t;
      FEEvaluationGen<FE_RaviartThomas<dim>,n_q_points_1d,dim,degree_p,Number> velocity (data, 0);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          velocity.reinit (cell);
          velocity.read_dof_values (src.block(0));
          velocity.evaluate (true,false,false); //Evaluate values

          for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
        	  Tensor<1,dim,vector_t> u = velocity.get_value(q);
              velocity.submit_value (u, q);
          }

          velocity.integrate (true,false);
          velocity.distribute_local_to_global (dst.block(0));
        }
    }

    void
    local_apply_opB_tr (const MatrixFree<dim,Number> &data,
                 VectorType          &dst,
                 const VectorType    &src,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
      typedef VectorizedArray<Number> vector_t;
      FEEvaluationGen<FE_RaviartThomas<dim>,n_q_points_1d,dim,degree_p,Number> velocity (data, 0);
      FEEvaluation<dim,degree_p,n_q_points_1d,1,Number> pressure (data, 1); //For scalar elements, use orig FEEvaluation

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          velocity.reinit (cell);
          velocity.read_dof_values (src.block(0));
          velocity.evaluate (true,true,false); //Evaluate values and gradients

          pressure.reinit (cell);
          pressure.read_dof_values (src.block(1));
          pressure.evaluate (true,false,false);

          for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
        	  Tensor<1,dim,vector_t> u = velocity.get_value(q);
              velocity.submit_value (u, q);

              Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
        	  vector_t pres = pressure.get_value(q);

        	  grad_u.clear();
              for (unsigned int d=0; d<dim; ++d)
                grad_u[d][d] = -pres;

              velocity.submit_gradient(grad_u, q);
          }

          velocity.integrate (true,true);
          velocity.distribute_local_to_global (dst.block(0));
        }
  }

    void
    local_apply_opB (const MatrixFree<dim,Number> &data,
                 VectorType          &dst,
                 const VectorType    &src,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
      typedef VectorizedArray<Number> vector_t;
      FEEvaluationGen<FE_RaviartThomas<dim>,n_q_points_1d,dim,degree_p,Number> velocity (data, 0);
      FEEvaluation<dim,degree_p,n_q_points_1d,1,Number> pressure (data, 1); //For scalar elements, use orig FEEvaluation

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          velocity.reinit (cell);
          velocity.read_dof_values (src.block(0));
          velocity.evaluate (false,true,false); //Evaluate gradients

          pressure.reinit (cell);
          pressure.read_dof_values (src.block(1));
          pressure.evaluate (true,false,false);

          for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
              Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
        	  vector_t pres = pressure.get_value(q);
        	  vector_t div = -trace(grad_u);
        	  pressure.submit_value (div, q);
          }

          pressure.integrate (true,false);
          pressure.distribute_local_to_global (dst.block(1));
        }
  }

    void Avmult (VectorType &dst,
                const VectorType &src) const
    {
    	dst.block(0) = 0;
    	dst.block(1) = 0;
    	data.cell_loop (&MF_MixedLaplace<dim,degree_p,VectorType>::local_apply_opA,
                    this, dst, src);
    };

    void Bvmult (VectorType &dst,
                const VectorType &src) const
    {
    	dst.block(0) = 0;
    	dst.block(1) = 0;
    	data.cell_loop (&MF_MixedLaplace<dim,degree_p,VectorType>::local_apply_opB,
                    this, dst, src);

    };

    void B_tra_vmult (VectorType &dst,
                const VectorType &src) const
    {
    	dst.block(0) = 0;
    	dst.block(1) = 0;
    	data.cell_loop (&MF_MixedLaplace<dim,degree_p,VectorType>::local_apply_opB_tr,
                    this, dst, src);

    };

  private:
    const MatrixFree<dim,Number> &data;
  };


  template <int dim, int degree>
  class TestMixedLaplace
  {
  public:
    TestMixedLaplace ();
    void run ();

  private:
    enum op_id {OP_A, OP_B, OP_B_tra}; //operator IDs

    void make_grid_and_dofs ();
    void assemble_sparse_system ();
    void test_mf_operator (enum op_id, Timer &time,
    				BlockSparseMatrix<double> &tmp_system_matrix);
    void compute_errors (int block_no) const;

    const unsigned int   n_q_points_1d;

    FESystem<dim>        fe;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;
    BlockVector<double>       system_rhs;

    //data structures For MF
    MatrixFree<dim,double> 	mf_data;

    FE_RaviartThomas<dim> fe_u;
    FE_DGQ<dim>           fe_p;

    DoFHandler<dim>      dof_handler_u;
    DoFHandler<dim>      dof_handler_p;

    BlockVector<double> 	  mf_solution;
    BlockVector<double>       mf_rhs;
  };



  template <int dim, int degree>
  TestMixedLaplace<dim, degree>::TestMixedLaplace ()
    :
    n_q_points_1d(degree+2),
    fe (FE_RaviartThomas<dim>(degree),1,FE_DGQ<dim>(degree),1),
    dof_handler (triangulation),
    //For MF
    fe_u(FE_RaviartThomas<dim>(degree)),
    fe_p(FE_DGQ<dim>(degree)),
    dof_handler_u (triangulation),
    dof_handler_p (triangulation)
  {
	  std::cout << "dim = "<<dim
			    <<std::endl
			    <<"RT_degree = "<<degree
			    <<std::endl
			    <<"FE_DGQ_degree = "<<degree;
  }


  template <int dim, int degree>
  void TestMixedLaplace<dim,degree>::make_grid_and_dofs ()
  {
    //We can only work on cartesian mesh cells in MF version - so let it be as-it-is
	GridGenerator::hyper_cube (triangulation, -1, 1);
	//First, lets test on one cell only
//#if 0
    triangulation.refine_global (3);
//#endif

    dof_handler.distribute_dofs (fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim];

    std::cout << "Number of active cells =  "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells = "
              << triangulation.n_cells()
              << std::endl
              << "Number of Velocity degrees of freedom = "
              << n_u
              << std::endl
              << "Number of Pressure degrees of freedom = "
              << n_p
              << std::endl
              << "Total Number of degrees of freedom = "
              << dof_handler.n_dofs()
              << std::endl;

    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit (n_u, n_u);
    dsp.block(1, 0).reinit (n_p, n_u);
    dsp.block(0, 1).reinit (n_u, n_p);
    dsp.block(1, 1).reinit (n_p, n_p);
    dsp.collect_sizes ();
    DoFTools::make_sparsity_pattern (dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();


    // setup matrix-free structure
    {
        dof_handler_u.distribute_dofs (fe_u);
        dof_handler_p.distribute_dofs (fe_p);

        mf_solution.reinit(solution);
        mf_rhs.reinit(solution);

    	std::vector<const DoFHandler<dim>*> dofs;
        dofs.push_back(&dof_handler_u);
        dofs.push_back(&dof_handler_p);
        ConstraintMatrix dummy_constraints;
        dummy_constraints.close();
        std::vector<const ConstraintMatrix *> constraints;
        constraints.push_back (&dummy_constraints);
        constraints.push_back (&dummy_constraints);
        QGauss<1> quad(n_q_points_1d);

        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.mapping_update_flags = update_quadrature_points;

        mf_data.reinit (dofs, constraints, quad, additional_data);
    }

  }


  template <int dim, int degree>
  void TestMixedLaplace<dim,degree>::assemble_sparse_system ()
  {
    QGauss<dim>   quadrature_formula(n_q_points_1d);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        local_matrix = 0;

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
              const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = fe_values[pressure].value (i, q);

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = fe_values[pressure].value (j, q);

                  local_matrix(i,j) += (phi_i_u * phi_j_u //assume unit K tensor
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * fe_values.JxW(q);
                }
            }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               local_matrix(i,j));
          }
      }

    // fill system_rhs with random numbers
    for (unsigned int i=0; i<2; ++i)
      for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
        {
          const double val = -1. + 2.*random_value<double>();
          system_rhs.block(i)(j) = val;
        }
  }

  template <int dim, int degree>
  void TestMixedLaplace<dim,degree>::compute_errors (int block_no) const
  {
	        // Verification
	        double error = 0.;
	        for (unsigned int i=0; i<2; ++i)
	          for (unsigned int j=0; j<solution.block(i).size(); ++j)
	                  error += std::fabs (solution.block(i)(j)-mf_solution.block(i)(j));

	        double relative = solution.block(block_no).l1_norm();

	        std::cout<<"solution L1 norm = "<<relative<<std::endl;

	        std::cout<<"Error = "<<error<<" Relative error = "<<error/relative<<std::endl;

	        deallog << "  Verification fe degree " << degree  <<  ": "
	                << error/relative << std::endl << std::endl;

	  	    if (error > 10e-6)
	  	    {
	  	      std::cout<<"Solution vector using dealii is "<<std::endl;
	  	      for (unsigned int i=0; i<2; ++i)
	  	      {
	  	    	  for (unsigned int j=0; j<solution.block(i).size(); ++j)
	  	    	  {
	  	    		  if (fabs(solution.block(i)(j)) < 10e-6)
	  	    			  std::cout<<std::setw(10)<<0;
	  	      		  else
	  	      			  std::cout<<std::setw(10)<<solution.block(i)(j);
	  	    	  }
	  	      		  std::cout<<std::endl;
	  	      }

	  	      std::cout<<"Solution vector using MF is "<<std::endl;
	  	      for (unsigned int i=0; i<2; ++i)
	  	      {
	  	    	  for (unsigned int j=0; j<mf_solution.block(i).size(); ++j)
	  	    	  {
	  	      		  if (fabs(mf_solution.block(i)(j)) < 10e-6)
	  	      			  std::cout<<std::setw(10)<<0;
	  	      		  else
	  	      			  std::cout<<std::setw(10)<<mf_solution.block(i)(j);
	  	      	  }
	  	      		  std::cout<<std::endl;
	  	      }
	  	   }
  }


  template <int dim, int degree>
  void TestMixedLaplace<dim,degree>::test_mf_operator (
		  	  	  	  	  	  	  	  enum op_id id,Timer &time,
		  	  	  	  	  	  	  	  BlockSparseMatrix<double> &tmp_system_matrix)
  {
	  //vmult using MF
      typedef  BlockVector<double> VectorType;
      MF_MixedLaplace<dim,degree,VectorType> mf (mf_data);

	  if (OP_A == id)
	  {
		  tmp_system_matrix.block(0,1) = 0;
		  tmp_system_matrix.block(1,0) = 0;
		  time.restart();
		  mf.Avmult(mf_solution, system_rhs);
	  }

	  if (OP_B == id)
	  {
		  tmp_system_matrix.block(0,0) = 0;
		  tmp_system_matrix.block(0,1) = 0;
		  time.restart();
		  mf.Bvmult(mf_solution, system_rhs);
	  }

	  if (OP_B_tra == id) //We test Au + B_tr.p
	  {
		  tmp_system_matrix.block(1,0) = 0;
		  time.restart();
		  mf.B_tra_vmult(mf_solution, system_rhs);
	  }

	  std::cout<<"MF operator evaluation, Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

	  time.restart();
	  //Solve using dealii sparse Matrix-vector multiplication
	  tmp_system_matrix.vmult (solution, system_rhs);
	  std::cout<<"Sparse Matrix vmult, Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
  }


  template <int dim, int degree>
  void TestMixedLaplace<dim,degree>::run ()
  {
	Timer time;

	std::cout<<std::endl<<"Making grid and distributing DoFs "<<std::endl;
	time.restart();
    make_grid_and_dofs();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Assembling the Sparse System Matrix and RHS"<<std::endl;
    time.restart();
    assemble_sparse_system ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Testing the operator evaluation with Sparse matrix and MatrixFree"<< std::endl;
    BlockSparseMatrix<double> tmp_system_matrix;
    tmp_system_matrix.reinit (sparsity_pattern);

    std::cout<<std::endl<<"Test for operator A.."<< std::endl;
    tmp_system_matrix.copy_from(system_matrix);
    test_mf_operator(OP_A,time,tmp_system_matrix);
    compute_errors (0);

    std::cout<<std::endl<<"Test for operator B_tr.."<< std::endl;
    tmp_system_matrix.copy_from(system_matrix);
    test_mf_operator(OP_B_tra,time,tmp_system_matrix);
    compute_errors (0);

    std::cout<<std::endl<<"Test for operator B.."<< std::endl;
    tmp_system_matrix.copy_from(system_matrix);
    test_mf_operator(OP_B,time,tmp_system_matrix);
    compute_errors (1);

  }



int main ()
{
	  deallog.attach(logfile);

	  deallog << std::setprecision (3);

  try
    {
	  deallog << std::endl << "Test with doubles" << std::endl << std::endl;
	  deallog.push("2d");
      TestMixedLaplace<2, 0> test1; test1.run(); //ok
      TestMixedLaplace<2, 1> test2; test2.run(); //ok
      TestMixedLaplace<2, 2> test3; test3.run(); //ok
      deallog.pop();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}


