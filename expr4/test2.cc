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
//Comparison of general output of FEEvaluation+MatrixFree
// and FEEvaluationGen+MatrixFree(non_primitive)


template <int dim, int degree_p, typename Number, typename VectorType>
class MatrixFreeTest
{
    using FEEvalGen_V =  FEEvaluationGen<FE_TaylorHood,degree_p+2,dim,degree_p+1,Number>;
    using FEEvalGen_P = FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p,Number>;

    using FEEval_V = FEEvaluation<dim,degree_p+1,degree_p+2,dim,Number>;
    using FEEval_P = FEEvaluation<dim,degree_p,degree_p+2,1,Number>;

public:
  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;

  MatrixFreeTest(const MatrixFree<dim,Number> &data_in, bool check_with_scalar=false):
	  	  	  data (data_in), n_q_points_1d(degree_p+2)
  {
	  deallog<<"Is MatrixFree using primitive element? = "<<data.is_primitive()<<std::endl;
	  deallog<<"n_array_elements on this machine = "<<n_array_elements<<std::endl;
	  deallog<<"n_q_points_1d = "<<n_q_points_1d<<std::endl;
	  deallog<<"No of (physical cells, macro cells) = ("<<data.n_physical_cells()<<", "<<data.n_macro_cells()<<")"<<std::endl;

  };

  template<typename TypeV, typename TypeP>
  void
  local_apply_vector (const MatrixFree<dim,Number> &data,
               VectorType          &dst,
               const VectorType    &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {

	typedef VectorizedArray<Number> vector_t;
	TypeV velocity (data, 0);
	TypeP pressure (data, 1);


    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (false,true,false);
        pressure.reinit (cell);
        pressure.read_dof_values (src.block(1));
        pressure.evaluate (true,false,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
            vector_t pres = pressure.get_value(q);
            vector_t div = -trace(grad_u);
            pressure.submit_value   (div, q);

            // subtract p * I
            for (unsigned int d=0; d<dim; ++d)
              grad_u[d][d] -= pres;

            velocity.submit_gradient(grad_u, q);
          }

        velocity.integrate (false,true);
        velocity.distribute_local_to_global (dst.block(0));
        pressure.integrate (true,false);
        pressure.distribute_local_to_global (dst.block(1));
      }

    //std::cout<<"Loop was internally run from "<<cell_range.first<<" to "<<cell_range.second<<std::endl;

}

  void vmult (VectorType &dst,
              const VectorType &src) const
  {
    if (data.is_primitive())
    {
    	data.cell_loop (&MatrixFreeTest<dim,degree_p,Number,VectorType>::local_apply_vector<FEEvalGen_V,FEEvalGen_P>,
                    this, dst, src);
    }
    else
    {
    		data.cell_loop (&MatrixFreeTest<dim,degree_p,Number,VectorType>::local_apply_vector<FEEval_V,FEEval_P>,
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
	typedef double Number;

  Triangulation<dim>   triangulation;
  create_mesh (triangulation);

  if (fe_degree == 1)
    triangulation.refine_global (4-dim);
  else
    triangulation.refine_global (3-dim);


  std::cout<<"No of active cells from triangulation = "<<triangulation.n_active_cells()<<std::endl;

  //Setup MatrixFree part

   //Common
  FE_Q<dim>            fe_p (fe_degree);
  DoFHandler<dim>      dof_handler_p (triangulation);
  dof_handler_p.distribute_dofs (fe_p);

  FE_Q<dim>            fe_u (fe_degree+1);

  FESystem<dim>        fe (fe_u, dim);
  DoFHandler<dim>      dof_handler (triangulation);
  dof_handler.distribute_dofs (fe);

  ConstraintMatrix dummy_constraints;
  dummy_constraints.close();
  std::vector<const ConstraintMatrix *> constraints;
  constraints.push_back (&dummy_constraints);
  constraints.push_back (&dummy_constraints);
  QGauss<1> quad(fe_degree+2);

  std::vector<const DoFHandler<dim>*> dofs_vector;
  //Remark: v then p == The same order should be maintained in reading inside local_apply
  dofs_vector.push_back(&dof_handler);
  dofs_vector.push_back(&dof_handler_p);

  std::cout<<"Initially dofs_per_cell(Pressure, velocity) = ("<<(dof_handler_p.get_fe()).n_dofs_per_cell()<<", "<<(dof_handler.get_fe()).n_dofs_per_cell()<<")"<<std::endl;


  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: "
            << triangulation.n_cells()
            << std::endl;

  //For FEEvaluation - using vector valued FE
  MatrixFree<dim,Number> mf_data_vec(false);
  mf_data_vec.reinit (dofs_vector, constraints, quad,
                  typename MatrixFree<dim>::AdditionalData
                  (MatrixFree<dim>::AdditionalData::none));

  //For FEEvaluationGen
  MatrixFree<dim,Number> mf_data_gen(true);
  mf_data_gen.reinit (dofs_vector, constraints, quad,
                          typename MatrixFree<dim>::AdditionalData
                          (MatrixFree<dim>::AdditionalData::none));

  //#5666: All components of velocity are treated as one block, pressure is treated as another block
  typedef  BlockVector<Number> VectorType;

  VectorType src_vec(2), dst_vec(2);
  //all components of velocity together
  src_vec.block(0).reinit(dof_handler.n_dofs());
  src_vec.block(1).reinit(dof_handler_p.n_dofs());
  src_vec.collect_sizes();
  dst_vec.reinit(src_vec);


  //Debug
  std::ofstream logfile_check_vec("check_vec_old");
  deallog.attach(logfile_check_vec);

  // init src with random numbers
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<src_vec.block(i).size(); ++j)
      {
        //const double val = -1. + 2.*random_value<Number>();
        const double val = random_value<Number>(0.1, 10.0);
        src_vec.block(i)(j) = val;

        deallog<<val << std::endl;
      }


  deallog.detach();

  //Actual Test with vector valued FE_Q
  std::ofstream logfile_vec_old("output_vec_old");
  std::ofstream logfile_new("output_vec_new");
  deallog << std::setprecision (3);
  deallog << std::setw (6);


  deallog.attach(logfile_vec_old);
  MatrixFreeTest<dim,fe_degree,Number, VectorType> mf_vec (mf_data_vec);
  mf_vec.vmult(dst_vec, src_vec);
  //Write to file
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<dst_vec.block(i).size(); ++j)
      {
        deallog<<dst_vec.block(i)(j)<< std::endl;
      }
  deallog.detach();


  deallog.attach(logfile_new);
  MatrixFreeTest<dim,fe_degree,Number, VectorType> mf_gen (mf_data_gen);
  mf_gen.vmult(dst_vec, src_vec);
  //Write to file
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<dst_vec.block(i).size(); ++j)
      {
        deallog<<dst_vec.block(i)(j)<< std::endl;
      }
  deallog.detach();


#if 0
  double error = 0., tol=1e-10;

  bool result = true;
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
    {
#ifdef GEN_IMPL
    	deallog << solution.block(i)(j) << "  "<<dst[i](j) << std::endl;
    	error += std::fabs (solution.block(i)(j)-dst[i](j));
#else
    	deallog << solution.block(i)(j) << "  "<<vec2[i](j) << std::endl;
    	error += std::fabs (solution.block(i)(j)-vec2[i](j));
#endif
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
