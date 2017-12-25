#include "tests.h"

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
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


template <int dim, int degree_p, typename VectorType>
class MatrixFreeTest
{

public:
  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
  typedef double Number;

  MatrixFreeTest(const MatrixFree<dim,Number> &data_in, const MatrixFree<dim,Number> &data_gen_in):
    data (data_in), data_gen(data_gen_in)
  {
  };


  void
  local_apply (const MatrixFree<dim,Number> &data,
               VectorType          &dst,
               const VectorType    &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
	  std::ofstream logfile("output_old");

	  deallog.attach(logfile);

	  deallog << std::setprecision (3);

    typedef VectorizedArray<Number> vector_t;

	VectorizedArray<Number> * val, *val_gen;

    FEEvaluation<dim,degree_p+1,degree_p+2,dim,Number> velocity (data, 0);
    FEEvaluation<dim,degree_p,degree_p+2,1,  Number> pressure (data, 1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
         	pressure.reinit(cell);
            pressure.read_dof_values (src, 1);
            pressure.evaluate (true,false,false);

            //break;
      }

    val = pressure.begin_values();

    int n_array_elements = VectorizedArray<Number>::n_array_elements;
    int max_i = src[1].size()/n_array_elements;

    deallog<<"Values for primitive = "<<data.is_primitive()<<std::endl;
    deallog<<"n_array_elements = "<<n_array_elements<<std::endl;

    for (int i =0; i<max_i; i++)
    {
    	for (unsigned int j=0; j<n_array_elements; j++)
    	{
    		deallog<<val[i][j] << std::endl;
    	}
    }

    deallog.detach();

  }

  void
  local_apply_gen (const MatrixFree<dim,Number> &data,
               VectorType          &dst,
               const VectorType    &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
	  std::ofstream logfile("output_new");

	  deallog.attach(logfile);

	  deallog << std::setprecision (3);

    typedef VectorizedArray<Number> vector_t;

	VectorizedArray<Number> * val, *val_gen;

    FEEvaluationGen<FE_TaylorHood,degree_p+2,dim,degree_p+1,Number> velocity (data,0);
    FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p,Number> pressure (data,1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
         	pressure.reinit(cell);
            pressure.read_dof_values (src, 1);
            pressure.evaluate (true,false,false);
      }

    val = pressure.begin_values();

    int n_array_elements = VectorizedArray<Number>::n_array_elements;
    int max_i = src[1].size()/n_array_elements;

    deallog<<"Values for primitive = "<<data.is_primitive()<<std::endl;
    deallog<<"n_array_elements = "<<n_array_elements<<std::endl;

    for (int i =0; i<max_i; i++)
    {
    	for (unsigned int j=0; j<n_array_elements; j++)
    	{
    		deallog<<val[i][j] << std::endl;
    	}
    }

    deallog.detach();
  }


  void vmult (VectorType &dst,
              const VectorType &src) const
  {
    data.cell_loop (&MatrixFreeTest<dim,degree_p,VectorType>::local_apply,
                    this, dst, src);

    data_gen.cell_loop (&MatrixFreeTest<dim,degree_p,VectorType>::local_apply_gen,
                   this, dst, src);
  };

private:
  const MatrixFree<dim,Number> &data;
  const MatrixFree<dim,Number> &data_gen;
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

  FE_Q<dim>            fe_u (fe_degree+1);
  FESystem<dim>        fe_gen_u (fe_u, dim);

  FE_Q<dim>            fe_p (fe_degree);

  DoFHandler<dim>      dof_handler_gen_u (triangulation);
  DoFHandler<dim>      dof_handler_u (triangulation);

  DoFHandler<dim>      dof_handler_p (triangulation);

  FESystem<dim>        fe (fe_u, dim, fe_p, 1);
  DoFHandler<dim>      dof_handler(triangulation);


  MatrixFree<dim,Number> mf_data(false);
  MatrixFree<dim,Number> mf_data_gen(true);

  dof_handler_gen_u.distribute_dofs (fe_gen_u);
  dof_handler_u.distribute_dofs (fe_u);
  dof_handler_p.distribute_dofs (fe_p);

  dof_handler.distribute_dofs(fe);


  // setup matrix-free structure

    std::vector<const DoFHandler<dim>*> dofs, dofs_gen;

    dofs_gen.push_back(&dof_handler_gen_u);
    dofs_gen.push_back(&dof_handler_p);

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


    mf_data_gen.reinit (dofs_gen, constraints, quad,
                        typename MatrixFree<dim>::AdditionalData
                        (MatrixFree<dim>::AdditionalData::none));

    std::vector<Vector<double> > src(2), dst(2);
    //all components of velocity together
    src[0].reinit(dof_handler_gen_u.n_dofs());
    src[1].reinit(dof_handler_p.n_dofs());

    // first system_rhs with random numbers
    for (unsigned int i=0; i<2; ++i)
      for (unsigned int j=0; j<src[i].size(); ++j)
        {
          const double val = -1. + 2.*random_value<double>();
          src[i](j) = val;
        }


    typedef std::vector<Vector<double> > VectorType;
    MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data, mf_data_gen);
    mf.vmult (dst, src);

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
