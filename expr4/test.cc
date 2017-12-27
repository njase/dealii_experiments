#include "tests.h"

std::ofstream logfile("output");

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

//This is an attempt to compare results of vmulu operation of FEEvaluationGen with FEEvaluation on vector vlaued FE_Q
//Currently the results dont match so doing more basic tests in other files

#if 0

#include <deal.II/fe/fe_raviart_thomas.h>
int main()
{
	using namespace dealii;
	using namespace std;

	const int dim=2,fe_degree=3;
	typedef double Number;

	FE_RaviartThomas<dim> fe(fe_degree);

	FESystem<dim> fesys(fe,1, FE_Q<dim>(fe_degree+2),1);
	cout<<"fe.degree = "<<fesys.degree<<endl;


}
#endif

//Without this flag, now the code will not work since vector valued evaluation using
//scalar valued FE expects that the respective dofs are component wise distributed
//in src vector. Whereas, here I've changed it to just 2

#define GEN_IMPL

template <int dim, int degree_p, typename VectorType>
class MatrixFreeTest
{
public:
  typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
  typedef double Number;

  MatrixFreeTest(const MatrixFree<dim,Number> &data_in):
    data (data_in)
  {};

  void
  local_apply (const MatrixFree<dim,Number> &data,
               VectorType          &dst,
               const VectorType    &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    typedef VectorizedArray<Number> vector_t;
#ifdef GEN_IMPL
        FEEvaluationGen<FE_TaylorHood,degree_p+2,dim,degree_p+1,Number> velocity (data,0);
        FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p,Number> pressure (data,1);
#else
    FEEvaluation<dim,degree_p+1,degree_p+2,dim,Number> velocity (data, 0);
    FEEvaluation<dim,degree_p,degree_p+2,1,  Number> pressure (data, 1);
#endif

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src, 0);
        velocity.evaluate (false,true,false);
        pressure.reinit (cell);
#ifdef GEN_IMPL
        pressure.read_dof_values (src, 1);
#else
        pressure.read_dof_values (src, 1);
#endif
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
        velocity.distribute_local_to_global (dst, 0);
        pressure.integrate (true,false);
#ifdef GEN_IMPL
        pressure.distribute_local_to_global (dst, 1);
#else
        pressure.distribute_local_to_global (dst, 1);
#endif
      }
  }


  void vmult (VectorType &dst,
              const VectorType &src) const
  {
#ifndef GEN_IMPL
    AssertDimension (dst.size(), 2);
    for (unsigned int d=0; d<2; ++d)
      dst[d] = 0;
#endif
    data.cell_loop (&MatrixFreeTest<dim,degree_p,VectorType>::local_apply,
                    this, dst, src);
  };

private:
  const MatrixFree<dim,Number> &data;
};



template <int dim, int fe_degree>
void test ()
{
  Triangulation<dim>   triangulation;
  create_mesh (triangulation);
  if (fe_degree == 1)
    triangulation.refine_global (4-dim);
  else
    triangulation.refine_global (3-dim);

  FE_Q<dim>            fe_u (fe_degree+1);
  FE_Q<dim>            fe_p (fe_degree);
  FESystem<dim>        fe (fe_u, dim, fe_p, 1);
#ifdef GEN_IMPL
  DoFHandler<dim>      dof_handler_gen_u (triangulation);
#else
  DoFHandler<dim>      dof_handler_u (triangulation);
#endif
  DoFHandler<dim>      dof_handler_p (triangulation);
  DoFHandler<dim>      dof_handler (triangulation);

#ifdef GEN_IMPL
  FESystem<dim>        fe_gen_u (fe_u, dim);
  MatrixFree<dim,double> mf_data(true);
#else
  MatrixFree<dim,double> mf_data(false);
#endif

  ConstraintMatrix     constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;
  std::vector<Vector<double> > vec1, vec2;

#ifdef GEN_IMPL
  std::vector<Vector<double> > src, dst;
#endif

  dof_handler.distribute_dofs (fe);
#ifdef GEN_IMPL
  dof_handler_gen_u.distribute_dofs (fe_gen_u);
#else
  dof_handler_u.distribute_dofs (fe_u);
#endif
  dof_handler_p.distribute_dofs (fe_p);

  std::vector<unsigned int> block_component (dim+1,0); //dim+1 components in total
  block_component[dim] = 1; //For pressure = last component

  DoFRenumbering::component_wise (dof_handler,block_component);


  constraints.close ();

  std::vector<types::global_dof_index> dofs_per_block (2);
  DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);

  //std::cout << "   Number of active cells: "
  //          << triangulation.n_active_cells()
  //          << std::endl
  //          << "   Number of degrees of freedom: "
  //          << dof_handler.n_dofs()
  //          << " (" << n_u << '+' << n_p << ')'
  //          << std::endl;

  {
    BlockDynamicSparsityPattern csp (2,2);

    for (unsigned int d=0; d<2; ++d)
      for (unsigned int e=0; e<2; ++e)
        csp.block(d,e).reinit (dofs_per_block[d], dofs_per_block[e]);

    csp.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
    sparsity_pattern.copy_from (csp);
  }

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (2);
  vec1.resize (2);
  vec2.resize (2);
  for (unsigned int i=0; i<2; ++i)
  {
    solution.block(i).reinit (dofs_per_block[i]);
    vec1[i].reinit (dofs_per_block[i]);
    vec2[i].reinit (dofs_per_block[i]);
  }
  solution.collect_sizes ();

  system_rhs.reinit (solution);

#ifdef GEN_IMPL
  //velocity and pressure
  src.resize(2);
  dst.resize(2);

  //all components of velocity together
  src[0].reinit(dof_handler_gen_u.n_dofs());
  src[1].reinit(dof_handler_p.n_dofs());

  dst[0].reinit(dof_handler_gen_u.n_dofs());
  dst[1].reinit(dof_handler_p.n_dofs());


#endif

  // this is from step-22
  {
    QGauss<dim>   quadrature_formula(fe_degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |
                             update_JxW_values |
                             update_gradients);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

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
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                phi_grad_u[k] = fe_values[velocities].gradient (k, q);
                div_phi_u[k]  = fe_values[velocities].divergence (k, q);
                phi_p[k]      = fe_values[pressure].value (k, q);
              }

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
          }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=i+1; j<dofs_per_cell; ++j)
            local_matrix(i,j) = local_matrix(j,i);

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (local_matrix,
                                                local_dof_indices,
                                                system_matrix);
      }
  }

  // first system_rhs with random numbers
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
      {
        const double val = -1. + 2.*random_value<double>();
        system_rhs.block(i)(j) = val;
        vec1[i](j) = val;
#ifdef GEN_IMPL
        src[i](j) = val;
#endif
      }

  // setup matrix-free structure
  {
    std::vector<const DoFHandler<dim>*> dofs;
#ifdef GEN_IMPL
    dofs.push_back(&dof_handler_gen_u);
#else
    dofs.push_back(&dof_handler_u);
#endif
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

  system_matrix.vmult (solution, system_rhs);

  typedef std::vector<Vector<double> > VectorType;


  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
#ifdef GEN_IMPL
  mf.vmult (dst, src);
#else
  mf.vmult (vec2, vec1);
#endif


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
}


int main ()
{
  deallog.attach(logfile);

  deallog << std::setprecision (3);

  {
    deallog << std::endl << "Test with doubles" << std::endl << std::endl;
    deallog.push("2d");
    test<2,1>();
    //test<2,2>();
    //test<2,3>();
    //test<2,4>();
    deallog.pop();
    //deallog.push("3d");
    //test<3,1>();
    //test<3,2>();
    //deallog.pop();
  }
}
