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

//#define GEN_IMPL

//#if 0
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
    //FEEvaluation<dim,degree_p+1,degree_p+2,dim,Number> velocity (data, 0);
        FEEvaluationGen<FE_TaylorHood,degree_p+2,dim,degree_p+1,Number> velocity (data,0);
        //FEEvaluation<dim,degree_p,degree_p+2,1,  Number> pressure (data, 1);
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
        pressure.read_dof_values (src, dim);
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
        pressure.distribute_local_to_global (dst, dim);
      }
  }


  void vmult (VectorType &dst,
              const VectorType &src) const
  {
    AssertDimension (dst.size(), dim+1);
    for (unsigned int d=0; d<dim+1; ++d)
      dst[d] = 0;
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

  FESystem<dim>       fe_u_new(fe_u, dim); //debug
#ifdef GEN_IMPL
  FESystem<dim>        fe_th_u (fe_u, dim);
#endif
  FE_Q<dim>            fe_p (fe_degree);
  FESystem<dim>        fe (fe_u, dim, fe_p, 1);
  DoFHandler<dim>      dof_handler_u (triangulation);
  DoFHandler<dim>      dof_handler_u_new (triangulation); //debug
#ifdef GEN_IMPL
  DoFHandler<dim>      dof_handler_th_u (triangulation);
#endif
  DoFHandler<dim>      dof_handler_p (triangulation);
  DoFHandler<dim>      dof_handler (triangulation);

#ifdef GEN_IMPL
  MatrixFree<dim,double> mf_data(true); //use primitive FE
#else
  MatrixFree<dim,double> mf_data(false); //use primitive FE
#endif

  ConstraintMatrix     constraints;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;
  std::vector<Vector<double> > vec1, vec2;

  dof_handler.distribute_dofs (fe);
  dof_handler_u.distribute_dofs (fe_u);
  dof_handler_u_new.distribute_dofs (fe_u_new); //debug
#ifdef GEN_IMPL
  dof_handler_th_u.distribute_dofs (fe_th_u);
#endif
  dof_handler_p.distribute_dofs (fe_p);
  DoFRenumbering::component_wise (dof_handler);

  /////debug
  //std::cout<<"n_dofs, n_locally_owned_dofs (OLD) = "<<dof_handler_u.n_dofs()<<", "<<dof_handler_u.n_locally_owned_dofs()<<std::endl;

  //std::cout<<"n_dofs, n_locally_owned_dofs (NEW) = "<<dof_handler_th_u.n_dofs()<<", "<<dof_handler_th_u.n_locally_owned_dofs()<<std::endl;

  //FESystem<dim>        debug_fe (FE_RaviartThomas<dim>(fe_degree),1);
  //std::cout<<"dofs_per_mesh - FE, FESys = "<<dof_handler_u.n_dofs()<<", "<<dof_handler_th_u.n_dofs()<<std::endl;

  /////////

  constraints.close ();

  //Note that we did not use block_component here. This means we did not combine
  //the different velocity components into single block => no. of blocks = no. of components

  std::vector<types::global_dof_index> dofs_per_block (dim+1);
  DoFTools::count_dofs_per_component (dof_handler, dofs_per_block);

  //std::cout << "   Number of active cells: "
  //          << triangulation.n_active_cells()
  //          << std::endl
  //          << "   Number of degrees of freedom: "
  //          << dof_handler.n_dofs()
  //          << " (" << n_u << '+' << n_p << ')'
  //          << std::endl;

  //std::cout<<"dofs_per_block values = "<<dofs_per_block[0]<<"  "<<dofs_per_block[1]<<"  "<<dofs_per_block[2]<<std::endl;

  {
    BlockDynamicSparsityPattern csp (dim+1,dim+1);

    for (unsigned int d=0; d<dim+1; ++d)
      for (unsigned int e=0; e<dim+1; ++e)
        csp.block(d,e).reinit (dofs_per_block[d], dofs_per_block[e]);

    csp.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
    sparsity_pattern.copy_from (csp);
  }

  system_matrix.reinit (sparsity_pattern);


  solution.reinit (dim+1);
  vec1.resize (dim+1);
  vec2.resize (dim+1);
  for (unsigned int i=0; i<dim+1; ++i)
  {
    solution.block(i).reinit (dofs_per_block[i]);
    vec1[i].reinit (dofs_per_block[i]);
    vec2[i].reinit (dofs_per_block[i]);
  }
  solution.collect_sizes ();

  system_rhs.reinit (solution);

#if 0
  //SAUR: I wonder why this was not done in earlier loop with solution vector
  //and why each individual element is not reinit to dofs_per_block[i]
  //in the end result will be the same because all velocity components have
  //equal dofs, but logically that would've been more clear
  //Just moved it
  vec1.resize (dim+1);
  vec2.resize (dim+1);
  vec1[0].reinit (dofs_per_block[0]);
  vec2[0].reinit (vec1[0]);
  for (unsigned int i=1; i<dim; ++i)
    {
      vec1[i].reinit (vec1[0]);
      vec2[i].reinit (vec1[0]);
    }
  vec1[dim].reinit (dofs_per_block[dim]);
  vec2[dim].reinit (vec1[dim]);
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
  for (unsigned int i=0; i<dim+1; ++i) //loop on blocks
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j) //loop on each element inside block
      {
        const double val = -1. + 2.*random_value<double>();
        system_rhs.block(i)(j) = val;
        vec1[i](j) = val;
      }

  // setup matrix-free structure
  {
    std::vector<const DoFHandler<dim>*> dofs;

#ifdef GEN_IMPL
    dofs.push_back(&dof_handler_th_u);
#else
    //dofs.push_back(&dof_handler_u); debug
    dofs.push_back(&dof_handler_u_new); //debug
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

  //vmult using traditional approach
  system_matrix.vmult (solution, system_rhs);


  //vmult using MF approach
  typedef std::vector<Vector<double> > VectorType;
  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
  mf.vmult (vec2, vec1); // => vec2 = SystemMatrixMF * system_rhs


  // Verification
  double error = 0., tol=1e-10;
  bool result = true;
  for (unsigned int i=0; i<dim+1; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
    {
    	error += std::fabs (solution.block(i)(j)-vec2[i](j));
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
    //deallog.push("2d");
    //test<2,1>();
    //test<2,2>();
    //test<2,3>();
    //test<2,4>();
    //deallog.pop();
    deallog.push("3d");
    test<3,1>();
    //test<3,2>();
    deallog.pop();
  }
}
//#endif

#if 0
// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------



// Tests CellwiseInverseMassMatrix on vector DG elements, otherwise the same
// as inverse_mass_01.cc

#include "tests.h"
#include <deal.II/base/function.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/operators.h>



template <int dim, int fe_degree, typename Number, typename VectorType=Vector<Number> >
class MatrixFreeTest
{
public:

  MatrixFreeTest(const MatrixFree<dim,Number> &data_in):
    data (data_in)
  {};

  void
  local_mass_operator (const MatrixFree<dim,Number>               &data,
                       VectorType                                 &dst,
                       const VectorType                           &src,
                       const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> fe_eval (data);
    const unsigned int n_q_points = fe_eval.n_q_points;

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit (cell);
        fe_eval.read_dof_values (src);
        fe_eval.evaluate (true, false);
        for (unsigned int q=0; q<n_q_points; ++q)
          fe_eval.submit_value (fe_eval.get_value(q),q);
        fe_eval.integrate (true, false);
        fe_eval.distribute_local_to_global (dst);
      }
  }

  void
  local_inverse_mass_operator (const MatrixFree<dim,Number>               &data,
                               VectorType                                 &dst,
                               const VectorType                           &src,
                               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> fe_eval (data);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,Number> mass_inv(fe_eval);
    const unsigned int n_q_points = fe_eval.n_q_points;
    AlignedVector<VectorizedArray<Number> > inverse_coefficients(n_q_points);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit (cell);
        mass_inv.fill_inverse_JxW_values(inverse_coefficients);
        fe_eval.read_dof_values (src);
        mass_inv.apply(inverse_coefficients, dim, fe_eval.begin_dof_values(),
                       fe_eval.begin_dof_values());
        fe_eval.distribute_local_to_global (dst);
      }
  }

  void vmult (VectorType   &dst,
              const VectorType &src) const
  {
    dst = 0;
    data.cell_loop (&MatrixFreeTest<dim,fe_degree,Number,VectorType>::local_mass_operator,
                    this, dst, src);
  };

  void apply_inverse (VectorType   &dst,
                      const VectorType &src) const
  {
    dst = 0;
    data.cell_loop (&MatrixFreeTest<dim,fe_degree,Number,VectorType>::local_inverse_mass_operator,
                    this, dst, src);
  };

private:
  const MatrixFree<dim,Number> &data;
};



template <int dim, int fe_degree, typename number>
void do_test (const DoFHandler<dim> &dof)
{

  deallog << "Testing " << dof.get_fe().get_name() << std::endl;

  MatrixFree<dim,number> mf_data;
  {
    const QGauss<1> quad (fe_degree+1);
    typename MatrixFree<dim,number>::AdditionalData data;
    data.tasks_parallel_scheme =
      MatrixFree<dim,number>::AdditionalData::partition_color;
    data.tasks_block_size = 3;
    ConstraintMatrix constraints;

    mf_data.reinit (dof, constraints, quad, data);
  }

  MatrixFreeTest<dim,fe_degree,number> mf (mf_data);
  Vector<number> in (dof.n_dofs()), inverse (dof.n_dofs()), reference(dof.n_dofs());

  for (unsigned int i=0; i<dof.n_dofs(); ++i)
    {
      const double entry = random_value<double>();
      in(i) = entry;
    }

  mf.apply_inverse (inverse, in);

  SolverControl control(1000, 1e-12);
  std::ostringstream stream;
  deallog.attach(stream);
  SolverCG<Vector<number> > solver(control);
  solver.solve (mf, reference, in, PreconditionIdentity());
  deallog.attach(logfile);

  inverse -= reference;
  const double diff_norm = inverse.linfty_norm() / reference.linfty_norm();

  deallog << "Norm of difference: " << diff_norm << std::endl << std::endl;
}



template <int dim, int fe_degree>
void test ()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube (tria, -1, 1);
  tria.refine_global(1);
  if (dim < 3 || fe_degree < 2)
    tria.refine_global(1);
  tria.begin(tria.n_levels()-1)->set_refine_flag();
  tria.last()->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  typename Triangulation<dim>::active_cell_iterator
  cell = tria.begin_active (),
  endc = tria.end();
  for (; cell!=endc; ++cell)
    if (cell->center().norm()<1e-8)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  const unsigned int degree = fe_degree;
  //FESystem<dim> fe (FE_DGQ<dim>(degree), dim,FE_Q<dim>(degree),1); - debug
  FESystem<dim> fe (FE_DGQ<dim>(degree), dim);
  DoFHandler<dim> dof (tria);
  dof.distribute_dofs(fe);

  do_test<dim, fe_degree, double> (dof);
}



int main ()
{
  deallog.attach(logfile);

  deallog << std::setprecision (3);

  {
   // deallog.push("2d");
   // test<2,1>();
   // test<2,2>();
  //  test<2,4>();
  //  deallog.pop();
    deallog.push("3d");
    test<3,1>();
  //  test<3,2>();
    deallog.pop();
  }
}
#endif
