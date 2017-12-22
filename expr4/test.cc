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

#if 0
using namespace dealii;
template <int dim, int n_components, int fe_degree, typename Number>
class MatrixFreeTest
{

	MatrixFree<dim,Number> mf_data;
	Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler;

public:
  MatrixFreeTest();
  ~MatrixFreeTest() {dof_handler.clear();triangulation.clear();mf_data.clear();};

  void test1();
  void test2();

  template<typename VectorType>
  void
  local_mass_operator (const MatrixFree<dim,Number>               &data,
                       VectorType                                 &dst,
                       const VectorType                           &src,
                       const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,n_components,Number> fe_eval (data);
    const unsigned int n_q_points = fe_eval.n_q_points;

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit (cell);
        fe_eval.read_dof_values (src);
        //not doing anything further...for simplicity of test
      }
  }
};


template <int dim, int n_components, int fe_degree, typename Number>
MatrixFreeTest<dim,n_components,fe_degree,Number>::MatrixFreeTest()
{
	  GridGenerator::hyper_cube (triangulation, -1, 1);
	  triangulation.refine_global(2);

	  FESystem<dim> fe (FE_Q<dim>(fe_degree), n_components); //vector valued FE

	  dof_handler.initialize(triangulation,fe);
	  {
		    const QGauss<1> quad (fe_degree+1);
		    typename MatrixFree<dim,Number>::AdditionalData data;
		    data.tasks_parallel_scheme =
		      MatrixFree<dim,Number>::AdditionalData::none;
		    ConstraintMatrix constraints;

		    mf_data.reinit (dof_handler, constraints, quad, data);
	  }
}

//test 1: No block structure
template <int dim, int n_components, int fe_degree, typename Number>
void
MatrixFreeTest<dim,n_components,fe_degree,Number>::test1()
{
	Vector<Number> src(dof_handler.n_dofs());
	Vector<Number> dst(dof_handler.n_dofs());

    mf_data.cell_loop (&MatrixFreeTest<dim,n_components,fe_degree,Number>::local_mass_operator,
                    this, dst, src);
}



//test 2: Values for each component are treated as a separate block
template <int dim, int n_components, int fe_degree, typename Number>
void
MatrixFreeTest<dim,n_components,fe_degree,Number>::test2()
{
    std::vector<types::global_dof_index> dofs_per_block (n_components);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_block);

    std::vector<Vector<double> > src, dst;
    src.resize (n_components);
    dst.resize (n_components);
      for (unsigned int i=0; i<n_components; ++i)
      {
    	  src[i].reinit (dofs_per_block[i]);
    	  dst[i].reinit (dofs_per_block[i]);
      }

      mf_data.cell_loop (&MatrixFreeTest<dim,n_components,fe_degree,Number>::local_mass_operator,
                      this, dst, src);
}


int main ()
{
    MatrixFreeTest<3,2,1,double> obj;
    std::cout<<"Test 1 started"<<std::endl;
    obj.test1();
    std::cout<<"Test 1 finished"<<std::endl<<std::endl;

    std::cout<<"Test 2 started"<<std::endl;
    obj.test2();
    std::cout<<"Test 2 finished"<<std::endl<<std::endl;
}

#endif



//#if 0

//#define GEN_IMPL

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
