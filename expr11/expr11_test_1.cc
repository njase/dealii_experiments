//This is the final version which works.
//This is changed to block Vector. But I noticed that by using single valued blocks and dealii Block vectors result was still the same
//Just by following #if defs, the single values block case can be obtained.
// This test case has some interesting things:
//  It uses Vector valued FE_Q and Blockvectors. A bug in existing dealii functionality was found
//  It shows that FE_Q using new enhancements to MF framework work well
//  This uses Blockvector. A similar test will be written using std::vectors

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

const VectorizedArray<double> *dofs_mf = nullptr;

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
    FEEvaluationGen<FE_Q<dim>,degree_p+2,dim,degree_p+1,Number> velocity (data, 0);
    FEEvaluation<dim,degree_p,degree_p+2,1, Number> pressure (data, 1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        dofs_mf = velocity.begin_dof_values();
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
  }


  void vmult (VectorType &dst,
              const VectorType &src) const
  {
    //AssertDimension (dst.size(), dim+1);
    //for (unsigned int d=0; d<dim+1; ++d)
    //  dst[d] = 0;
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
  test_mesh<dim>(triangulation);
//#if 0
  if (fe_degree == 1)
    triangulation.refine_global (4-dim);
  else
    triangulation.refine_global (3-dim);
//#endif

  FESystem<dim>        fe_u (FE_Q<dim>(fe_degree+1), dim);
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
  BlockVector<double> src_vec;
  BlockVector<double> dst_vec;
  //std::vector<Vector<double> > vec1, vec2;


  dof_handler.distribute_dofs (fe);
  dof_handler_u.distribute_dofs (fe_u);
  dof_handler_p.distribute_dofs (fe_p);
  DoFRenumbering::component_wise (dof_handler);

    constraints.close ();
  //////////////////////////////

  int n_u = dof_handler_u.n_dofs();
  int n_p = dof_handler_p.n_dofs();


  {
	  BlockDynamicSparsityPattern dsp(2, 2);
	  dsp.block(0, 0).reinit (n_u, n_u);
	  dsp.block(1, 0).reinit (n_p, n_u);
	  dsp.block(0, 1).reinit (n_u, n_p);
	  dsp.block(1, 1).reinit (n_p, n_p);
	  dsp.collect_sizes ();
	  DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
	  sparsity_pattern.copy_from(dsp);
  }

  system_matrix.reinit (sparsity_pattern);

  //////////////////////////////

#if 0
  std::vector<types::global_dof_index> dofs_per_block (dim+1);
  DoFTools::count_dofs_per_component (dof_handler, dofs_per_block);

  //std::cout << "   Number of active cells: "
  //          << triangulation.n_active_cells()
  //          << std::endl
  //          << "   Number of degrees of freedom: "
  //          << dof_handler.n_dofs()
  //          << " (" << n_u << '+' << n_p << ')'
  //          << std::endl;

  {
    BlockDynamicSparsityPattern csp (dim+1,dim+1);

    for (unsigned int d=0; d<dim+1; ++d)
      for (unsigned int e=0; e<dim+1; ++e)
        csp.block(d,e).reinit (dofs_per_block[d], dofs_per_block[e]);

    csp.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
    sparsity_pattern.copy_from (csp);
  }
#endif

  system_matrix.reinit (sparsity_pattern);


  system_rhs.reinit (2);
  system_rhs.block(0).reinit (n_u);
  system_rhs.block(1).reinit (n_p);
  system_rhs.collect_sizes ();

  solution.reinit (system_rhs);

#if 0
  solution.reinit (dim+1);
  for (unsigned int i=0; i<dim+1; ++i)
    solution.block(i).reinit (dofs_per_block[i]);
  solution.collect_sizes ();

  system_rhs.reinit (solution);
#endif

#if 0
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

  src_vec.reinit (2);
  src_vec.block(0).reinit (n_u);
  src_vec.block(1).reinit (n_p);
  src_vec.collect_sizes ();
  dst_vec.reinit(src_vec);

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

#if 0
  // first system_rhs with random numbers
  int b_i = 0;
  int b_j = 0;
  for (unsigned int i=0; i<dim+1; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
      {
        const double val = -1. + 2.*random_value<double>();
        system_rhs.block(i)(j) = val;
        //vec1[i](j) = val;

        src_vec.block(b_i)(b_j++) = val;
        if (b_j >= src_vec.block(0).size())
        {
        	b_i++;
        	b_j = 0;
        }
      }
#endif

  // first system_rhs with random numbers
    float t = 1.0f;
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
      {
    	//all zeros
        const double val = -1. + 2.*random_value<double>();
        system_rhs.block(i)(j) = val; //t++;  //val;
      }

  system_matrix.vmult (solution, system_rhs);

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

  //typedef std::vector<Vector<double> > VectorType;
  typedef  BlockVector<double> VectorType;
  MatrixFreeTest<dim,fe_degree,VectorType> mf (mf_data);
  //mf.vmult (vec2, vec1);
  //Reorder system_rhs.block(0) and store into src_vec as MF expects
  for (int i=0; i<n_u/dim;i++)
  {
	  src_vec.block(0)(i*dim) = system_rhs.block(0)(i);
	  src_vec.block(0)(i*dim+1) = system_rhs.block(0)(n_u/dim+i);
  }
  src_vec.block(1) = system_rhs.block(1);

  BlockVector<double> tmp_dst_vec(dst_vec);
  mf.vmult(tmp_dst_vec, src_vec);

  //Reorder back the dst_vec
  for (int i=0; i<n_u/dim;i++)
  {
	  dst_vec.block(0)(i) = tmp_dst_vec.block(0)(i*dim);
	  dst_vec.block(0)(n_u/dim+i) = tmp_dst_vec.block(0)(i*dim+1);
  }
  dst_vec.block(1) = tmp_dst_vec.block(1);


  // Verification
  double error = 0.;
  for (unsigned int i=0; i<2; ++i)
    for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
      error += std::fabs (solution.block(i)(j)-dst_vec.block(i)(j));
  double relative = solution.block(0).l1_norm();
  std::cout<<"Error = "<<error<<" solution L1 norm = "<<relative<<std::endl;
  deallog << "  Verification fe degree " << fe_degree  <<  ": "
          << error/relative << std::endl << std::endl;

  if (error > 10e-6)
  {
	  std::cout<<"Solution vector using dealii is "<<std::endl;
	  for (unsigned int i=0; i<dim+1; ++i)
		  for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
			  std::cout<<std::setw(10)<<solution.block(i)(j);

	  std::cout<<"Solution vector using MF is "<<std::endl;
	  for (unsigned int i=0; i<2; ++i)
	  {
		  for (unsigned int j=0; j<dst_vec.block(i).size(); ++j)
		  {
			  std::cout<<std::setw(10)<<dst_vec.block(i)(j);
		  }
		  std::cout<<std::endl;
	  }
  }
}



int main ()
{
  deallog.attach(logfile);

  deallog << std::setprecision (3);

  {
    deallog << std::endl << "Test with doubles" << std::endl << std::endl;
    deallog.push("2d");
    //test<2,1>();
    //test<2,2>();
    //test<2,3>();
    test<2,4>();
    //deallog.pop();
    //deallog.push("3d");
    //test<3,1>();
    //test<3,2>();
    deallog.pop();
  }
}
