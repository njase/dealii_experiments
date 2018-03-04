/*
 * This test shows how to solve a mixed duffusion problem using the Hdiv
 * confirming Raviart-Thomas elements using MatrixFree framework.
 * This is the same problem which is solved in step-20 using sparse matrix
 * based approach.
 * The MF framework for anisotropic RT elements is still in development.
 * The basic operators are ready, but there are some features which are
 * not supported.
 * Notable differences from Step-20:
 * 1. Only cartesian mesh cells should be used
 * 2. Permeability tensor must be proportional to Identity Matrix
 * 3. Face integrals can't be evaluated. the corresponding RHS integration
 *    term has to be provided from outside the MF framework (we take it
 *    as constant in this test)
 * 4. MinRes Solver is used - This is not a limitation, but used only for
 *    experimentation and demonstration purposes
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
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


std::ofstream logfile ("iterations.log");


using namespace dealii;

template <int dim,int degree>
class MixedLaplaceBase
{
public:
	MixedLaplaceBase (Triangulation<dim> &tri);

	SolverControl::State write_intermediate_solution
	  	  	  	  	  (const unsigned int    iteration,
	                   const double check_value,
	                   const BlockVector<double> &current_iterate) const;

	int convergence_steps() {return solver_control.last_step();};

protected:
  static const unsigned int   n_q_points_1d = degree+2;

  Triangulation<dim>   &triangulation;

  FE_RaviartThomas<dim> fe_u;
  FE_DGQ<dim>           fe_p;

  SolverControl solver_control;
  bool record_intermediate;


  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    template <typename number>
    VectorizedArray<number> value (const Point<dim,VectorizedArray<number>> &p,
                                    const unsigned int component = 0) const;
  };

  class PressureBoundaryValues : public Function<dim>
  {
	  bool const_bry;
  public:
    PressureBoundaryValues (bool is_const_bry = true) :
    					Function<dim>(1), const_bry(is_const_bry)
    					{}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    double constant_boundary_value() const;
  };
};

template <int dim,int degree>
MixedLaplaceBase<dim,degree>::MixedLaplaceBase (Triangulation<dim> &tri)
  :
  triangulation(tri),
  fe_u(FE_RaviartThomas<dim>(degree)),
  fe_p(FE_DGQ<dim>(degree)),
  solver_control(1000,1e-12),
  record_intermediate(false)
{}


template <int dim,int degree>
SolverControl::State
MixedLaplaceBase<dim,degree>::write_intermediate_solution (
				const unsigned int    iteration,
                const double check_value,
                const BlockVector<double> &current_iterate) const
{
	  logfile<<"ITERATION "<<iteration<<"("<<check_value<<") = ";
	  for (int i=0; i<2; i++)
		  for (int j=0; j<current_iterate.block(i).size(); j++)
	  		  logfile << current_iterate.block(i)(j)<<"  ";
	  logfile<<std::endl;

    return SolverControl::success;
}

template <int dim,int degree>
double
MixedLaplaceBase<dim,degree>::RightHandSide::
value (const Point<dim>  &p, const unsigned int /*component*/) const
{
  //return 0; //as in dealii
  return 1. / (0.05 + 2.*p.square()); //for experimentation
}


template <int dim,int degree>
template <typename number>
VectorizedArray<number>
MixedLaplaceBase<dim,degree>::RightHandSide::
value (const Point<dim,VectorizedArray<number>> &p,
                                const unsigned int /*component*/) const
{
	//return VectorizedArray<number>(); //= 0 as in dealii step 20
    return 1. / (0.05 + 2.*p.square()); //for experimentation
}



template <int dim,int degree>
double
MixedLaplaceBase<dim,degree>::PressureBoundaryValues::
value (const Point<dim>  &p, const unsigned int /*component*/) const
{
	  if (const_bry)
		  return constant_boundary_value();

	//Else, non constant boundary as earlier in dealii
  const double alpha = 0.3;
  const double beta = 1;
  return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

}

template <int dim,int degree>
double
MixedLaplaceBase<dim,degree>::PressureBoundaryValues::
constant_boundary_value () const
{
	  Assert(const_bry == true, ExcInvalidState ());

	  return 1; //const. non-zero boundary
}



////////// Sparse matrix approach, from in step20
template <int dim,int degree>
class MixedLaplaceSparse : public MixedLaplaceBase<dim,degree>
{
	using Base = MixedLaplaceBase<dim,degree>;
	using Base::fe_p;
	using Base::fe_u;
	using Base::n_q_points_1d;
	using Base::triangulation;
	using Base::solver_control;

private:
    FESystem<dim>        fe;

    DoFHandler<dim>      dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double>       solution;
    BlockVector<double>       system_rhs;

    //Private classes
    class KInverse : public TensorFunction<2,dim>
    {
    public:
      KInverse () : TensorFunction<2,dim>() {}

      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const;
    };

public:
    MixedLaplaceSparse (Triangulation<dim> &tri);
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve();
    const BlockVector<double> & get_solution();
    size_t n_dofs() {return dof_handler.n_dofs();};
};

template <int dim,int degree>
MixedLaplaceSparse<dim,degree>::MixedLaplaceSparse (Triangulation<dim> &tri)
  :
  Base(tri),
  fe (fe_u,1,fe_p,1),
  dof_handler (triangulation)
{}


template <int dim,int degree>
void
MixedLaplaceSparse<dim,degree>::KInverse::
value_list (const std::vector<Point<dim> > &points,
                           std::vector<Tensor<2,dim> >    &values) const
{
  Assert (points.size() == values.size(),
          ExcDimensionMismatch (points.size(), values.size()));

  for (unsigned int p=0; p<points.size(); ++p)
    {
      values[p].clear ();

      for (unsigned int d=0; d<dim; ++d)
        values[p][d][d] = 1.;
    }
}

template <int dim,int degree>
void
MixedLaplaceSparse<dim,degree>::make_grid_and_dofs()
{
    dof_handler.distribute_dofs (fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim];

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

    system_rhs.reinit (solution);
}

template <int dim,int degree>
void
MixedLaplaceSparse<dim,degree>::assemble_system()
{
  QGauss<dim>   quadrature_formula(n_q_points_1d);
  QGauss<dim-1> face_quadrature_formula(n_q_points_1d);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    | update_gradients |
                           update_quadrature_points  | update_JxW_values);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                    update_values    | update_normal_vectors |
                                    update_quadrature_points  | update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int   n_face_q_points = face_quadrature_formula.size();
  const unsigned int 	 n_u = fe_u.dofs_per_cell; //For velocity dofs

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const typename Base::RightHandSide          right_hand_side;
  const typename Base::PressureBoundaryValues pressure_boundary_values;
  const KInverse                     k_inverse;

  std::vector<double> rhs_values (n_q_points);
  std::vector<double> boundary_values (n_face_q_points);
  std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      local_matrix = 0;
      local_rhs = 0;

      right_hand_side.value_list (fe_values.get_quadrature_points(),
                                  rhs_values);
      k_inverse.value_list (fe_values.get_quadrature_points(),
                            k_inverse_values);

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

                local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                      - div_phi_i_u * phi_j_p
                                      - phi_i_p * div_phi_j_u)
                                     * fe_values.JxW(q);
              }

            //Since DOFs are ordered as velocity followed by pressure,
            //this only works on pressure test functions
            local_rhs(i) += -phi_i_p *
                            rhs_values[q] *
                            fe_values.JxW(q);
          }

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             local_matrix(i,j));

        }
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += local_rhs(i);

      //Set face related RHS as constant
      system_rhs.block(0) = pressure_boundary_values.constant_boundary_value();
    }
}

template <int dim,int degree>
void
MixedLaplaceSparse<dim,degree>::solve()
{
    SolverMinRes<BlockVector<double>> mres (solver_control);

    if (this->record_intermediate)
    {
    	using std::placeholders::_1;
    	using std::placeholders::_2;
    	using std::placeholders::_3;
    	auto conn = mres.connect (std::bind (&Base::write_intermediate_solution,
                                    this,_1,_2,_3));

    	logfile<<"Intermediate results with traditional approach"<<std::endl;
    }

    mres.solve(system_matrix,solution,system_rhs,PreconditionIdentity());
}

template <int dim,int degree>
const BlockVector<double> &
MixedLaplaceSparse<dim,degree>::get_solution()
{
	return solution;
}

///////////////// MF approach

template <int dim,int degree>
class MixedLaplaceMF : public MixedLaplaceBase<dim,degree>
{
	using Base = MixedLaplaceBase<dim,degree>;
	using Base::fe_p;
	using Base::fe_u;
	using Base::n_q_points_1d;
	using Base::triangulation;
	using Base::solver_control;
	typedef  BlockVector<double> VectorType;
	typedef double Number;

private:
    DoFHandler<dim>      dof_handler_u;
	DoFHandler<dim>      dof_handler_p;

    MatrixFree<dim,double> 	mf_data;

    BlockVector<double> 	  mf_solution;
    BlockVector<double>       mf_rhs;

    class MFKernel
    {
    private:
    	const MatrixFree<dim,Number> &data;

    public:
      typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
      MFKernel(const MatrixFree<dim,Number> &data_in):
      	data (data_in)
      {};

      //Take inverse permeability matrix as unity
      void
      local_apply_vector (const MatrixFree<dim,Number> &data,
                   VectorType          &dst,
                   const VectorType    &src,
                   const std::pair<unsigned int,unsigned int> &cell_range) const
      {
        typedef VectorizedArray<Number> vector_t;
        FEEvaluationAni<FE_RaviartThomas<dim>,dim,degree,n_q_points_1d,Number> velocity (data, 0);
        FEEvaluation<dim,degree,n_q_points_1d,1,Number> pressure (data, 1); //For scalar elements, use orig FEEvaluation


        for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
        	velocity.reinit (cell);
            velocity.read_dof_values (src.block(0));
            //Evaluate values and gradients
            velocity.evaluate (true,true,false);
            pressure.reinit (cell);
            pressure.read_dof_values (src.block(1));
            pressure.evaluate (true,false,false);

            for (unsigned int q=0; q<velocity.n_q_points; ++q)
            {
            	Tensor<1,dim,vector_t> u = velocity.get_value(q);
                velocity.submit_value (u, q);

                Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
                vector_t pres = pressure.get_value(q);
                vector_t div = -trace(grad_u);
                pressure.submit_value (div, q);

                grad_u.clear();
                for (unsigned int d=0; d<dim; ++d)
                  grad_u[d][d] = -pres;

                velocity.submit_gradient(grad_u, q);
            }

            velocity.integrate (true,true);
            velocity.distribute_local_to_global (dst.block(0));
            pressure.integrate (true,false);
            pressure.distribute_local_to_global (dst.block(1));

        }
    }

    void
    local_rhs (const MatrixFree<dim,Number> &data,
                    VectorType          &dst,
                    const VectorType    &/*src*/,
                    const std::pair<unsigned int,unsigned int> &cell_range) const
    {
         typedef VectorizedArray<Number> vector_t;
         FEEvaluation<dim,degree,n_q_points_1d,1,Number> pressure (data, 1);

         const typename Base::RightHandSide   right_hand_side;
         vector_t val;
         typename Base::PressureBoundaryValues pressure_boundary_values;

         for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
         {
         	   pressure.reinit(cell);

      	   for (unsigned int q=0; q<pressure.n_q_points; ++q)
      	   {
      		   val = right_hand_side.value(pressure.quadrature_point(q));

      		   pressure.submit_value (-val,q);
      	   }

      	   pressure.integrate (true,false);
             pressure.distribute_local_to_global (dst.block(1));
         }

         dst.block(0) = pressure_boundary_values.constant_boundary_value();
     }


      void vmult (VectorType &dst,
                  const VectorType &src) const
      {
      	dst.block(0) = 0;
      	dst.block(1) = 0;
      	data.cell_loop (&MixedLaplaceMF<dim,degree>::MFKernel::local_apply_vector,
                      this, dst, src);

      };

      void rhsvmult (VectorType &dst,
                  const VectorType &src) const
      {
    	  data.cell_loop (&MixedLaplaceMF<dim,degree>::MFKernel::local_rhs,
                      this, dst, src);

      };
    };


public:
    MixedLaplaceMF (Triangulation<dim> &tri);
    void make_grid_and_dofs();
    void assemble_system();
    void solve();
    const BlockVector<double> & get_solution();
    size_t n_dofs() {return (dof_handler_u.n_dofs()+dof_handler_p.n_dofs()); };
};

template <int dim,int degree>
MixedLaplaceMF<dim,degree>::MixedLaplaceMF (Triangulation<dim> &tri)
  :
  Base(tri),
  dof_handler_u (triangulation),
  dof_handler_p (triangulation)
{}

template <int dim,int degree>
void
MixedLaplaceMF<dim,degree>::make_grid_and_dofs()
{
    dof_handler_u.distribute_dofs (fe_u);
    dof_handler_p.distribute_dofs (fe_p);

    mf_solution.reinit (2);
    mf_solution.block(0).reinit (dof_handler_u.n_dofs());
    mf_solution.block(1).reinit (dof_handler_p.n_dofs());
    mf_solution.collect_sizes ();

    mf_rhs.reinit (mf_solution);
}

template <int dim,int degree>
void
MixedLaplaceMF<dim,degree>::assemble_system()
{
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

    MFKernel rhs_ker(mf_data);
    rhs_ker.rhsvmult(mf_rhs, BlockVector<double>());

}

template <int dim,int degree>
void
MixedLaplaceMF<dim,degree>::solve()
{
	SolverMinRes<BlockVector<double>> mres (solver_control);

	if (this->record_intermediate)
	{
		using std::placeholders::_1;
		using std::placeholders::_2;
		using std::placeholders::_3;
		auto conn = mres.connect (std::bind (&Base::write_intermediate_solution,
                                     this,_1,_2,_3));

    logfile<<std::endl<<std::endl<<"Intermediate results with MatrixFree approach"<<std::endl;
	}

    MFKernel op_ker(mf_data);
    mres.solve(op_ker,mf_solution,mf_rhs,PreconditionIdentity());
}

template <int dim,int degree>
const BlockVector<double> &
MixedLaplaceMF<dim,degree>::get_solution()
{
	return mf_solution;
}

/////////////////////////
bool compute_errors (const BlockVector<double> &solution,
					 const BlockVector<double> &mf_solution)
{
     // Verification
     double error = 0.;
     for (unsigned int i=0; i<2; ++i)
        for (unsigned int j=0; j<solution.block(i).size(); ++j)
                error += std::fabs (solution.block(i)(j)-mf_solution.block(i)(j));

      double relative = solution.block(0).l1_norm();

      std::cout<<std::endl<<"Error = "<<error<<" solution L1 norm = "<<relative<<std::endl;

	  std::cout<<" Relative error = "<< error/relative << std::endl << std::endl;

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

	  	  return false;
	 }

	 return true;
}

template <typename T>
void run(T &obj)
{
	Timer time;

	std::cout<<std::endl<<"====================";

	std::cout<<std::endl<<"Distributing DoFs, ";
	time.restart();
    obj.make_grid_and_dofs();
    std::cout<<"Time taken CPU = "<<time.cpu_time() << "s/" << "WALL = "<<time.wall_time() << "s";

    std::cout<<std::endl<<"Assembling the System Matrix and RHS, ";
    time.restart();
    obj.assemble_system ();
    std::cout<<"Time taken CPU = "<<time.cpu_time() << "s/" << "WALL = "<<time.wall_time() << "s";

    std::cout<<std::endl<<"Solving the Linear system, ";
    time.restart();
    obj.solve ();
    std::cout<<"Time taken CPU = "<<time.cpu_time() << "s/" << "WALL = "<<time.wall_time() << "s";

}

int main()
{
	Triangulation<2> triangulation;
    //Use only cartesian mesh cells
	GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (3);


    constexpr int dim = 2;
    constexpr int degree = 2;

    std::cout<<std::endl<<"Running test with Sparse Matrix approach ";
    MixedLaplaceSparse<dim,degree> spMv_test(triangulation);
    run(spMv_test);

    std::cout<<std::endl<<std::endl<<"Running test with Matrix free approach ";
    MixedLaplaceMF<dim,degree> MF_test(triangulation);
    run(MF_test);

    compute_errors(spMv_test.get_solution(), MF_test.get_solution());

    std::cout << "Number of active cells = "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells = "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom = "
              << MF_test.n_dofs()
			  <<std::endl
			  << "Convergence iterations = "
			  << MF_test.convergence_steps()
              << std::endl;

	return 0;
}

