/*
 * Adaptation of step20 (mixed laplace problem) to work with matrixFree framework
 */

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


int g_stop = 0;
std::ofstream logfile ("orig-solution.log");
std::ofstream newlogfile ("new-solution.log");

namespace Step20
{
  using namespace dealii;

  template<int> class RightHandSide;
  template <int dim> class PressureBoundaryValues;

  template <int dim, int degree_p, typename VectorType>
  class MF_MixedLaplaceProblem
  {
  public:
    typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
    typedef double Number;

    static constexpr unsigned int  n_q_points_1d = degree_p+2;

    MF_MixedLaplaceProblem(const MatrixFree<dim,Number> &data_in):
    	data (data_in)
    	//n_q_points_1d(degree_p+2)
    {};

    //FIXME We are ignoring inverse permeability matrix for time being ==> taking as unit matrix
    void
    local_apply_vector (const MatrixFree<dim,Number> &data,
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

      //std::cout<<"Loop was internally run from "<<cell_range.first<<" to "<<cell_range.second<<std::endl;

  }

    void
     local_rhs (const MatrixFree<dim,Number> &data,
                  VectorType          &dst,
                  const VectorType    &/*src*/,
                  const std::pair<unsigned int,unsigned int> &cell_range) const
  {
       typedef VectorizedArray<Number> vector_t;
       FEEvaluation<dim,degree_p,n_q_points_1d,1,Number> pressure (data, 1);

       const RightHandSide<dim>   right_hand_side;
       vector_t val;
       PressureBoundaryValues<dim> pressure_boundary_values;

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

       //std::cout<<"Loop was internally run from "<<cell_range.first<<" to "<<cell_range.second<<std::endl;

   }


    void vmult (VectorType &dst,
                const VectorType &src) const
    {
    	dst.block(0) = 0;
    	dst.block(1) = 0;
  	  data.cell_loop (&MF_MixedLaplaceProblem<dim,degree_p,VectorType>::local_apply_vector,
                    this, dst, src);

    };

    void rhsvmult (VectorType &dst,
                const VectorType &src) const
    {
  	  data.cell_loop (&MF_MixedLaplaceProblem<dim,degree_p,VectorType>::local_rhs,
                    this, dst, src);

    };

  private:
    const MatrixFree<dim,Number> &data;
  };


  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem (const unsigned int degree);
    void run ();

  private:
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve_minres ();
    void compute_errors () const;
    void output_results () const;

    const unsigned int   degree;

    const unsigned int   n_q_points_1d;

    Triangulation<dim>   triangulation;

    FESystem<dim>        fe;

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

    void test_assembly ();
    void test_rhs_assembly();
    SolverControl::State write_intermediate_solution (const unsigned int    iteration,
                                        const double check_value,
                                        const BlockVector<double> &current_iterate) const;
  };


  template <int dim>
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



  template <int dim>
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


  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int /*component*/) const
  {
    //return 0; //as in dealii
    return 1. / (0.05 + 2.*p.square()); //for experimentation
  }

  template <int dim>
  template <typename number>
  VectorizedArray<number> RightHandSide<dim>::value (const Point<dim,VectorizedArray<number>> &p,
                                  const unsigned int /*component*/) const
  {
	  //return VectorizedArray<number>(); //= 0 as in dealii step 20
      return 1. / (0.05 + 2.*p.square()); //for experimentation
  }


  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
	  if (const_bry)
		  return constant_boundary_value();

	//Else, non constant boundary as earlier in dealii
    const double alpha = 0.3;
    const double beta = 1;
    return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

  }

  template <int dim>
  double PressureBoundaryValues<dim>::constant_boundary_value () const
  {
	  Assert(const_bry == true, ExcInvalidState ());

	  return 1; //something const. non-zero boundary. With zero boundary, the iteration does not proceed
  }



  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
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


  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (const unsigned int degree)
    :
    degree (degree),
    n_q_points_1d(degree+2),
    fe (FE_RaviartThomas<dim>(degree),1,FE_DGQ<dim>(degree),1),
    dof_handler (triangulation),
    //For MF
    fe_u(FE_RaviartThomas<dim>(degree)),
    fe_p(FE_DGQ<dim>(degree)),
    dof_handler_u (triangulation),
    dof_handler_p (triangulation)
  {}


  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs ()
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

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
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


  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system ()
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

    const RightHandSide<dim>          right_hand_side;
    const PressureBoundaryValues<dim> pressure_boundary_values;
    const KInverse<dim>               k_inverse;

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

#if 0
        for (unsigned int face_n=0;
             face_n<GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n))
            {
              fe_face_values.reinit (cell, face_n);

              pressure_boundary_values
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);

              for (unsigned int q=0; q<n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                 		local_rhs(i) += -(fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));
            }
#endif

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


  template <int dim>
  SolverControl::State
  MixedLaplaceProblem<dim>::write_intermediate_solution (const unsigned int    iteration,
                                      const double check_value,
                                      const BlockVector<double> &current_iterate) const
    {
	  logfile<<"ITERATION "<<iteration<<"("<<check_value<<") = ";
	  for (int i=0; i<2; i++)
		  for (int j=0; j<current_iterate.block(i).size(); j++)
	  		  logfile << current_iterate.block(i)(j)<<"  ";
	  logfile<<std::endl;

	  g_stop++;

#if 0
	  if (g_stop == 100)
	  {
		  logfile.close();
		  return SolverControl::failure; //Just to break out
	  }
#endif

#if 0
      DataOut<2> data_out;
      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (current_iterate, "solution");
      data_out.build_patches ();
      std::ofstream output ((std::string("solution-")
                             + Utilities::int_to_string(iteration,4) + ".vtu").c_str());
      data_out.write_vtu (output);
#endif
      return SolverControl::success;
    }

  template <int dim>
  void MixedLaplaceProblem<dim>::solve_minres ()
  {
      SolverControl solver_control (500 /*solution.block(1).size()*/,
                                    1e-12);
      SolverMinRes<BlockVector<double>> mres (solver_control);

      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;
      auto conn = mres.connect (std::bind (&MixedLaplaceProblem::write_intermediate_solution,
                                       this,_1,_2,_3));

      logfile<<"Intermediate results with traditional approach"<<std::endl;
      mres.solve(system_matrix,solution,system_rhs,PreconditionIdentity());


      logfile<<std::endl<<std::endl<<"Intermediate results with MatrixFree approach"<<std::endl;
      typedef  BlockVector<double> VectorType;
      MF_MixedLaplaceProblem<dim,1,VectorType> mf (mf_data);
      mf.rhsvmult(mf_rhs, BlockVector<double>());
      mres.solve(mf,mf_solution,mf_rhs,PreconditionIdentity());


      logfile.close();
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors () const
  {
	        // Verification
	        double error = 0.;
	        for (unsigned int i=0; i<2; ++i)
	          for (unsigned int j=0; j<solution.block(i).size(); ++j)
	                  error += std::fabs (solution.block(i)(j)-mf_solution.block(i)(j));

	        double relative = solution.block(0).l1_norm();

	        std::cout<<"Error = "<<error<<" solution L1 norm = "<<relative<<std::endl;

	        std::cout << "  Verification fe degree " << degree  <<  ": "
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


  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
#if 0
    std::vector<std::string> solution_names(dim, "u");
    solution_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);

    data_out.build_patches (degree+1);

    std::ofstream output ("step-20-solution.vtu");
    data_out.write_vtu (output);
    
    std::ofstream logfile ("step-20-solution.log");
    for (const auto &elem : solution)
    {
    	logfile << elem<<std::endl;    	
    }
    
    logfile.close();
#endif

  }

  template <int dim>
  void MixedLaplaceProblem<dim>::test_assembly ()
  {
	  //vmult using dealii
	  system_matrix.vmult (solution, system_rhs);

	  //vmult using MF
      typedef  BlockVector<double> VectorType;

      if (degree != 1)
      {
    	  std::cout<<"Matrix free Test not implemented"<<std::endl;
    	  return;
      }

      MF_MixedLaplaceProblem<dim,1,VectorType> mf (mf_data);

      mf.vmult(mf_solution, system_rhs);

//#if 0
      //if (error > 10e-6)
      //{
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
      //}
//#endif

      // Verification
      double error = 0.;
      for (unsigned int i=0; i<2; ++i)
        for (unsigned int j=0; j<solution.block(i).size(); ++j)
                error += std::fabs (solution.block(i)(j)-mf_solution.block(i)(j));

      double relative = solution.block(0).l1_norm();

      std::cout<<"Error = "<<error<<" solution L1 norm = "<<relative<<std::endl;

      std::cout << "  Verification fe degree " << degree  <<  ": "
                  << error/relative << std::endl << std::endl;

  }

  template <int dim>
  void MixedLaplaceProblem<dim>::test_rhs_assembly ()
  {
	  //system_rhs is already available

	  typedef  BlockVector<double> VectorType;
      if (degree != 1)
      {
    	  std::cout<<"Matrix free Test not implemented"<<std::endl;
    	  return;
      }

	  MF_MixedLaplaceProblem<dim,1,VectorType> mf (mf_data);

      mf.rhsvmult(mf_rhs, BlockVector<double>());


#if 0
      //if (error > 10e-6)
      //{
    	  std::cout<<"RHS vector using dealii is "<<std::endl;
    	  for (unsigned int i=0; i<2; ++i)
    	  {
    		  for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
    		  {
    			  if (fabs(system_rhs.block(i)(j)) < 10e-6)
    				  std::cout<<std::setw(10)<<0;
    			  else
    				  std::cout<<std::setw(10)<<system_rhs.block(i)(j);
    		  }
    		  std::cout<<std::endl;
    	  }

    	  std::cout<<"RHS vector using MF is "<<std::endl;
    	  for (unsigned int i=0; i<2; ++i)
    	  {
    		  for (unsigned int j=0; j<mf_rhs.block(i).size(); ++j)
    		  {
    			  if (fabs(mf_rhs.block(i)(j)) < 10e-6)
    				  std::cout<<std::setw(10)<<0;
    			  else
    				  std::cout<<std::setw(10)<<mf_rhs.block(i)(j);
    		  }
    		  std::cout<<std::endl;
    	  }
      //}
#endif

      // Verification
      double error = 0.;
      for (unsigned int i=0; i<2; ++i)
        for (unsigned int j=0; j<system_rhs.block(i).size(); ++j)
                error += std::fabs (system_rhs.block(i)(j)-mf_rhs.block(i)(j));

      double relative = system_rhs.block(1).l1_norm();

      std::cout<<"Error = "<<error<<" RHS L1 norm = "<<relative<<std::endl;

      std::cout << "  Verification fe degree " << degree  <<  ": "
                  << error/relative << std::endl << std::endl;

  }


  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {
	Timer time;

	std::cout<<std::endl<<"Making grid and distributing DoFs, ";
	time.restart();
    make_grid_and_dofs();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Assembling the System Matrix and RHS, ";
    time.restart();
    assemble_system ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    //std::cout<<std::endl<<"Testing the Matrix assembly and Matrix Free assembly"<< std::endl;
    //test_assembly ();

    //std::cout<<std::endl<<"Testing the RHS assembly and Matrix Free RHS assembly"<< std::endl;
    //test_rhs_assembly ();

//#if 0 //To be opened later
    std::cout<<std::endl<<"Solving the Linear system, ";
    time.restart();
    //solve ();
    solve_minres();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Computing the errors, ";
    time.restart();
    compute_errors ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Printing the results, ";
    time.restart();
    output_results ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
//#endif
  }
}


int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      //Observations on dealii version of code:
      //degree 1 gives bad results. deg0,2, and onwards give good results
      constexpr int dim = 2;
      constexpr int degree = 1;

      //MixedLaplaceProblem<2> mixed_laplace_problem(0);
      MixedLaplaceProblem<dim> mixed_laplace_problem(degree);
      mixed_laplace_problem.run ();
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
