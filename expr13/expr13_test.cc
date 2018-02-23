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



namespace Step20
{
  using namespace dealii;

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

        	  Tensor<2,dim,vector_t> grad_u = velocity.get_gradient (q);
        	  vector_t pres = pressure.get_value(q);
        	  vector_t div = -trace(grad_u);
        	  pressure.submit_value (div, q);

              for (unsigned int d=0; d<dim; ++d)
            	  u[d] -= pres;

              velocity.submit_value (u, q);
          }

          velocity.integrate (true,false);
          velocity.distribute_local_to_global (dst.block(0));
          pressure.integrate (true,false);
          pressure.distribute_local_to_global (dst.block(1));

#if 0
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
#endif
        }

      std::cout<<"Loop was internally run from "<<cell_range.first<<" to "<<cell_range.second<<std::endl;

  }

    void vmult (VectorType &dst,
                const VectorType &src) const
    {
  	  data.cell_loop (&MF_MixedLaplaceProblem<dim,degree_p,VectorType>::local_apply_vector,
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
    void solve ();
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

    void test_assembly ();
  };


  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &/*p*/,
                                    const unsigned int /*component*/) const
  {
    return 0;
  }



  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
    const double alpha = 0.3;
    const double beta = 1;
    return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
  }



  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

    const double alpha = 0.3;
    const double beta = 1;

    values(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
    values(1) = alpha*p[0]*p[1];
    values(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
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
    //We can only work cartesian mesh cells in MF version - so let it be as-it-is
	GridGenerator::hyper_cube (triangulation, -1, 1);
	//First, lets test on one cell only
#if 0
    triangulation.refine_global (3);
#endif

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

    	std::vector<const DoFHandler<dim>*> dofs;
        dofs.push_back(&dof_handler_u);
        dofs.push_back(&dof_handler_p);
        ConstraintMatrix dummy_constraints;
        dummy_constraints.close();
        std::vector<const ConstraintMatrix *> constraints;
        constraints.push_back (&dummy_constraints);
        constraints.push_back (&dummy_constraints);
        QGauss<1> quad(n_q_points_1d);
        mf_data.reinit (dofs, constraints, quad,
                        typename MatrixFree<dim>::AdditionalData
                        (MatrixFree<dim>::AdditionalData::none));
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

              local_rhs(i) += -phi_i_p *
                              rhs_values[q] *
                              fe_values.JxW(q);
            }

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

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               local_matrix(i,j));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);
      }
  }


  template <class MatrixType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &m);

    void vmult(Vector<double>       &dst,
               const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
  };


  template <class MatrixType>
  InverseMatrix<MatrixType>::InverseMatrix (const MatrixType &m)
    :
    matrix (&m)
  {}


  template <class MatrixType>
  void InverseMatrix<MatrixType>::vmult (Vector<double>       &dst,
                                         const Vector<double> &src) const
  {
    SolverControl solver_control (std::max(src.size(), static_cast<std::size_t> (200)),
                                  1e-8*src.l2_norm());
    SolverCG<>    cg (solver_control);

    dst = 0;

    cg.solve (*matrix, dst, src, PreconditionIdentity());
  }


  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement (const BlockSparseMatrix<double>            &A,
                     const InverseMatrix<SparseMatrix<double> > &Minv);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const InverseMatrix<SparseMatrix<double> > > m_inverse;

    mutable Vector<double> tmp1, tmp2;
  };


  SchurComplement
  ::SchurComplement (const BlockSparseMatrix<double>            &A,
                     const InverseMatrix<SparseMatrix<double> > &Minv)
    :
    system_matrix (&A),
    m_inverse (&Minv),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void SchurComplement::vmult (Vector<double>       &dst,
                               const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    m_inverse->vmult (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }

  class ApproximateSchurComplement : public Subscriptor
  {
  public:
    ApproximateSchurComplement (const BlockSparseMatrix<double> &A);

    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

    mutable Vector<double> tmp1, tmp2;
  };



  ApproximateSchurComplement::ApproximateSchurComplement
  (const BlockSparseMatrix<double> &A) :
    system_matrix (&A),
    tmp1 (A.block(0,0).m()),
    tmp2 (A.block(0,0).m())
  {}


  void
  ApproximateSchurComplement::vmult
  (Vector<double>       &dst,
   const Vector<double> &src) const
  {
    system_matrix->block(0,1).vmult (tmp1, src);
    system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
    InverseMatrix<SparseMatrix<double> > inverse_mass (system_matrix.block(0,0));
    Vector<double> tmp (solution.block(0).size());

    // Now on to the first equation. The right hand side of it is $B^TM^{-1}F-G$,
    // which is what we compute in the first few lines:
    {
      SchurComplement schur_complement (system_matrix, inverse_mass);
      Vector<double> schur_rhs (solution.block(1).size());
      inverse_mass.vmult (tmp, system_rhs.block(0));
      system_matrix.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      // Now that we have the right hand side we can go ahead and solve for the
      // pressure, using our approximation of the inverse as a preconditioner:
      SolverControl solver_control (solution.block(1).size(),
                                    1e-12*schur_rhs.l2_norm());
      SolverCG<> cg (solver_control);

      ApproximateSchurComplement approximate_schur (system_matrix);
      InverseMatrix<ApproximateSchurComplement> approximate_inverse
      (approximate_schur);
      cg.solve (schur_complement, solution.block(1), schur_rhs,
                approximate_inverse);

      std::cout << solver_control.last_step()
                << " CG Schur complement iterations to obtain convergence."
                << std::endl;
    }

    // After we have the pressure, we can compute the velocity. The equation
    // reads $MU=-BP+F$, and we solve it by first computing the right hand
    // side, and then multiplying it with the object that represents the
    // inverse of the mass matrix:
    {
      system_matrix.block(0,1).vmult (tmp, solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      inverse_mass.vmult (solution.block(0), tmp);
    }
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors () const
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);

    ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, n_q_points_1d);

    // With this, we can then let the library compute the errors and output
    // them to the screen:
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error
              << std::endl;
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
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

    std::cout<<std::endl<<"Testing the Matrix assembly and Matrix Free assembly";
    test_assembly ();


#if 0 //To be opened later
    std::cout<<std::endl<<"Solving the Linear system, ";
    time.restart();
    solve ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Computing the errors, ";
    time.restart();
    compute_errors ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;

    std::cout<<std::endl<<"Printing the results, ";
    time.restart();
    output_results ();
    std::cout<<"Time taken CPU/WALL = "<<time.cpu_time() << "s/" << time.wall_time() << "s" << std::endl;
#endif
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
