//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for FE_Q and RT elements, for all components

#if 0
//To compare the results of vector valued FEEvaluationGen
//           against FEValues for FE_Q and RT elements and using appropriate ShapeInfo object
//			 No MatrixFree object is needed, this is low level unit test
//
//			 First check for FEEvaluation and then replace with FEEvaluationGen
//			 Compare for values, gradients, hesians, integration - on both unit cell
//			 and real cell
#endif
/////////////////

#include "tests.h"
#include "create_mesh.h"
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

#include <iostream>
#include <complex>
#include <vector>

using namespace dealii;
using namespace dealii::internal;
using namespace std;

class debugStream{

public:
    // the type of std::cout
    typedef std::basic_ostream<char, std::char_traits<char> > CoutType;
    // function signature of std::endl
    typedef CoutType& (*StandardEndLine)(CoutType&);

	bool debug = true;
	bool lineend = false;

	template <typename T>
	debugStream& operator<<( T const& obj )
    {
		if (debug == true)
		{
			std::cout << obj;
			if (lineend)
				std::cout<<std::endl;
		}
        return *this;
    }

    // overload << to accept in std::endl
	debugStream& operator<<(StandardEndLine manip)
    {
		if (debug == true)
			manip(std::cout);
        return *this;
    }
};

void test_mesh (Triangulation<2> &tria,
                  const double scale_grid = 1.)
{
  const unsigned int dim = 2;

#if 0
  std::vector<Point<dim> > points (12);

  // build the mesh layer by layer from points

  // 1. cube cell
  points[0] = Point<dim> (0, 0);
  points[1] = Point<dim> (0, 1);
  points[2] = Point<dim> (1,0);
  points[3] = Point<dim> (1,1);

  // 2. rectangular cell
  points[4] = Point<dim> (3., 0);
  points[5] = Point<dim> (3., 1);

  // 3. parallelogram cell
  points[6] = Point<dim> (5., 1.);
  points[7] = Point<dim> (5., 2.);

  // almost square cell (but trapezoidal by
  // 1e-8)
  points[8] = Point<dim> (6., 1.);
  points[9] = Point<dim> (6., 2.+1e-8);

  // apparently trapezoidal cell
  points[10] = Point<dim> (7., 1.4);
  points[11] = Point<dim> (7.5, numbers::PI);

  if (scale_grid != 1.)
    for (unsigned int i=0; i<points.size(); ++i)
      points[i] *= scale_grid;


  // connect the points to cells
  std::vector<CellData<dim> > cells(5);
  for (unsigned int i=0; i<5; ++i)
    {
      cells[i].vertices[0] = 0+2*i;
      cells[i].vertices[1] = 2+2*i;
      cells[i].vertices[2] = 1+2*i;
      cells[i].vertices[3] = 3+2*i;
      cells[i].material_id = 0;
    }
#endif

  std::vector<Point<dim> > points (4);

  // almost square cell (but trapezoidal by
  // 1e-8)
  points[0] = Point<dim> (6., 1.);
  points[1] = Point<dim> (6., 2.+1e-8);

  // apparently trapezoidal cell
  points[2] = Point<dim> (7., 1.4);
  points[3] = Point<dim> (7.5, numbers::PI);


  std::vector<CellData<dim> > cells(1);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 2;
  cells[0].vertices[2] = 1;
  cells[0].vertices[3] = 3;

  tria.create_triangulation (points, cells, SubCellData());
}

//TBD: For n_components > 1, vector-valued problem has to be constructed
//const int g_fe_degree_1c = g_fe_degree, g_fe_degree_2c = g_fe_degree, g_fe_degree_3c = g_fe_degree;

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d=fe_degree+1, int base_fe_degree=fe_degree>
class Test{
    const bool evaluate_values, evaluate_gradients, evaluate_hessians;
    const bool integrate_values, integrate_gradients;

    debugStream mylog;
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    UpdateFlags          update_flags;

	bool compare_values(int);
	bool compare_gradients(int);
	bool compare_hessians(int);
public:
	Test() = delete;
	Test(bool evaluate_values, bool evaluate_gradients, bool evaluate_hessians,
		bool integrate_values, bool integrate_gradients);
	bool run(bool debug);
};

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::Test(
		bool evaluate_values, bool evaluate_gradients, bool evaluate_hessians,
		bool integrate_values, bool integrate_gradients) :
		evaluate_values(evaluate_values), evaluate_gradients(evaluate_gradients),
		evaluate_hessians(evaluate_hessians), integrate_values(integrate_values),
		integrate_gradients(integrate_gradients)
{
	if (integrate_values == true && evaluate_values == false)
	{
		std::cout<<"Error: Must evaluate values before integrating them "<<std::endl;
		return;
	}

	if (integrate_gradients == true && evaluate_gradients == false)
	{
		std::cout<<"Error: Must evaluate gradients before integrating them "<<std::endl;
		return;
	}

	//create_mesh (triangulation);
	test_mesh(triangulation);
	//GridGenerator::hyper_cube (triangulation, -1, 1);
	//  const Point<2> center (1,0);
	//  const double inner_radius = 0.5, outer_radius = 1.0;
	//GridGenerator::hyper_shell (triangulation,
	//                           center, inner_radius, outer_radius,10);

	update_flags = update_JxW_values;

	if (evaluate_values)
		update_flags |= update_values;

	if (evaluate_gradients)
		update_flags |= update_gradients;

	if (evaluate_hessians)
		update_flags |= update_hessians;
}

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::compare_values(int id)
{
	bool res_values = true;

	return res_values;
}



template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::compare_gradients(int id)
{
	bool res_gradients = true;

	return res_gradients;
}


template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::compare_hessians(int id)
{
	bool res_hessians = true;

	return res_hessians;

}

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::run(bool debug)
{
	mylog.debug = debug;


    std::cout<<"No of active cells from triangulation = "<<triangulation.n_active_cells()<<std::endl;

	FE_Q<dim> fe_u(fe_degree);
	FESystem<dim>  fe (fe_u, n_components);
	QGauss<dim> fe_quad(n_q_points_1d);
	FEValues<dim> fe_values (fe, fe_quad, update_flags);

	dof_handler.initialize(triangulation, fe);
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = fe_quad.size();
	Point<dim> p;

#if 0
	if (fe_degree == 1)
	    triangulation.refine_global (4-dim);
	else
	    triangulation.refine_global (3-dim);
#endif

	typename DoFHandler<dim>::active_cell_iterator cell, endc = dof_handler.end();

	mylog<<"output for Jacobians on real cell"<<std::endl;

	cell = dof_handler.begin_active();
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);

		for (unsigned int q=0; q<n_q_points; ++q)
		{
			mylog<<"(q) = ("<<","<<q<<") and value = "<<fe_values.JxW(q)<<std::endl;
		}
	}

	if (evaluate_values)
	{
		mylog<<"output for evaluate_values on real cell"<<std::endl;

		cell = dof_handler.begin_active();
		for (; cell!=endc; ++cell)
		{
			fe_values.reinit (cell);

			for (int c=0; c<n_components; c++)
			{
				mylog<<"Result for component ("<<c<<")"<<std::endl;

				for (unsigned int q=0; q<n_q_points; ++q)
				{
	    		  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    		  {
	    			  mylog<<"(i,q) = ("<<i<<","<<q<<") and value = "<<fe_values.shape_value_component(i,q,c)<<std::endl;
	    		  }
				}
			}
		}
#if 0
		mylog<<"output for evaluate_values on unit cell"<<std::endl;

		for (int c=0; c<n_components; c++)
		{
			mylog<<"Result for component ("<<c<<")"<<std::endl;

			for (unsigned int q=0; q<n_q_points; ++q)
			{
			  p = fe_quad.point(q);
			  std::cout<<"Point is ("<<p[0]<<", "<<p[1]<<")"<<std::endl;
	    	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    	  {
	    		  mylog<<"(i,q) = ("<<i<<","<<q<<") and value = "<<fe.shape_value_component(i,p,c)<<std::endl;
	    	  }
			}
		}
#endif
	}

	if (evaluate_gradients)
	{
		mylog<<"output for evaluate_gradients"<<std::endl;

		cell = dof_handler.begin_active();
		for (; cell!=endc; ++cell)
		{
			fe_values.reinit (cell);

			for (int c=0; c<n_components; c++)
			{
				mylog<<"Result for component ("<<c<<")"<<std::endl;

				for (unsigned int q=0; q<n_q_points; ++q)
	    		  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    		  {
	    			  mylog<<"(i,q) = ("<<i<<","<<q<<") and gradient = "<<fe_values.shape_grad_component(i,q,c)<<std::endl;
	    		  }
			}
		}
	}

	if (evaluate_hessians)
	{
		mylog<<"output for evaluate_hessians"<<std::endl;

		cell = dof_handler.begin_active();
		for (; cell!=endc; ++cell)
		{
			fe_values.reinit (cell);

			for (int c=0; c<n_components; c++)
			{
				mylog<<"Result for component ("<<c<<")"<<std::endl;

				for (unsigned int q=0; q<n_q_points; ++q)
	    		  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    		  {
	    			  mylog<<"(i,q) = ("<<i<<","<<q<<") and hessian = "<<fe_values.shape_hessian_component(i,q,c)<<std::endl;
	    		  }
			}
		}
	}

	return true;
}


int main(int argc, char *argv[])
{
	bool res, debug = false;

	if ( argc > 2 )
	{
		std::cout<<"Warning : too many input arguments - ignored"<<std::endl;
	    cout<<"usage: "<< argv[0] <<" <filename>\n";
	}

	if (argc > 1 && std::string(argv[1]) == "--debug")
		debug = true;

	const bool evaluate_values = true;
	const bool evaluate_gradients = false;
	const bool evaluate_hessians = false;

	const bool integrate_values = false;
	const bool integrate_gradients = false;


	//note: This test only suppotrs n_components = dim since thats how FEEvaluationGen is designed

	//n_comp, dim, fe_deg, q_1d, base_degree


    res = Test<double,1,2,1>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    //2-D tests
#if 0
    res = Test<double,2,2,1>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    res = Test<double,2,2,2>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    res = Test<double,2,2,3>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    //3-D tests
    res = Test<double,3,3,1>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    res = Test<double,3,3,2>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);

    res = Test<double,3,3,3>(evaluate_values,evaluate_gradients,evaluate_hessians,
			integrate_values,integrate_gradients).run(debug);
#endif

	return 0;
}
