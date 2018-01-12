//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for FE_Q and RT elements, for all components
//			 - compare values, gradients and hessians
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

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d=fe_degree+1, int base_fe_degree=fe_degree>
class Test{

	static const int n_array_elements = VectorizedArray<Number>::n_array_elements;

    const bool evaluate_values, evaluate_gradients, evaluate_hessians;

    debugStream mylog;
    UpdateFlags          update_flags;

    //std::vector<Number> fe_unit_values;
    //std::vector<Number> shape_values_old;

	bool compare_values(int);
	bool compare_gradients(int);
	bool compare_hessians(int);
public:
	Test() = delete;
	Test(bool evaluate_values, bool evaluate_gradients, bool evaluate_hessians);
	bool run(bool debug);
};

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::Test(
		bool evaluate_values, bool evaluate_gradients, bool evaluate_hessians) :
		evaluate_values(evaluate_values), evaluate_gradients(evaluate_gradients),
		evaluate_hessians(evaluate_hessians)
{
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

	FE_Q<dim> fe_u(fe_degree);
	FESystem<dim>  fe (fe_u, n_components);
	FE_RaviartThomas<dim> fe_rt(fe_degree);
	QGauss<dim> fe_quad(n_q_points_1d);
	QGauss<1>   quad_1d(n_q_points_1d);

	const unsigned int first_selected_component = 0;
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl(quad_1d, fe, fe.component_to_base_index(first_selected_component).first);
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl(quad_1d, fe, fe.component_to_base_index(first_selected_component).first,true);

    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_rt_impl(quad_1d, fe_rt, fe_rt.component_to_base_index(first_selected_component).first,true);

	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = fe_quad.size();
	//fe_unit_values.resize((dofs_per_cell*n_q_points)/n_components);
	//shape_values_old.resize((dofs_per_cell*n_q_points)/n_components);


	Point<dim> p;


	const unsigned int   rt_dofs_per_cell = fe_rt.dofs_per_cell;

	mylog<<"Using FiniteElement functions "<<std::endl;
	for (unsigned int i=0; i<rt_dofs_per_cell; ++i)
	{
	  for (unsigned int q=0; q<1/*n_q_points*/; ++q)
	  {
		  p = fe_quad.point(q);
		  //std::cout<<"Point is ("<<p[0]<<", "<<p[1]<<")"<<std::endl;
		  for (int c=0; c<n_components; c++)
		  {
			  mylog<<"(c,i,q) = ("<<c<<","<<i<<","<<q<<") and value = "<<fe_rt.shape_value_component(i,p,c)<<std::endl;
		  }
	  }
	}

	mylog<<"using New ShapeInfo functions "<<std::endl;
	for (unsigned int i=0; i<rt_dofs_per_cell; ++i)
	{
		for (unsigned int q=0; q<1; ++q)
		{
			mylog<<"(i,q) = ("<<i<<","<<q<<") and value = "<<
					shape_info_rt_impl.shape_values_vec[c][0][i*n_q_points_1d+q][0]<<std::endl;
		}
	}

#if 0
	if (evaluate_values)
	{
		mylog<<"output for evaluate_values on unit cell"<<std::endl;

			//mylog<<"Result for component ("<<c<<")"<<std::endl;

			mylog<<"Using FiniteElement functions "<<std::endl;
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  for (unsigned int q=0; q<1/*n_q_points*/; ++q)
	    	  {
				  p = fe_quad.point(q);
				  //std::cout<<"Point is ("<<p[0]<<", "<<p[1]<<")"<<std::endl;
				  for (int c=0; c<n_components; c++)
				  {
					  mylog<<"(c,i,q) = ("<<c<<","<<i<<","<<q<<") and value = "<<fe.shape_value_component(i,p,c)<<std::endl;
				  }
	    	  }
			}

			mylog<<"using Old ShapeInfo functions "<<std::endl;
			unsigned int shape_info_size = shape_info_old_impl.shape_values.size();
			unsigned int n_dof_1d = shape_info_size/n_q_points_1d;
			for (unsigned int i=0; i<n_dof_1d; ++i)
			{
				for (unsigned int q=0; q<n_q_points_1d; ++q)
				{
					mylog<<"(i,q) = ("<<i<<","<<q<<") and value = "<<
								shape_info_old_impl.shape_values[i*n_q_points_1d+q][0]<<std::endl;
				}
			}

			mylog<<"using New ShapeInfo functions "<<std::endl;
			for (unsigned int i=0; i<n_dof_1d; ++i)
			{
				for (unsigned int q=0; q<n_q_points_1d; ++q)
				{
					mylog<<"(i,q) = ("<<i<<","<<q<<") and value = "<<
								shape_info_new_impl.shape_values_vec[c][0][i*n_q_points_1d+q][0]<<std::endl;
				}
			}
		}
#endif
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


	//note: This test only suppotrs n_components = dim since thats how FEEvaluationGen is designed

	//n_comp, dim, fe_deg, q_1d, base_degree


    res = Test<double,2,2,1>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

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
