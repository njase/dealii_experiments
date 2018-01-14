//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for FE_Q elements, for all components
//			 - compare values, gradients and hessians
//update: So this experiment is currently stopped here, continue with RT in expr6 instead
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
	QGauss<dim> fe_quad(n_q_points_1d);
	QGauss<1>   quad_1d(n_q_points_1d);

	const unsigned int first_selected_component = 0;
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl(quad_1d, fe, fe.component_to_base_index(first_selected_component).first);
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl(quad_1d, fe, fe.component_to_base_index(first_selected_component).first,true);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = fe_quad.size();
	//fe_unit_values.resize((dofs_per_cell*n_q_points)/n_components);
	//shape_values_old.resize((dofs_per_cell*n_q_points)/n_components);


	Point<dim> p;

	//Remark: It should be possible to compare ShapeInfo evaluations against FEValues for FE_Q elements
	//by following the below algo. I am not going to implement this now as it seems unnecessary because
	// expr4 UT case is already working
	//Focus instead on Raviart Thomas element using this knowledge

	if (evaluate_values)
	{
		//Steps for primitive FEs
		//1. Pre-calculate a mapping from (global basis function number) to (local=component wise basis function number)
		//   using fe.system_to_component_index()
		//2. Evaluate global shape function values/gradients/hessians for each component // alternatively find out non-zero indexes and
		//   only calculate there
		//   using fe.shape_value_component()
		//3. Store the calculated result component wise in order of local basis function number

		//For comparison ==>
		//1. Find out relation of local basis numbering to lexicographic numbering
		//   using get_poly_space_numbering_inverse()
		// 2. Evaluate tensor product of ShapeInfo results and store them lexicographically, each component has same results.
		//   Put results component wise one after the other
		//   Tensor product has to be calculated ourself (no existing function)
		// 3. compare the ShapeInfo results which are lexicographiclaly ordered and put component wise with the previously
		//  evaluated results

#if 0

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


    res = Test<double,2,2,2>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

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
