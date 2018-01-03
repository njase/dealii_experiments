//////////////////
//
// Purpose : To compare the results of vector valued FEEvaluation against FEEvaluationGen
//           and using appropriate ShapeInfo object
//			 No MatrixFree object is needed, this is low level unit test
//
//           - therefore verifies only FE_Q elements
//           - compare results with tensor_general
//			 - what about other element types??
//			 - compare results with FEValues
//
/////////////////

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

//TBD: For n_components > 1, vector-valued problem has to be constructed
//const int g_fe_degree_1c = g_fe_degree, g_fe_degree_2c = g_fe_degree, g_fe_degree_3c = g_fe_degree;

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d=fe_degree+1, int base_fe_degree=fe_degree>
bool test(bool debug)
{
	debugStream mylog;
	mylog.debug = debug;

	const bool evaluate_values = true, evaluate_gradients = false, evaluate_hessians = false;

	bool res_values = true, res_gradients = true, res_hessians = true;

	//////////////////

	VectorizedArray<Number> *values_dofs_actual[n_components]; //in
	VectorizedArray<Number> *scratch_data; //in


	//Just allocate a large memory, actual size unimportant for our test
	scratch_data = new VectorizedArray<Number> [1000];


	//out
	VectorizedArray<Number> *values_quad_old_impl[n_components],
							*values_quad_new_impl[n_components];
	VectorizedArray<Number> *gradients_quad_old_impl[n_components][dim],
							*gradients_quad_new_impl[n_components][dim];
	VectorizedArray<Number> *hessians_quad_old_impl[n_components][(dim*(dim+1))/2],
							*hessians_quad_new_impl[n_components][(dim*(dim+1))/2];


	/////For simplicity in comparison, we use arrays instead of allocating memory to above output variables
	const unsigned int first_selected_component = 0;
	FE_Q<dim> fe_u(fe_degree);
	FESystem<dim>  fe (fe_u, n_components);
	QGauss<1> quad(n_q_points_1d);

	//in
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl(quad, fe, fe.component_to_base_index(first_selected_component).first);
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl(quad, fe, fe.component_to_base_index(first_selected_component).first,true);

	//FIXME:This count will change when the dofs for each component are different
	//Give known values to values_dofs_actual, i.e. to nodal_values
	const int dofs_cnt_per_cell = fe.n_dofs_per_cell ();
	const int quad_cnt_per_cell = Utilities::fixed_int_power<n_q_points_1d,dim>::value*n_components;

	const int n_array_elements = VectorizedArray<Number>::n_array_elements;
	const int n_dof_elements = 100;
	//const int n_eval_elements = (dofs_cnt_per_cell * quad_cnt_per_cell)/static_cast<float>(n_array_elements);
	const int n_eval_elements = (dofs_cnt_per_cell)/static_cast<float>(n_array_elements);


	using namespace std;
	cout<<endl<<"======= parameters ============"<<endl<<endl;

	cout<<"components = "<<n_components<<" ,dim = "<<dim<<"  ,fe_degree = "<<fe_degree<<"  ,n_q_points_1d = "<<n_q_points_1d<<endl;
	cout<<"DOF per cell = "<<dofs_cnt_per_cell<<", Quad points per cell = "<<quad_cnt_per_cell<<endl<<"Length of one VectorArray element = "<<n_array_elements<<endl;
	cout<<"Total source DoFs used in this test = "<<n_dof_elements<<endl;
	cout<<"Number of vectorArray elements used for result = "<<n_eval_elements<<endl;

	cout<<endl<<"======= ========== ============"<<endl;

	VectorizedArray<Number> nodal_values[n_dof_elements];

	std::srand(std::time(nullptr)); // use current time as seed for random generator
	for (int d = 0; d < n_dof_elements; d++)
	{
		for (int n = 0; n < n_array_elements; n++)
		{
			nodal_values[d][n] = std::rand()/static_cast<Number>(RAND_MAX);
		}
	}

	VectorizedArray<Number> out_values_quad_old_impl[n_components][n_eval_elements],
							out_values_quad_new_impl[n_components][n_eval_elements];
	VectorizedArray<Number> out_gradients_quad_old_impl[n_components][dim][n_eval_elements],
							out_gradients_quad_new_impl[n_components][dim][n_eval_elements];
	VectorizedArray<Number> out_hessians_quad_old_impl[n_components][(dim*(dim+1))/2][n_eval_elements],
							out_hessians_quad_new_impl[n_components][(dim*(dim+1))/2][n_eval_elements];


	shape_info_old_impl.element_type = internal::MatrixFreeFunctions::tensor_general;

	for (int c = 0; c < n_components; c++)
	{
		//Same nodal values for all the components
		values_dofs_actual[c] = nodal_values;

		values_quad_old_impl[c] = out_values_quad_old_impl[c];
		values_quad_new_impl[c] = out_values_quad_new_impl[c];

		for (int d = 0; d < dim; d++)
		{
			gradients_quad_old_impl[c][d] = out_gradients_quad_old_impl[c][d];
			gradients_quad_new_impl[c][d] = out_gradients_quad_new_impl[c][d];
		}

		for (int e = 0; e < (dim*(dim+1))/2; e++)
		{
			hessians_quad_old_impl[c][e] = out_hessians_quad_old_impl[c][e];
			hessians_quad_new_impl[c][e] = out_hessians_quad_new_impl[c][e];
		}
	}


    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
    		dim, fe_degree, n_q_points_1d, n_components, Number>
             ::evaluate(shape_info_old_impl, values_dofs_actual, values_quad_old_impl,
            		 gradients_quad_old_impl, hessians_quad_old_impl, scratch_data,
                        evaluate_values, evaluate_gradients, evaluate_hessians);


	internal::FEEvaluationImplGen<internal::MatrixFreeFunctions::tensor_general,
				FE_TaylorHood, n_q_points_1d, dim, base_fe_degree, Number>
           ::evaluate(shape_info_new_impl, values_dofs_actual, values_quad_new_impl,
        		   gradients_quad_new_impl, hessians_quad_new_impl, scratch_data,
                      evaluate_values, evaluate_gradients, evaluate_hessians);


	std::cout<<"Computation finished...printing results"<<std::endl;

	bool res = false;

	if (evaluate_values == true) {
	//Compare output of both and tell result
	mylog<<"Result for values=============="<<std::endl;

	for (int c = 0; c < n_components; c++)
	{
		mylog<<"============================================"<<std::endl;
		mylog<<"Comparison result for component ("<<c<<")"<<std::endl;
		mylog<<"============================================"<<std::endl;
		for (int e = 0; e < n_eval_elements; e++)
		{
			for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
			{
				res = (std::abs(out_values_quad_old_impl[c][e][n] - out_values_quad_new_impl[c][e][n]) < 10e-4)?true:false;
				mylog<<"(old,new) = ("<<out_values_quad_old_impl[c][e][n]<<", "<<out_values_quad_new_impl[c][e][n]<<")   AND  result = "<<res<<std::endl;
				if (false == res)
					res_values = false;
			}
		}
	}
	}

	if (evaluate_gradients == true) {
	mylog<<"Result for gradients=============="<<std::endl;

	for (int c = 0; c < n_components; c++)
	{
		mylog<<"============================================"<<std::endl;
		mylog<<"Comparison result for component ("<<c<<")"<<std::endl;
		mylog<<"============================================"<<std::endl;
		for (int d = 0; d < dim; d++)
		{
			for (int e = 0; e < n_eval_elements; e++)
			{
				for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
				{
					res = (std::abs(out_gradients_quad_old_impl[c][d][e][n] - out_gradients_quad_new_impl[c][d][e][n]) < 10e-4)?true:false;
					mylog<<"(old,new) = ("<<out_gradients_quad_old_impl[c][d][e][n]<<", "<<out_gradients_quad_new_impl[c][d][e][n]<<")   AND  result = "<<res<<std::endl;
					if (false == res)
							res_gradients = false;
				}
			}
		}
	}
	}

	if (evaluate_hessians == true) {
	mylog<<"Result for hessians=============="<<std::endl;

	for (int c = 0; c < n_components; c++)
	{
		mylog<<"============================================"<<std::endl;
		mylog<<"Comparison result for component ("<<c<<")"<<std::endl;
		mylog<<"============================================"<<std::endl;
		for (int d = 0; d < (dim*(dim+1))/2; d++)
		{
			for (int e = 0; e < n_eval_elements; e++)
			{
				for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
				{
					res = (std::abs(out_hessians_quad_old_impl[c][d][e][n] - out_hessians_quad_new_impl[c][d][e][n]) < 10e-4)?true:false;
					mylog<<"(old,new) = ("<<out_hessians_quad_old_impl[c][d][e][n]<<", "<<out_hessians_quad_new_impl[c][d][e][n]<<")   AND  result = "<<res<<std::endl;
					if (false == res)
							res_hessians = false;
				}
			}
		}
	}
	}

	std::cout<<"============================="<<std::endl;
	std::cout<<"         Overall result       "<<std::endl;
	std::cout<<"============================="<<std::endl;
	if (evaluate_values)
		std::cout<<" Function values = "<<(res_values == true?"pass":"fail")<<std::endl;
	else
		std::cout<<" Function values = "<<"Not evaluated"<<std::endl;

	if (evaluate_gradients)
		std::cout<<" Function gradients = "<<(res_gradients == true?"pass":"fail")<<std::endl;
	else
		std::cout<<" Function gradients = "<<"Not evaluated"<<std::endl;

	if (evaluate_hessians)
		std::cout<<" Function hessians = "<<(res_hessians == true?"pass":"fail")<<std::endl;
	else
		std::cout<<" Function hessians = "<<"Not evaluated"<<std::endl;


	delete[] scratch_data;

	return true;
}


int main(int argc, char *argv[])
{
	bool res, debug = false;

	if ( argc > 2 )
	{
		std::cout<<"Warning : too many input arguments - ignored"<<std::endl;
	    // We print argv[0] assuming it is the program name
	    cout<<"usage: "<< argv[0] <<" <filename>\n";
	}

	if (argc > 1 && std::string(argv[1]) == "--debug")
		debug = true;


	//n_comp, dim, fe_deg, q_1d, base_degree

#if 0 //TBD: failures but analyze later
	//1 comp, 1 dim, 1 degree
	res = test<double,1,1,1>(debug);

	//1 comp, 1 dim, 2 degrees
	res = test<double,1,1,2>(debug);

	//1 comp, 1 dim, 3 degrees
	res = test<double,1,1,3>(debug);

	//1 comp, 2 dim, 1 degree
	res = test<double,1,2,1>(debug);

	//1 comp, 2 dim, 2 degrees
	res = test<double,1,2,2>(debug);

	//1 comp, 2 dim, 3 degrees
	res = test<double,1,2,3>(debug);

	//1 comp, 3 dim, 1 degree
	res = test<double,1,3,1>(debug);

	//1 comp, 3 dim, 2 degrees
	res = test<double,1,3,2>(debug);

	//1 comp, 3 dim, 3 degree
	res = test<double,1,3,3>(debug);
#endif


	//2 comp, 2 dim, 1 degree -> Pass
	//res = test<double,2,2,1>(debug);

	//2 comp, 2 dim, 2 degrees -> Fail
	res = test<double,2,2,2>(debug);

	//2 comp, 2 dim, 3 degrees -> Fail
	//res = test<double,2,2,3>(debug);


#if 0
	//2 comp, 3 dim, 1 degree
	res = test<double,2,3,1>(debug);

	//2 comp, 3 dim, 2 degrees
	res = test<double,2,3,2>(debug);

	//2 comp, 3 dim, 3 degrees
	res = test<double,2,3,3>(debug);


	//3 comp, 2 dim, 1 degree
	res = test<double,3,2,1>(debug);

	//3 comp, 2 dim, 2 degrees
	res = test<double,3,2,2>(debug);

	//3 comp, 2 dim, 3 degrees
	res = test<double,3,2,3>(debug);


	//3 comp, 3 dim, 1 degree
	res = test<double,3,3,1>(debug);

	//3 comp, 3 dim, 2 degrees
	res = test<double,3,3,2>(debug);

	//3 comp, 3 dim, 3 degrees
	res = test<double,3,3,3>(debug);

#endif

	//1 comp, 2 dim
	//res = test<double,2,2,1,2>(debug);
	//res = test<double,2,2,2,3>(debug); //fails
	//res = test<double,3,2,1,2>(); //Fails for new

	return 0;
}
