//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for FE_Q elements, for all components
//			 - compare values, gradients and hessians
//update: So this experiment is currently stopped here, continue with RT in expr6 instead
//however, i've added a small sample to evaluate values using fe_poly and using tensor product from
// shape info. The results can be manually verified using ./expr5_test --debug and are identical

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

	//FE_Q<dim> fe_u(fe_degree);
	//FESystem<dim>  fe (fe_u, n_components);
	FE_Q<dim> fe(fe_degree); //For debugging
	QGauss<dim> fe_quad(n_q_points_1d);
	QGauss<1>   quad_1d(n_q_points_1d);

	const unsigned int first_selected_component = 0;
	MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old(quad_1d, fe, fe.component_to_base_index(first_selected_component).first);
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl(quad_1d, fe, fe.component_to_base_index(first_selected_component).first,true);

    const unsigned int   fe_dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = fe_quad.size();


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


		const FE_Poly<TensorProductPolynomials<dim>,dim,dim> *fe_poly =
				dynamic_cast<const FE_Poly<TensorProductPolynomials<dim>,dim,dim>*>(&fe);

		if (fe_dofs_per_cell != fe_poly->poly_space.n())
		{
			std::cout<<"Conceptual bug..Error " <<std::endl;
			return false;
		}

		std::vector<std::vector<double>> values(n_q_points);
		std::vector<Tensor<1,dim>> unused2_1;
		std::vector<Tensor<2,dim>> unused3_1;
		std::vector<Tensor<3,dim>> unused4_1;
		std::vector<Tensor<4,dim>> unused5_1;

		//Evaluate each basis function on every quad point
		for (unsigned int q=0; q<n_q_points; ++q)
		{
			//size = no. of dofs = no of tensor product polynomials = total no of basis functions
			values[q].resize(fe_dofs_per_cell);
			p = fe_quad.point(q);
			fe_poly->poly_space.compute(p,values[q],unused2_1,unused3_1,unused4_1,unused5_1);
		}

		mylog<<"Using FEPoly functions "<<std::endl;
		std::vector<unsigned int> numbering = fe_poly->get_poly_space_numbering_inverse();

		//for (int c=0; c<n_components; c++)
		//{
		//	mylog<<"For component number ==========="<<c<<std::endl;
			for (unsigned int i=0; i<fe_dofs_per_cell; ++i)
			{
				mylog<<std::setw(10)<<i;
			  for (unsigned int q=0; q<n_q_points; ++q)
		  	  {
			  	  mylog<<std::setw(20)<<values[q][numbering[i]];
		  	  }
		  	  mylog<<std::endl;
			}
		//}

		//--------------

		//Tensor product for evaluation of values = N2XN1 where N1,N2 are 1-d matrices as stored in shapeInfo
			//This is here done only for 2-D tensor product. 3-D can be similarly created

		mylog<<"Using ShapeInfo functions "<<std::endl;
		double xShapeMatrix[fe_degree+1][n_q_points_1d];
		double yShapeMatrix[fe_degree+1][n_q_points_1d];
		double zShapeMatrix[fe_degree+1][n_q_points_1d];

		//Fill n1, N2 matrices
		for (int i=0; i<fe_degree+1; i++)
		{
			for (int j=0; j<n_q_points_1d; j++)
			{
				xShapeMatrix[i][j] = shape_info_old.shape_values[i*n_q_points_1d+j][0];
				yShapeMatrix[i][j] = xShapeMatrix[i][j];
				zShapeMatrix[i][j] = xShapeMatrix[i][j];
				mylog<<std::setw(20)<<xShapeMatrix[i][j];
			}
			mylog<<std::endl;
		}


		//Calculate and store their tensor product
		double outputMatrix[fe_dofs_per_cell][n_q_points];
		int row = 0, col = 0;
		int yrows = fe_degree+1, ycols = n_q_points_1d;
		int xrows = fe_degree+1, xcols = n_q_points_1d;

		double temp;

		for (int yi=0; yi<yrows; yi++)
		{
			for (int yj=0; yj<ycols; yj++)
			{
				temp = yShapeMatrix[yi][yj];
				for (int xi=0; xi<xrows; xi++)
				{
					for (int xj=0; xj<xcols; xj++)
					{
						col = yj*xcols+xj;
						row = yi*xrows+xi;
						outputMatrix[row][col] = temp * xShapeMatrix[xi][xj];
					}
				}
			}
		}

		mylog<<"Using Tensor Product of ShapeInfo "<<std::endl;
		//for (int c=0; c<n_components; c++)
		//{
		//	mylog<<"For component number ==========="<<c<<std::endl;
			for (unsigned int i=0; i<fe_dofs_per_cell; ++i)
			{
				mylog<<std::setw(10)<<i;
			  for (unsigned int q=0; q<n_q_points; ++q)
		  	  {
			  	  mylog<<std::setw(20)<<outputMatrix[i][q];
		  	  }
		  	  mylog<<std::endl;
			}

#if 0
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
