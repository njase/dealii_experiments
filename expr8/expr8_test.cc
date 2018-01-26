//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
//			 - compare values, gradients and hessians
//Remark: This experiment works and by using different quadrature points for different
// components, the resulting matrix is stable and invertible.
//HOWVER: dealii inverson results are incorrect!
// I checked with matlab
// After identifying the coordinate transformation matrices,
// these will be hardcoded for several degress due to the dealii bug!
//IT DOES NOT WORK THIS WAY _ SEE EXPLANATIONS BELOW
//Since dealii raw basis functions are also defined Lagrange polynomials with TP on each
//component -- this is exactly what I need.
//C matrix is already calculated and need not repeat this effort. This test is stopped here
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
#include <deal.II/base/table_indices.h>//debug
#include <deal.II/lac/full_matrix.h> //debug

#include <iostream>
#include <complex>
#include <vector>

using namespace dealii;
using namespace dealii::internal;
using namespace std;


const double RTbasisTp_2_1[12][12] = {
		{3.688745294522960, 3.688698417739943, 2.688731866219314 ,2.688677954691229,
		 2.438744373735972, 2.438673018332338, -2.727004161948571, -8.055992644745857,
		 -3.719523632898927, -2.575390277750557, 13.214999134885147, 0.365811850788305},
		 {3.086195172087173 ,  6.550358689288259 ,  4.818251285221777 ,  4.818329790563439 ,
		  5.251267675455892 ,  4.385278240035404, -4.955912901394186, -24.134806016518269,
		  -8.528221198532265,  -4.443853018761729,  47.716408709995449,   5.271485424047569},
		{-2.321653073420748, -2.321583998855203, -1.321635395521298, -1.321555427042767,
		 -2.571651071077213, -2.571555789792910, 2.319398310733959,  2.004084509797394,
		 2.260587957454845, 2.241976860328577, -8.855871140956879, 0.174781426321715},
		 {-2.782075365306810, -2.781963143381290, -4.514094172744080, -1.049879585276358,
		  -2.349055554368533, -3.214945252635516, 2.615872393362224, -20.540133458096534,
		  -1.697300350177102,  2.605661406880245, -21.967637538909912, -1.971524186665192},
		  {2.592302530189045, 2.592274433583952,  2.592294550384395,  2.592260976671241,
			2.592302673263475, 2.592257398413494, -1.618354659294710, -6.247031801380217,
		   -3.544195846538059, -1.519955282099545, 7.558230282738805, -0.892738583381288},
		   {13.106823971960694, 13.106469337362796, 13.106727236765437, 13.106328266789205,
			13.106808487442322, 13.106316825607792, -14.928340289858170,-25.654511229135096,
			-15.083336541894823,-10.697132536442950, 81.964068967849016, 4.720074991928414},
			{0.670345053018536, 0.670335718605202, 0.670342258759774, 0.670330918743275,
			0.670345312624704, 0.670329049113207, -0.695101419114508, -3.143738654209301,
			-1.587432346539572, -0.648148790409323, 3.443830498959869, -0.322217432025354},
			{-2.327534593641758, -2.327473459183238, -2.327517794212326, -2.327449288568459,
			-2.327531837858260, -2.327446638140827, 2.339844674454071,  2.321756316814572,
			3.092028058250435, 2.205773913534358, -13.023093588650227, -1.386479660286568},
			{-0.432371724047698, -0.432399128796533, -0.432377111050300, -0.432408949243836,
			1.067628073506057, 1.067598665715195, 0.482284876634367, 7.436145572923124,
			1.777538653695956, 0.392393874702975, -5.177840638905764, -0.645085904398002},
			{-2.327927327249199, -2.327895103720948, -2.327917691320181, -2.327878558076918,
			-2.327928208513185, -2.327872153371572,2.388135411543772, 10.775004612281919,
			5.450231271563098, 2.227084867889062, -11.820772463455796, 1.110464304685593},
			{-2.456214245932642, -2.456336154486053, -2.456235158198979, -2.456385111145210,
			-5.054298089176882,  0.141729230817873,2.683952090097591, 34.411464667646214,
			8.593633295502514, 2.236989879922476,-28.307372552808374, -3.452056165609974},
			{-2.389480209210888, -2.389509639004245, -2.389481949619949, -2.389528895495459,
			-2.389485988533124, -2.389504124177620,2.382965010125190,  1.475021407939494,
			-0.384207728784531,  2.033448050031438, -47.567886173725128, -4.607211145572364}
};

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



template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d=fe_degree+2, int base_fe_degree=fe_degree>
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
	FullMatrix<double> evaluate_values_2dtensorp(
			int c, int n_dofs, std::vector<Point<dim>> &q_points);
	FullMatrix<double> evaluate_fe_unit_values(int c, std::vector<Point<dim>> &q_points);
	void generate_reference_points(int c, std::vector<Point<dim,Number>> &q_points);

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
FullMatrix<double> Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::
evaluate_fe_unit_values(int c, std::vector<Point<dim>> &q_points)
{
	FE_RaviartThomas<dim> fe_rt(fe_degree);
	FullMatrix<double> res_matrix(fe_rt.dofs_per_cell,q_points.size());

	//Evaluate all basis functions on these points, for only one vector component
	for (int i=0;i<fe_rt.dofs_per_cell;i++)
	{
		for (int q=0; q<q_points.size();q++)
		{
			res_matrix(i,q) = fe_rt.shape_value_component(i,q_points[q],c);
		}
	}

	return res_matrix;
}


template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
FullMatrix<double> Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::
	evaluate_values_2dtensorp(int c, int n_dofs, std::vector<Point<dim>> &q_points)
{
	//Lets evaluate phi_hat matrix for c = 0
	const FE_Q<1> fe1(fe_degree+1);
	const FE_Q<1> fe2(fe_degree);

	FullMatrix<double> res_matrix(n_dofs,q_points.size());
	int k = 0;

	if (c == 0)
	{
		for (int q=0; q<q_points.size();q++)
		{
			k = 0;
			Point<1> px(q_points[q][0]);
			Point<1> py(q_points[q][1]);

			for (int i=0; i<fe1.dofs_per_cell; i++)
			{
				for (int j=0; j<fe2.dofs_per_cell; j++)
				{
					double v1 = fe1.shape_value(i,px);
					double v2 = fe2.shape_value(j,py);
					double vres = v1*v2;

					res_matrix(k++,q) = vres;

			//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[q]<<") = ("<<
			//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
				}
			}
		}
	}
	if (c == 1)
	{
		for (int q=0; q<q_points.size();q++)
		{
			k = fe1.dofs_per_cell*fe2.dofs_per_cell;
			Point<1> px(q_points[q][0]);
			Point<1> py(q_points[q][1]);

			for (int i=0; i<fe2.dofs_per_cell; i++)
			{
				for (int j=0; j<fe1.dofs_per_cell; j++)
				{
					double v1 = fe2.shape_value(i,px);
					double v2 = fe1.shape_value(j,py);
					double vres = v1*v2;

					res_matrix(k++,q) = vres;

				//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[q]<<") = ("<<
				//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
				}
			}
		}
	}

	return res_matrix;
}

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
void Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::generate_reference_points(
		int c, std::vector<Point<dim,Number>> &q_points)
{
	//We choose points as tensor product of Gauss quadrature
	//THe following selection has shown to give stable results
	int qdegree = fe_degree+3;
	if (c==1)
		qdegree = fe_degree+5;

	QGauss<1> quad(qdegree);

	int p = 0;

	for (auto y:quad.get_points())
	{
		for (auto x:quad.get_points())
		{
			if (p < q_points.size())
			{
				q_points[p][0] = x[0];
				q_points[p][1] = y[0];
				//std::cout<<"  Point = "<<q_points[p];
				p++;
			}
			else
				return;

		}
	}

}

template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::run(bool debug)
{
	mylog.debug = debug;

	bool res = false;

	FE_RaviartThomas<dim> fe_rt(fe_degree);
	QGauss<dim> fe_quad(n_q_points_1d);

    const unsigned int n_dofs = fe_rt.dofs_per_cell;
    const unsigned int n_dofs_per_comp = fe_rt.dofs_per_cell/n_components;

    std::cout<<"n_dofs = "<<n_dofs<<std::endl;


	//Define some points and dofs values on which to test results
	std::srand(std::time(nullptr)); // random dof values between 0 and 1
    Vector<double> dofs_actual(n_dofs);

    for (int i=0; i<n_dofs; i++)
    	dofs_actual[i] = std::rand()/static_cast<Number>(RAND_MAX);


#if 0 //This commented portion works - this is for reference evaluation of basis transformation matrix
    //Find no. of shape functions
	unsigned int n_points = n_dofs;

	if ((n_dofs != (fe_degree+2)*(fe_degree+1)*dim) && (n_dofs_per_comp != (fe_degree+2)*(fe_degree+1)))
	{
		std::cout<<"Conceptual bug..Error " <<std::endl;
		return false;
	}

    FullMatrix<Number> phi_matrix(n_dofs,n_dofs);
	FullMatrix<Number> phi_hat_matrix(n_dofs,n_dofs);
	FullMatrix<Number> C_matrix(n_dofs,n_dofs);

	std::vector<Point<dim,Number>> q_points(n_points);

	int c = 0;
	generate_reference_points(c,q_points);

	//lets generate phi matrix
	phi_matrix = evaluate_fe_unit_values(c,q_points);

	//Lets evaluate phi_hat matrix for c = 0 FIXME : This is only for 2 dim
	phi_hat_matrix = evaluate_values_2dtensorp(c, n_dofs, q_points);

	//Work for c = 1
	c = 1;
	generate_reference_points(c,q_points);

	phi_matrix.add(1,evaluate_fe_unit_values(c,q_points));

	//Lets evaluate phi_hat matrix for c = 1, on some different points
	phi_hat_matrix.add(1,evaluate_values_2dtensorp(c, n_dofs, q_points));

	std::cout<<std::endl<<std::endl;

	//print
	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(12)<<phi_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;

	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(12)<<phi_hat_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;


	//phi_hat_matrix.gauss_jordan();

	//phi_matrix.mmult(C_matrix,phi_hat_matrix);

	FullMatrix<Number> temp_phi_matrix(n_dofs,n_dofs);
	for (unsigned int i=0; i<n_dofs; i++)
		{
			for (unsigned int j=0; j<n_dofs; j++)
			{
				C_matrix(i,j) = RTbasisTp_2_1[i][j];
			}

		}

	//Calculate C.phi_hat := phi  Matches!
	C_matrix.mmult(temp_phi_matrix,phi_hat_matrix);

	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(12)<<temp_phi_matrix(i,j);
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl<<std::endl;
#endif

//This is test -- it fails
	//Analysis: I used not all basis functions to evaluate C matrix.
	//In principle all the basis functions should be used in one go to find out C matrix
	//This will be an overdetemrined system, and maybe LS method is then needed. Or maybe not.
	//But thats for experimentation
	//Since dealii raw basis functions are also defined Lagrange polynomials with TP on each
	//component -- this is exactly what I need.
	//C matrix is already calculated and need not repeat this effort.
	//
	if (evaluate_values)
	{
		//Lets take quadrature points and evaluate FE on it
		//std::vector<Point<dim>> test_points = fe_quad.get_points();
		std::vector<Point<dim>> test_points(1);
		test_points[0][0] = 0.15;
		test_points[0][1] = 0.15;
		const int n_points = test_points.size();

	    FullMatrix<Number> phi_matrix(n_dofs,n_points);
	    FullMatrix<Number> test_phi_matrix(n_dofs,n_points);
		FullMatrix<Number> phi_hat_matrix(n_dofs,n_points);
		FullMatrix<Number> C_matrix(n_dofs,n_dofs);

		for (unsigned int i=0; i<n_dofs; i++)
		{
			for (unsigned int j=0; j<n_dofs; j++)
			{
				C_matrix(i,j) = RTbasisTp_2_1[i][j];
			}
		}


		int c = 0;
		//lets generate phi matrix
		phi_matrix = evaluate_fe_unit_values(c,test_points);

		for (unsigned int i=0; i<n_dofs; i++)
		{
			for (unsigned int j=0; j<n_points; j++)
			{
				std::cout <<std::setw(12)<<phi_matrix(i,j);
			}
			std::cout<<std::endl;
		}

		std::cout<<std::endl<<std::endl;

		//Lets evaluate phi_hat matrix
		phi_hat_matrix = evaluate_values_2dtensorp(c, n_dofs, test_points);

		for (unsigned int i=0; i<n_dofs; i++)
		{
			for (unsigned int j=0; j<n_points; j++)
			{
				std::cout <<std::setw(12)<<phi_hat_matrix(i,j);
			}
			std::cout<<std::endl;
		}

		std::cout<<std::endl<<std::endl;

		//Calculate C.phi_hat := phi
		C_matrix.mmult(test_phi_matrix,phi_hat_matrix);
		for (unsigned int i=0; i<n_dofs; i++)
		{
			for (unsigned int j=0; j<n_points; j++)
			{
				std::cout <<std::setw(12)<<test_phi_matrix(i,j);
			}
			std::cout<<std::endl;
		}

		std::cout<<std::endl<<std::endl;
	}


	std::cout<<std::endl;


#if 0
	//This debug test shows that even raw RT polynomials are zero for one half
	// and non-zero for other half..just like mine
	//But still their basis transformation works!! and mine fails -- why??
	const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
			dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_rt);

	//Choose some point
	Point<dim> p(0.15, 0.18);
	//Evaluate basis functions on this point
	std::vector<Tensor<1,dim>> poly_values(fe_rt.dofs_per_cell);
	std::vector<Tensor<2,dim>> unused2;
	std::vector<Tensor<3,dim>> unused3;
	std::vector<Tensor<4,dim>> unused4;
	std::vector<Tensor<5,dim>> unused5;

	fe_poly->poly_space.compute(p,poly_values,unused2,unused3,unused4,unused5);

	std::vector<Tensor<1,dim>> fe_values(fe_rt.dofs_per_cell);
	//Evaluate all basis functions on these points, for both vector components
	for (int i=0;i<fe_rt.dofs_per_cell;i++)
	{
		fe_values[i][0] = fe_rt.shape_value_component(i,p,0);
		fe_values[i][1] = fe_rt.shape_value_component(i,p,1);
	}

	for (int i=0;i<fe_rt.dofs_per_cell;i++)
	{
		std::cout<<"("<<poly_values[i][0]<<", "<<fe_values[i][0]<<")"<<std::endl;
	}

	std::cout<<std::endl;
	for (int i=0;i<fe_rt.dofs_per_cell;i++)
	{
		std::cout<<"("<<poly_values[i][1]<<", "<<fe_values[i][1]<<")"<<std::endl;
	}
#endif

#if 0
	///Debugging to confirm my understanding of TP -- this understanding is correct
	//TP evalation inside dealii happens similar to as I expect
	const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
			dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_rt);

	//Choose some point
	Point<dim> p(0.15, 0.18);
	//Evaluate basis functions on this point
	std::vector<Tensor<1,dim>> poly_values(fe_rt.dofs_per_cell);
	std::vector<Tensor<2,dim>> unused2;
	std::vector<Tensor<3,dim>> unused3;
	std::vector<Tensor<4,dim>> unused4;
	std::vector<Tensor<5,dim>> unused5;

	fe_poly->poly_space.compute(p,poly_values,unused2,unused3,unused4,unused5);


	std::vector<Point<dim>> test_points(1);
	test_points[0] = p;
	const int n_points = test_points.size();

    FullMatrix<Number> phi_hat_matrix(n_dofs,n_points);

	phi_hat_matrix = evaluate_values_2dtensorp(0 /*c*/, n_dofs, test_points);

	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_points; j++)
		{
			std::cout <<std::setw(12)<<phi_hat_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;
#endif

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

	//for RT, give n-1_points appropriately
    res = Test<double,2,2,1>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);
    //res = Test<double,2,2,2>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

    //res = Test<double,3,3,2>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);
    //res = Test<double,3,3,3>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

	//Dont run higher degrees as it hangs...FIXME debug later
	//res = Test<double,3,3,3>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

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
