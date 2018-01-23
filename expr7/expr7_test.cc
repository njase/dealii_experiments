//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
//			 - compare values, gradients and hessians
//Remark: This experiment stopped after finding that the basis transformation matrix evaluation
// using same gauss quad for each coordinate is not feasible as the matrix is singular
// This code is now kept for reference reasons
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
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::run(bool debug)
{
	mylog.debug = debug;

	bool res = false;

	FE_RaviartThomas<dim> fe_rt(fe_degree);
	QGauss<dim> fe_quad(n_q_points_1d);
	QGauss<1>   quad_1d(n_q_points_1d);


    const unsigned int   n_q_points    = fe_quad.size();
    const unsigned int n_dofs = fe_rt.dofs_per_cell;
    const unsigned int n_dofs_per_comp = fe_rt.dofs_per_cell/n_components;


	//Define some points and dofs values on which to test results
	std::vector<Point<dim,Number>> points(n_q_points);
	std::srand(std::time(nullptr)); // random dof values between 0 and 1
    Vector<double> dofs_actual(n_dofs);

    for (int i=0; i<n_dofs; i++)
    	dofs_actual[i] = std::rand()/static_cast<Number>(RAND_MAX);


//This is inverting to find out C matrix -- inversion is unstable

    //Find no. of shape functions
	unsigned int n_points = n_dofs;

	if ((n_dofs != (fe_degree+2)*(fe_degree+1)*dim) && (n_dofs_per_comp != (fe_degree+2)*(fe_degree+1)))
	{
		std::cout<<"Conceptual bug..Error " <<std::endl;
		return false;
	}


	//We choose points as tensor product of Gauss quadrature
	//testing for deg=1m so 12 points
	QGauss<1>   quad_x(4);
	QGauss<1>   quad_y(3);


	std::vector<Point<dim,Number>> q_points(n_points);
	int p = 0;

	for (auto y:quad_y.get_points())
	{
		for (auto x:quad_x.get_points())
		{
			if (p < n_points)
			{
				q_points[p][0] = x[0];
				q_points[p][1] = y[0];
				std::cout<<"  Point = "<<q_points[p];
			}
			p++;
		}
	}

	std::cout<<std::endl;
	std::cout<<std::endl;


	int c = 0;

	FullMatrix<Number> phi_matrix(n_dofs,n_dofs);
	FullMatrix<Number> phi_hat_matrix(n_dofs,n_dofs);
	FullMatrix<Number> C_matrix(n_dofs,n_dofs);

	//lets generate phi matrix
	//Evaluate all basis functions on these points, for only one vector component
	for (int i=0;i<n_dofs;i++)
	{
		for (int q=0; q<q_points.size();q++)
		{
			phi_matrix(i,q) = fe_rt.shape_value_component(i,q_points[q],c);
		}
	}


	//Lets evaluate phi_hat matrix for c = 0 FIXME : This is only for 2 dim
	const FE_Q<1> fe1(fe_degree+1);
	const FE_Q<1> fe2(fe_degree);

	int k = 0;
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

				phi_hat_matrix(k++,q) = vres;

			//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[q]<<") = ("<<
			//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
			}
		}
	}


	//Work for c = 1
	c = 1;


	//Evaluate all basis functions for other vector component
	for (int i=0;i<n_dofs;i++)
	{
		for (int q=0; q<q_points.size();q++)
		{
			phi_matrix(i,q) += fe_rt.shape_value_component(i,q_points[q],c);
		}
	}

	//Lets evaluate phi_hat matrix for c = 1,
	for (int q=0; q<q_points.size();q++)
	{
		k = n_dofs_per_comp;
		Point<1> px(q_points[q][0]);
		Point<1> py(q_points[q][1]);

		for (int i=0; i<fe2.dofs_per_cell; i++)
		{
			for (int j=0; j<fe1.dofs_per_cell; j++)
			{
				double v1 = fe2.shape_value(i,px);
				double v2 = fe1.shape_value(j,py);
				double vres = v1*v2;

				phi_hat_matrix(k++,q) = vres;

			//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[q]<<") = ("<<
			//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
			}
		}
	}



#if 0
	//now here is an alternative. i choose just one point, and evaluate
	//all the basis functions for all compnents in this one point
	//problem: This will be a non-square matrix, so inverse cant be correctly evaluated

	FullMatrix<Number> phi_matrix(n_dofs,dim);
	FullMatrix<Number> phi_hat_matrix(n_dofs,dim);
	FullMatrix<Number> C_matrix(n_dofs,n_dofs);


	Point<dim> p(0.0694318,0.887298); //This is chosen from above commented code
	Point<1> px(0.0694318);
	Point<1> py(0.887298);

	//lets generate phi matrix
	//Evaluate all basis functions on this point, for all vector components
	for (int i=0;i<n_dofs;i++)
	{
		for (int c=0;c<n_components;c++)
		{
			phi_matrix(i,c) = fe_rt.shape_value_component(i,p,c);
		}
	}

	//Lets evaluate phi_hat matrix FIXME : This is only for 2 dim
	int c = 0,k=0;
	for (int i=0; i<fe1.dofs_per_cell; i++)
	{
		for (int j=0; j<fe2.dofs_per_cell; j++)
		{
			double v1 = fe1.shape_value(i,Point<1>(px));
			double v2 = fe2.shape_value(j,Point<1>(py));
			double vres = v1*v2;

			phi_hat_matrix(k++,c) = vres;

		//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[k]<<") = ("<<
		//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
		}
	}

	c = 1,k=fe1.dofs_per_cell*fe2.dofs_per_cell;
	for (int i=0; i<fe2.dofs_per_cell; i++)
	{
		for (int j=0; j<fe1.dofs_per_cell; j++)
		{
			double v1 = fe2.shape_value(i,Point<1>(px));
			double v2 = fe1.shape_value(j,Point<1>(py));
			double vres = v1*v2;

			phi_hat_matrix(k++,c) = vres;

		//std::cout<<i<<","<<j<<" shape functions values on Point ("<<q_points[k]<<") = ("<<
		//v1<<", "<<v2<<", "<<vres<<")"<<std::endl; //1st shape function for 1st point
		}
	}
#endif


	//print
	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(15)<<phi_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;


	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(15)<<phi_hat_matrix(i,j);
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl<<std::endl;

	FullMatrix<Number> temp_phi_hat_matrix(n_dofs,n_dofs);
	temp_phi_hat_matrix = phi_hat_matrix;

	phi_hat_matrix.gauss_jordan();

	std::cout<<std::endl<<std::endl;

	phi_matrix.mmult(C_matrix,phi_hat_matrix);


	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(15)<<C_matrix(i,j);
		}
		std::cout<<std::endl;
	}


	std::cout<<std::endl<<std::endl;

	C_matrix.mmult(phi_matrix,temp_phi_hat_matrix);

	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_dofs; j++)
		{
			std::cout <<std::setw(15)<<phi_matrix(i,j);
		}
		std::cout<<std::endl;
	}


	/////////////////Debug over



	std::cout<<std::endl;

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
