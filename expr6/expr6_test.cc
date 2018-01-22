//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
//			 - compare values, gradients and hessians
//Remark: This experiment stopped after finding that the basis transformation matrix evaluation
// using generalized support points is not feasible.
// This code is now kept for reference reasons
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

	bool reference_test(
			FE_RaviartThomas<dim> &fe_rt,
			std::vector<Point<dim,Number>> &points,
			Vector<double> &dofs_actual);
	bool actual_test(QGauss<dim> &fe_quad, Vector<double> &dofs_actual);
	FullMatrix<double> create_node_matrix();
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


//This is just to confirm the understanding that rt.shape_value_component actually evaluates
// in 2 stages:
// 1. evaluate in raw basis using underlying poly space compute function
// 2. Convert results to real basis using transformation matrix / node matrix
template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::reference_test(
		FE_RaviartThomas<dim> &fe_rt,
		std::vector<Point<dim,Number>> &points,
		Vector<double> &dofs_actual)
{
	bool res = true;

	const Number tol = 10e-12;
	const unsigned int n_dofs = fe_rt.dofs_per_cell;

	const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
			dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_rt);

	for (auto p:points)
	{
		for (int c=0; c<n_components; c++)
		{
			//Evaluate phi and phi_hat on this point
			std::vector<Tensor<1,dim>> phi_hat_values(n_dofs);
			std::vector<Tensor<2,dim>> unused2;
			std::vector<Tensor<3,dim>> unused3;
			std::vector<Tensor<4,dim>> unused4;
			std::vector<Tensor<5,dim>> unused5;

			fe_poly->poly_space.compute(p,phi_hat_values,unused2,unused3,unused4,unused5);

			std::vector<Tensor<1,dim>> phi_values(n_dofs);
			for (int i=0; i<n_dofs; i++)
				phi_values[i][c] = fe_rt.shape_value_component(i,p,c);

			Vector<double> rhs(n_dofs);
			Vector<double> lhs(n_dofs);
			Number resrhs = 0, reslhs=0;
			//Evaluate RHS and LHS
			for (int i=0; i<n_dofs; i++)
			{
				rhs[i] = 0.;
				for (int j=0; j<n_dofs;j++) //cols of C_tra
				{
					rhs[i] += dofs_actual[j]*(fe_poly->inverse_node_matrix(i,j));
				}

				rhs[i] *= phi_hat_values[i][c];
				resrhs += rhs[i];

				lhs[i] = phi_values[i][c]*dofs_actual[i];
				reslhs += lhs[i];

				//std::cout<<"lhs, rhs = ("<<lhs[i]<<" ,"<<rhs[i]<<")"<<std::endl;
			}

			res &= ((std::abs(reslhs - resrhs) < tol)?true:false);
		}
	}

	return res;

}



template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::actual_test(
		QGauss<dim> &fe_quad, Vector<double> &dofs_actual)
{
	bool res = true;

	FE_RaviartThomas<dim> fe_rt(fe_degree);
	QGauss<1>   quad_1d(n_q_points_1d);

	const unsigned int first_selected_component = 0;
	    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info(quad_1d, fe_rt, fe_rt.component_to_base_index(first_selected_component).first,true);

	const Number tol = 10e-12;
	const unsigned int rt_dofs_per_cell = fe_rt.dofs_per_cell;
	const unsigned int rt_dofs_per_comp = rt_dofs_per_cell/n_components;

	const unsigned int   n_q_points    = fe_quad.size();


    FullMatrix<double> inverse_node_matrix;
    FullMatrix<double> node_matrix;


    node_matrix = create_node_matrix();
    for (int i=0; i<rt_dofs_per_cell; i++)
    {
    	for (int j=0; j<rt_dofs_per_cell; j++)
    		std::cout<<std::setw(15)<<node_matrix(i,j);

    	std::cout<<std::endl;
    }
    inverse_node_matrix.reinit(rt_dofs_per_cell, rt_dofs_per_cell);
    inverse_node_matrix.invert(node_matrix); //this is C_tra = X_inv

    //Convert dofs using node matrix, u_hat = C_tra.u
    Vector<double> dofs_new(dofs_actual.size());
    inverse_node_matrix.vmult(dofs_new,dofs_actual);


	int c; //component
	c = 0;

    //using Finite Element functions

    mylog<<"Using Finite Element functions "<<std::endl;

    Vector<double> lhs_x(rt_dofs_per_comp);

    FullMatrix<double> phi_values(rt_dofs_per_cell,n_q_points);

    for (int i=0; i<rt_dofs_per_cell; i++)
    {
    	for (int j=0; j<n_q_points;j++)
    	{
    		Point<dim> p = fe_quad.point(i);
    		phi_values(i,j) = fe_rt.shape_value_component(i,p,c);
    	}
    }
    phi_values.Tvmult(lhs_x,dofs_actual);


	//Tensor product for evaluation of values = N3XN2XN1 where N1,N2,N3 are 1-d matrices as stored in shapeInfo

	mylog<<"Using ShapeInfo functions "<<std::endl;
	if (rt_dofs_per_comp != ((fe_degree+2)*(fe_degree+1)))
	{
		std::cout<<"Conceptual bug - exit"<<std::endl;
		return false;
	}

	FullMatrix<double> outputMatrix;
	outputMatrix.reinit((fe_degree+2)*(fe_degree+1),n_q_points);

	FullMatrix<double> bigMatrix(fe_degree+2, n_q_points_1d);
	FullMatrix<double> smallMatrix(fe_degree+1,n_q_points_1d);
	int yrows, ycols = n_q_points_1d;
	int xrows, xcols = n_q_points_1d;
	int row = 0, col = 0;

	const int x_dir=0;
	const int y_dir=1;
	const int z_dir=2;

	//Although we could simply make them from shape values of any component, but for better verification
	//of code, lets make them separately during each component

	//For 0th component, X is bigMatrix and y is smallmatrix

	//make x and y matrices for c = 0
	xrows = fe_degree+2; //dof = fe_degree + 1
	yrows = fe_degree+1;

	for (int i=0; i<xrows; i++)
	{
		for (int j=0; j<xcols; j++)
		{
			bigMatrix(i,j) = shape_info.shape_values_vec[c][x_dir][i*n_q_points_1d+j][0];
			mylog<<std::setw(20)<<bigMatrix(i,j);
		}
		mylog<<std::endl;
	}

	for (int i=0; i<yrows; i++)
	{
		for (int j=0; j<ycols; j++)
		{
			smallMatrix(i,j) = shape_info.shape_values_vec[c][y_dir][i*n_q_points_1d+j][0];
			mylog<<std::setw(20)<<smallMatrix(i,j);
		}
		mylog<<std::endl;
	}

	//now evaluate
	double temp;

	for (int yi=0; yi<yrows; yi++)
	{
		for (int yj=0; yj<ycols; yj++)
		{
			temp = smallMatrix(yi,yj);
			for (int xi=0; xi<xrows; xi++)
			{
				for (int xj=0; xj<xcols; xj++)
				{
					col = yj*xcols+xj;
					row = yi*xrows+xi;
					outputMatrix(row,col) = temp * bigMatrix(xi,xj);
				}
			}
		}
	}


	//now lets multiply the outputMatrix transpose with dofs_new for c=0
	Vector<double> rhs(rt_dofs_per_comp);
	outputMatrix.Tvmult(rhs,dofs_new);


	for (int i=0;i<rt_dofs_per_comp;i++)
	{
		std::cout<<"lhs, rhs = ("<<lhs_x[i]<<" ,"<<rhs[i]<<")"<<std::endl;
	}

	return res;

}


template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree>
FullMatrix<double> Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::create_node_matrix()
{
	//std::cout<<"Hello"; FullMatrix<double> N; return N;
	//Now we need to temporarily create node matrix
	//Reference compute_node_matrix
	////debug - temporary stuff

	FE_RaviartThomas<dim> fe_in(fe_degree);
	const unsigned int n_dofs = fe_in.dofs_per_cell;
	FullMatrix<double> N (n_dofs, n_dofs);

	const std::vector<Point<dim> > &points = fe_in.get_generalized_support_points();

    std::vector<std::vector<Vector<double> > >
    support_point_values (n_dofs, std::vector<Vector<double> >(points.size(), Vector<double>(dim)));


	const FE_Q<1> fe1(fe_degree+1);
	const FE_Q<1> fe2(fe_degree);

#if 0 //debug
	std::cout<<"1D support points are "<<std::endl;
	//for (auto p:fe1.get_unit_support_points())
	QGauss<1>   quad_1d(n_q_points_1d);
	for (auto p:quad_1d.get_points())
		std::cout<<std::setw(10)<<"("<<p<<","<<fe1.shape_value(0,p)<<")";
	std::cout<<std::endl;
	//for (auto p:fe2.get_unit_support_points())
	for (auto p:quad_1d.get_points())
		std::cout<<std::setw(10)<<"("<<p<<","<<fe2.shape_value(0,p)<<")";
	std::cout<<std::endl;


	for (int i=0; i<fe1.dofs_per_cell; i++)
	{
		for (int j=0; j<fe2.dofs_per_cell; j++)
		{
		std::cout<<i<<","<<j<<" shape functions values on Point ("<<points[0]<<") = ("<<
		fe1.shape_value(i,Point<1>(points[0][0]))<<", "<< /*1st shape function for 1st point */
		fe2.shape_value(j,Point<1>(points[0][1]))<<")"<<std::endl; //1st shape function for 1st point
		double v1 = fe1.shape_value(i,Point<1>(points[0][0]));
		double v2 = fe2.shape_value(j,Point<1>(points[0][1]));
		double vres = v1*v2;
		std::cout<<"v1 = "<<v1<<" v2 = "<<v2<<" Result = "<< vres<<std::endl;
				//fe2.shape_value(j,Point<1>(points[0][0]));

		}
	}
#endif

	if ((fe1.dofs_per_cell*fe2.dofs_per_cell*dim) != n_dofs)
	{
		std::cout<<"Conceptual bug - EXIT"<<std::endl;
		return N;
	}

	int pn = 0; //point index
	for (auto p:points)
	{
		std::cout<<"Point is "<<p<<std::endl;
		Point<1> px(p[0]), py(p[1]);
		int n = 0; //dof index
		//Assume only 2 components
		//first component
		for (int i=0; i<fe1.dofs_per_cell; i++)
		{
			for (int j=0; j<fe2.dofs_per_cell; j++)
			{
				support_point_values[n][pn][0] = fe1.shape_value(i,px)*fe2.shape_value(j,py);
				support_point_values[n][pn][1] = 0;
			}
		}

		//second component
		for (int i=0; i<fe2.dofs_per_cell; i++)
		{
			for (int j=0; j<fe1.dofs_per_cell; j++)
			{
				support_point_values[n][pn][0] = 0;
				support_point_values[n][pn][1] = fe2.shape_value(i,px)*fe1.shape_value(j,py);
			}
		}
		pn++;
		n++;
	}

	//////debug
	//Print the support point values
	for (int i=0; i<n_dofs;i++)
	{
		for (int j=0;j<points.size();j++)
			std::cout<<std::setw(15)<<support_point_values[i][j][0]<<", "<<support_point_values[i][j][1];

		std::cout<<std::endl;
	}


	/////debug over

	std::vector<double> nodal_values(n_dofs);
    for (unsigned int i=0; i<n_dofs; ++i)
    {
        // get the values of the current set of shape functions
        // at the generalized support points
        fe_in.convert_generalized_support_point_values_to_dof_values(support_point_values[i],
                                                                  nodal_values);

        // Enter the interpolated dofs into the matrix
        for (unsigned int j=0; j<n_dofs; ++j)
          {
            N(j,i) = nodal_values[j];
            Assert (numbers::is_finite(nodal_values[j]), ExcInternalError());
          }
    }

    return N;

	/////debug over
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
	//fe_unit_values.resize((dofs_per_cell*n_q_points)/n_components);
	//shape_values_old.resize((dofs_per_cell*n_q_points)/n_components);


	//Define some points and dofs values on which to test results
	std::vector<Point<dim,Number>> points(n_q_points);
	std::srand(std::time(nullptr)); // random dof values between 0 and 1
    //std::vector<Number> dofs_actual(n_dofs);
    Vector<double> dofs_actual(n_dofs);

    for (int i=0; i<n_dofs; i++)
    	dofs_actual[i] = std::rand()/static_cast<Number>(RAND_MAX);


    res = actual_test(fe_quad,dofs_actual);
    std::cout<<"Actual result = "<<(res == true?"Pass":"Fail")<<std::endl;

#if 0
    res = reference_test(fe_rt,points,dofs_actual);
    std::cout<<"Reference result = "<<(res == true?"Pass":"Fail")<<std::endl;
#endif

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
