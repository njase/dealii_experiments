//////////////////
//
// Purpose : To compare unit cell evaluation of - basis functions evaluated on all quad points
//			 - Using FiniteElement functions and ShapeInfo functions
// 			 - for RT elements, for all components
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

void test_mesh (Triangulation<3> &tria,
                  const double scale_grid = 1.)
{
  const unsigned int dim = 3;
  std::vector<Point<dim> > points (8);

  // build the mesh layer by layer from points

  // 1. cube cell
  points[0] = Point<dim> (0,0,0);
  points[1] = Point<dim> (0,1.,0);
  points[2] = Point<dim> (0,0,1);
  points[3] = Point<dim> (0,1.,1);
  points[4] = Point<dim> (1.,0,0);
  points[5] = Point<dim> (1.,1.,0);
  points[6] = Point<dim> (1.,0,1);
  points[7] = Point<dim> (1.,1.,1);

  // connect the points to cells
  std::vector<CellData<dim> > cells(1);
  for (unsigned int i=0; i<1; ++i)
    {
      cells[i].vertices[0] = 0+4*i;
      cells[i].vertices[1] = 4+4*i;
      cells[i].vertices[2] = 1+4*i;
      cells[i].vertices[3] = 5+4*i;
      cells[i].vertices[4] = 2+4*i;
      cells[i].vertices[5] = 6+4*i;
      cells[i].vertices[6] = 3+4*i;
      cells[i].vertices[7] = 7+4*i;
      cells[i].material_id = 0;
    }
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
	bool reference_test(
			FE_RaviartThomas<dim> &fe_rt,
			std::vector<Point<dim,Number>> &points,
			std::vector<Number> &dofs_actual);
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
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::reference_test(
		FE_RaviartThomas<dim> &fe_rt,
		std::vector<Point<dim,Number>> &points,
		std::vector<Number> &dofs_actual)
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

			std::vector<Number> rhs(n_dofs);
			std::vector<Number> lhs(n_dofs);
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
bool Test<Number,n_components,dim,fe_degree,n_q_points_1d,base_fe_degree>::run(bool debug)
{
	mylog.debug = debug;

	bool res = false;

	FE_RaviartThomas<dim> fe_rt(fe_degree);
	QGauss<dim> fe_quad(n_q_points_1d);
	QGauss<1>   quad_1d(n_q_points_1d);

	const unsigned int first_selected_component = 0;
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info(quad_1d, fe_rt, fe_rt.component_to_base_index(first_selected_component).first,true);

    const unsigned int   n_q_points    = fe_quad.size();
    const unsigned int n_dofs = fe_rt.dofs_per_cell;
	//fe_unit_values.resize((dofs_per_cell*n_q_points)/n_components);
	//shape_values_old.resize((dofs_per_cell*n_q_points)/n_components);


	Point<dim> p;


#if 0
	//Define some points and dofs values on which to test results
	std::vector<Point<dim,Number>> points(n_q_points);
	std::srand(std::time(nullptr)); // random dof values between 0 and 1
    std::vector<Number> dofs_actual(n_dofs);
    for (int i=0; i<n_dofs; i++)
    	dofs_actual[i] = std::rand()/static_cast<Number>(RAND_MAX);

    res = reference_test(fe_rt,points,dofs_actual);
    std::cout<<"Reference result = "<<(res == true?"Pass":"Fail")<<std::endl;
#endif

    //////////////////////////// Debug
    //Evaluate nodal values (i.e. node functional applied) for basis functions on generalized support points
    const std::vector<Point<dim> > &points = fe_rt.get_generalized_support_points();
    std::vector<Vector<double> > support_point_values (points.size(), Vector<double>(dim));
    std::vector<double> nodal_values(n_dofs);

    for (unsigned int i=0; i<n_dofs; ++i)
    {
    	for (unsigned int k=0; k<points.size(); ++k)
    		for (unsigned int d=0; d<dim; ++d)
    		{
    			support_point_values[k][d] = fe_rt.shape_value_component(i, points[k], d);
    		}
    }

    //const FiniteElement<dim,dim> *fe = dynamic_cast<const FiniteElement<dim,dim>*>(&fe_rt);
    fe_rt.convert_generalized_support_point_values_to_nodal_values(support_point_values,nodal_values);

    for (unsigned int j=0; j<n_dofs; ++j)
    {
    	std::cout<<"(basis evaluation, nodal value) = ("<<support_point_values[j][0]<<" ,"<<nodal_values[j]<<")"<<std::endl;
    }


    //////////// Debug end




#if 0 //This is inverting to find out C matrix -- inversion is unstable
	const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
			dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_rt);

	//Find no. of shape functions
	unsigned int n_dofs = fe_rt.dofs_per_cell;
	unsigned int n_points = n_dofs;

	if (n_dofs != (fe_degree+2)*(fe_degree+1)*dim)
	{
		std::cout<<"Conceptual bug..Error " <<std::endl;
		return false;
	}


	//Choose equidistant points between 0 and 1 for reference cell -- FIXME improve later after debugging
	//testing for deg=1m so 12 points
	QGauss<1>   quad_x(4);
	QGauss<1>   quad_y(3);

	std::vector<Point<dim,Number>> points(n_points);
	int p = 0;

	for (auto y:quad_y.get_points())
	{
		for (auto x:quad_x.get_points())
		{
			points[p][0] = x[0];
			points[p][1] = y[0];
			std::cout<<"  Point = "<<points[p];
			p++;
		}
	}

	std::cout<<std::endl;
	std::cout<<std::endl;

	FullMatrix<Number> phi_matrix(n_dofs,n_points);
	FullMatrix<Number> phi_hat_matrix(n_dofs,n_points);
	FullMatrix<Number> C_matrix(n_dofs,n_points);

	int c = 0; //component
	for (unsigned int i=0; i<n_dofs; i++)
	{
		//Evaluate phi (using rt.shape_value) and phi_hat on all points
		for (unsigned int j=0; j<n_points; j++)
		{
			//phi
			phi_matrix(i,j) = fe_rt.shape_value_component(i,points[j],c);
		}
	}


	//phi_hat
	std::vector<Tensor<1,dim>> values(n_dofs);
	std::vector<Tensor<2,dim>> unused2;
	std::vector<Tensor<3,dim>> unused3;
	std::vector<Tensor<4,dim>> unused4;
	std::vector<Tensor<5,dim>> unused5;

	for (unsigned int i=0; i<n_points; i++)
	{
		fe_poly->poly_space.compute(points[i],values,unused2,unused3,unused4,unused5);

		for (unsigned int j=0; j<n_dofs; j++)
		{
			phi_hat_matrix(j,i) = values[j][c];
		}
	}


	//print
	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_points; j++)
		{
			std::cout <<std::setw(15)<<phi_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;


	for (unsigned int i=0; i<n_dofs; i++)
	{
		for (unsigned int j=0; j<n_points; j++)
		{
			std::cout <<std::setw(15)<<phi_hat_matrix(i,j);
		}
		std::cout<<std::endl;
	}

	std::cout<<std::endl<<std::endl;

	phi_hat_matrix.gauss_jordan(); //inverse (phi_hat)

	//phi_matrix.mmult(C_matrix,phi_hat_matrix);


	for (int i=0; i<n_dofs; i++)
	{
		for (int j=0; j<n_points; j++)
		{
			Number temp = 0;
			for (int k=0; k<n_points;k++)
			{
					temp += (fe_poly->inverse_node_matrix(i,k))*phi_hat_matrix(k,j);
			}
			//std::cout <<std::setw(15)<<fe_poly->inverse_node_matrix(i,j);
			std::cout <<std::setw(15)<<temp;
		}
		std::cout<<std::endl;
	}

	//Compare C_matrix with fe_poly->inverse_node_matrix
#endif


	/////////////////Debug over

#if 0

		const unsigned int   rt_dofs_per_cell = fe_rt.dofs_per_cell;

		const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim> *fe_poly =
				dynamic_cast<const FE_PolyTensor<PolynomialsRaviartThomas<dim>,dim,dim>*>(&fe_rt);

		if (rt_dofs_per_cell != fe_poly->poly_space.n())
		{
			std::cout<<"Conceptual bug..Error " <<std::endl;
			return false;
		}

		std::vector<std::vector<Tensor<1,dim>>> values(n_q_points);
		std::vector<Tensor<2,dim>> unused2;
		std::vector<Tensor<3,dim>> unused3;
		std::vector<Tensor<4,dim>> unused4;
		std::vector<Tensor<5,dim>> unused5;

		for (unsigned int q=0; q<n_q_points; ++q)
		{
			//size = no. of dofs = no of tensor product polynomials = total no of basis functions
			values[q].resize(rt_dofs_per_cell);
			p = fe_quad.point(q);
			fe_poly->poly_space.compute(p,values[q],unused2,unused3,unused4,unused5);
		}

		mylog<<"Using FEPoly functions "<<std::endl;
		for (int c=0; c<n_components; c++)
		{
			mylog<<"For component number ==========="<<c<<std::endl;
			for (unsigned int i=0; i<rt_dofs_per_cell; ++i)
			{
				mylog<<std::setw(10)<<i;
			  for (unsigned int q=0; q<n_q_points; ++q)
		  	  {
			  	  mylog<<std::setw(20)<<values[q][i][c];
		  	  }
		  	  mylog<<std::endl;
			}
		}

		//Tensor product for evaluation of values = N3XN2XN1 where N1,N2,N3 are 1-d matrices as stored in shapeInfo

		mylog<<"Using ShapeInfo functions "<<std::endl;
		double outputMatrix[rt_dofs_per_cell][n_q_points];
		double bigMatrix[fe_degree+2][n_q_points_1d];
		double smallMatrix[fe_degree+2][n_q_points_1d];
		int yrows, ycols = n_q_points_1d;
		int xrows, xcols = n_q_points_1d;
		int row = 0, col = 0;

		const int x_dir=0;
		const int y_dir=1;
		const int z_dir=2;

		int c; //component

		//Although we could simply make them from shape values of any component, but for better verification
		//of code, lets make them separately during each component

		//For 0th component, X is bigMatrix and y is smallmatrix

		//make x and y matrices
		c = 0;
		xrows = fe_degree+2; //dof = fe_degree + 1
		yrows = fe_degree+1;

		for (int i=0; i<xrows; i++)
		{
			for (int j=0; j<xcols; j++)
			{
				bigMatrix[i][j] = shape_info.shape_values_vec[c][x_dir][i*n_q_points_1d+j][0];
				mylog<<std::setw(20)<<bigMatrix[i][j];
			}
			mylog<<std::endl;
		}

		for (int i=0; i<yrows; i++)
		{
			for (int j=0; j<ycols; j++)
			{
				smallMatrix[i][j] = shape_info.shape_values_vec[c][y_dir][i*n_q_points_1d+j][0];
				mylog<<std::setw(20)<<smallMatrix[i][j];
			}
			mylog<<std::endl;
		}

		//now evaluate
		double temp;

		for (int yi=0; yi<yrows; yi++)
		{
			for (int yj=0; yj<ycols; yj++)
			{
				temp = smallMatrix[yi][yj];
				for (int xi=0; xi<xrows; xi++)
				{
					for (int xj=0; xj<xcols; xj++)
					{
						col = yj*xcols+xj;
						row = yi*xrows+xi;
						outputMatrix[row][col] = temp * bigMatrix[xi][xj];
					}
				}
			}
		}

		mylog<<"Using Tensor Product of ShapeInfo "<<std::endl;
		//for (int c=0; c<n_components; c++)
		//{
		//	mylog<<"For component number ==========="<<c<<std::endl;
			for (unsigned int i=0; i<rt_dofs_per_cell; ++i)
			{
				mylog<<std::setw(10)<<i;
			  for (unsigned int q=0; q<n_q_points; ++q)
		  	  {
			  	  mylog<<std::setw(20)<<outputMatrix[i][q];
		  	  }
		  	  mylog<<std::endl;
			}
#endif

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


		}


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
    res = Test<double,3,3,1>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);
    res = Test<double,3,3,2>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);
    res = Test<double,3,3,3>(evaluate_values,evaluate_gradients,evaluate_hessians).run(debug);

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
