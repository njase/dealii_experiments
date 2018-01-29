//Purpose: This is not to test any new functionality but just to


#include <deal.II/base/utilities.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>


#include <iostream>
#include <complex>
#include <vector>

using namespace dealii;
using namespace dealii::internal;
using namespace std;



int main(int argc, char *argv[])
{
	constexpr int dim = 2;
	constexpr int fe_degree_x=2, fe_degree_y=1;
	constexpr int n_q_points_1d = 3;
	constexpr int x_size = (fe_degree_x+1)*n_q_points_1d;
	constexpr int y_size = (fe_degree_y+1)*n_q_points_1d;
	constexpr int u_size = (fe_degree_x+1)*(fe_degree_y+1);
	constexpr int out_size = n_q_points_1d*n_q_points_1d;

	typedef double Number;
	typedef VectorizedArray<Number> VecArr;
	const int n_array_elements = VectorizedArray<Number>::n_array_elements;

	VecArr X_shape_data[x_size] = {{.6872,.6872},{0,0} ,{-0.0872,-0.0872},
			{0.3999,0.3999},{1,1},{0.3999,0.3999},
			{-0.0872,-0.0872},{0,0},{0.6872,0.6872}};
	VecArr Y_shape_data[y_size] = {{0.8872,0.8872}, {0.5,0.5}, {0.11270,0.11270},
			{0.11270,0.11270}, {0.5,0.5},{0.8872,0.8872}};
	VecArr u[u_size];
	VecArr quad_out[out_size];
	VecArr temp_quad_out[1000];

	int t = 10;

#if 0
	for (int i=0; i<x_size; i++)
	{
		for (int n=0; n<n_array_elements; n++)
			X_shape_data[i][n] = t;
		t++;
	}


	t = 100;
	for (int i=0; i<y_size; i++)
	{
		for (int n=0; n<n_array_elements; n++)
			Y_shape_data[i][n] = t;
		t++;
	}


	t=1;
	for (int i=0; i<u_size; i++)
	{
		for (int n=0; n<n_array_elements; n++)
			u[i][n] = t;
		t++;
	}
#endif

	for (int i=0; i<u_size; i++)
	{
		for (int n=0; n<n_array_elements; n++)
			u[i][n] = 0.0;
		t++;
	}
	u[1][0] = 1.0; u[1][1] = 1.0;


#if 0 //Works
	//perform [tr(Y) \otime tr(X)]u
	apply_anisotropic<dim, fe_degree_x, n_q_points_1d, VecArr,0,true,false,fe_degree_y>(X_shape_data, u,temp_quad_out);
	apply_anisotropic<dim, fe_degree_y, n_q_points_1d, VecArr,1,true,false,fe_degree_x>(Y_shape_data, temp_quad_out,quad_out);
#endif

//#if 0 //Works
	//perform [tr(X) \otime tr(Y)]u
	apply_anisotropic<dim, fe_degree_y, n_q_points_1d, VecArr,0,true,false,fe_degree_x>(Y_shape_data, u,temp_quad_out);
	apply_anisotropic<dim, fe_degree_x, n_q_points_1d, VecArr,1,true,false,fe_degree_y>(X_shape_data, temp_quad_out,quad_out);
//#endif

	std::cout<<std::endl<<"X matrix"<<std::endl;
	for (int i=0; i<(fe_degree_x+1); i++)
	{
		for (int j=0; j<n_q_points_1d; j++)
			std::cout<<setw(20)<<X_shape_data[i*n_q_points_1d+j][0];
		t++;
		std::cout<<std::endl;
	}

	std::cout<<"Y matrix"<<std::endl;
	for (int i=0; i<(fe_degree_y+1); i++)
	{
		for (int j=0; j<n_q_points_1d; j++)
			std::cout<<setw(20)<<Y_shape_data[i*n_q_points_1d+j][0];
		t++;
		std::cout<<std::endl;
	}

	std::cout<<"input u"<<std::endl;
	for (int i=0; i<u_size; i++)
	{
		std::cout<<"  "<<u[i][0];
	}

//#if 0
	std::cout<<std::endl<<"First output "<<std::endl;
	for (int i=0; i<out_size; i++)
	{
		std::cout<<setw(20)<<temp_quad_out[i][0];
	}
//#endif

	std::cout<<std::endl;
	std::cout<<std::endl;
	std::cout<<"Last output "<<std::endl;
	for (int i=0; i<out_size; i++)
	{
		std::cout<<"    "<<quad_out[i][0];
	}

	std::cout<<std::endl;

	return 0;
}
