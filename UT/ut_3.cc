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

template <int dim, int fe_degree, int n_q_points_1d, typename Number,
	  	    int direction, bool dof_to_quad, bool add, int fe_degree_other1>
inline
void
test_apply_anisotropic (const Number *shape_data,
         const Number in [],
         Number       out [])
{
  AssertIndexRange (direction, dim);
  const int mm     = dof_to_quad ? (fe_degree+1) : n_q_points_1d,
            nn     = dof_to_quad ? n_q_points_1d : (fe_degree+1),
            mm_other1 = dof_to_quad ? (fe_degree_other1+1) : n_q_points_1d;
            //mm_other2 = dof_to_quad ? (fe_degree_other2+1) : n_q_points_1d;

  const int n_blocks1 = (dim > 1 ? (direction > 0 ? nn : mm_other1) : 1);
  const int n_blocks2 = (dim > 2 ? (direction > 1 ? nn : mm) : 1); //FIXME TBD for dim=3
  const int stride    = Utilities::fixed_int_power<nn,direction>::value;

  for (int i2=0; i2<n_blocks2; ++i2)
    {
      for (int i1=0; i1<n_blocks1; ++i1)
        {
          for (int col=0; col<nn; ++col)
            {
              Number val0;
              if (dof_to_quad == true)
                val0 = shape_data[col];
              else
                val0 = shape_data[col*n_q_points_1d];
              Number res0 = val0 * in[0];
              for (int ind=1; ind<mm; ++ind)
                {
                  if (dof_to_quad == true)
                    val0 = shape_data[ind*n_q_points_1d+col];
                  else
                    val0 = shape_data[col*n_q_points_1d+ind];
                  res0 += val0 * in[stride*ind];
                }
              if (add == false)
                out[stride*col]  = res0;
              else
                out[stride*col] += res0;
            }

          // increment: in regular case, just go to the next point in
          // x-direction. If we are at the end of one chunk in x-dir, need
          // to jump over to the next layer in z-direction
          switch (direction)
            {
            case 0:
              in += mm;
              out += nn;
              break;
            case 1:
            case 2:
              ++in;
              ++out;
              break;
            default:
              Assert (false, ExcNotImplemented());
            }
        }
      if (direction == 1)
        {
          in += nn*(mm-1);
          out += nn*(nn-1);
        }
    }
}


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
#if 0
	AlignedVector<VecArr> X_shape_data(x_size);
	AlignedVector<VecArr> Y_shape_data(y_size);
	AlignedVector<VecArr> u(u_size);
	AlignedVector<VecArr> quad_out(out_size);
	AlignedVector<VecArr> temp_quad_out(1000);

	int t = 10;
	X_shape_data[0][0] = X_shape_data[0][1] = .6872;
	X_shape_data[1][0] = X_shape_data[1][1] = 0;
	X_shape_data[2][0] = X_shape_data[2][1] = -0.0872;
	X_shape_data[3][0] = X_shape_data[3][1] = 0.3999;
	X_shape_data[4][0] = X_shape_data[4][1] = 1;
	X_shape_data[5][0] = X_shape_data[5][1] = 0.3999;
	X_shape_data[6][0] = X_shape_data[6][1] = -0.0872;
	X_shape_data[7][0] = X_shape_data[7][1] = 0;
	X_shape_data[8][0] = X_shape_data[8][1] = 0.6872;

	Y_shape_data[0][0] = Y_shape_data[0][1] = 0.8872;
	Y_shape_data[1][0] = Y_shape_data[1][1] = 0.5;
	Y_shape_data[2][0] = Y_shape_data[2][1] = 0.11270;
	Y_shape_data[3][0] = Y_shape_data[3][1] = 0.11270;
	Y_shape_data[4][0] = Y_shape_data[4][1] = 0.5;
	Y_shape_data[5][0] = Y_shape_data[5][1] = 0.8872;
#endif

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
	test_apply_anisotropic<dim, fe_degree_y, n_q_points_1d, VecArr,0,true,false,fe_degree_x>(Y_shape_data, u,temp_quad_out);
	test_apply_anisotropic<dim, fe_degree_x, n_q_points_1d, VecArr,1,true,false,fe_degree_y>(X_shape_data, temp_quad_out,quad_out);
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
