#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/evaluation_kernels.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/matrix_free/evaluation_selector.h>

using namespace dealii;
using namespace dealii::internal;



template <MatrixFreeFunctions::ElementType type, int dim, 
			int fe_degree_1c, int fe_degree_2c, int fe_degree_3c, //fe degree for 3 coordinates
          int n_q_points_1d_1c, int n_q_points_1d_2c, int n_q_points_1d_3c, //quad points in 1-D for 3 coordinates
          int n_components, int c, typename Number> 
struct EvaluateAsymmetric
{ 
  static void
  evaluate_asymmetric (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info[], //in
              VectorizedArray<Number> *values_dofs_actual[], //in
              VectorizedArray<Number> *values_quad[], //out
              VectorizedArray<Number> *gradients_quad[][dim], //out
              VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2], //out
              VectorizedArray<Number> *scratch_data, //intermediate
              const bool               evaluate_values, //in
              const bool               evaluate_gradients, //in
              const bool               evaluate_hessians) //in;
  { 

  constexpr int fe_degree[] = {fe_degree_1c,fe_degree_2c,fe_degree_3c};
  constexpr int n_q_points_1d[] = {n_q_points_1d_1c, n_q_points_1d_2c, n_q_points_1d_3c};
  
  VectorizedArray<Number> *temp1;
  VectorizedArray<Number> *temp2; 


  VectorizedArray<Number> **values_dofs = values_dofs_actual;
  VectorizedArray<Number> *expanded_dof_values[n_components];
  
#if 0 //TBD
  if (type == MatrixFreeFunctions::truncated_tensor)
    {
  	//Move the above declaration here for clarity
  	//VectorizedArray<Number> *expanded_dof_values[n_components];
      values_dofs = expanded_dof_values;
      for (unsigned int c=0; c<n_components; ++c)
        expanded_dof_values[c] = scratch_data+2*(std::max(shape_info.dofs_per_component_on_cell,
                                                          shape_info.n_q_points)) +
                                 c*Utilities::fixed_power<dim>(shape_info.fe_degree+1);
      const int degree = fe_degree != -1 ? fe_degree : shape_info.fe_degree;
      unsigned int count_p = 0, count_q = 0;
      for (int i=0; i<(dim>2?degree+1:1); ++i)
        {
          for (int j=0; j<(dim>1?degree+1-i:1); ++j)
            {
              for (int k=0; k<degree+1-j-i; ++k, ++count_p, ++count_q)
                for (unsigned int c=0; c<n_components; ++c)
                  expanded_dof_values[c][count_q] = values_dofs_actual[c][count_p];
              for (int k=degree+1-j-i; k<degree+1; ++k, ++count_q)
                for (unsigned int c=0; c<n_components; ++c)
                  expanded_dof_values[c][count_q] = VectorizedArray<Number>();
            }
          for (int j=degree+1-i; j<degree+1; ++j)
            for (int k=0; k<degree+1; ++k, ++count_q)
              for (unsigned int c=0; c<n_components; ++c)
                expanded_dof_values[c][count_q] = VectorizedArray<Number>();
        }
      AssertDimension(count_q, Utilities::fixed_power<dim>(shape_info.fe_degree+1));
    }
#endif
  
  // These avoid compiler warnings; they are only used in sensible context but
  // compilers typically cannot detect when we access something like
  // gradients_quad[2] only for dim==3.
  const unsigned int d1 = dim>1?1:0;
  const unsigned int d2 = dim>2?2:0;
  const unsigned int d3 = dim>2?3:0;
  const unsigned int d4 = dim>2?4:0;
  const unsigned int d5 = dim>2?5:0;

//  for (unsigned int c = 0; c < n_components; c++)
//  {
	  const EvaluatorVariant variant =
	    EvaluatorSelector<type,(fe_degree[c]+n_q_points_1d[c]>4)>::variant;
	  typedef EvaluatorTensorProduct<variant, dim, fe_degree[c], n_q_points_1d[c],
	          VectorizedArray<Number> > Eval;
	  Eval eval (variant == evaluate_evenodd ? shape_info[c].shape_values_eo :
			     shape_info[c].shape_values,
	             variant == evaluate_evenodd ? shape_info[c].shape_gradients_eo :
	             shape_info[c].shape_gradients,
	             variant == evaluate_evenodd ? shape_info[c].shape_hessians_eo :
	             shape_info[c].shape_hessians,
	             shape_info[c].fe_degree,
	             shape_info[c].n_q_points_1d);
	  
	  const unsigned int temp_size = Eval::dofs_per_cell == numbers::invalid_unsigned_int ? 0
	                                 : (Eval::dofs_per_cell > Eval::n_q_points ?
	                                    Eval::dofs_per_cell : Eval::n_q_points);
	  
	  if (temp_size == 0)
	    {
	      temp1 = scratch_data;	      
	      temp2 = temp1 + Utilities::fixed_power<dim>(std::max(shape_info[c].fe_degree+1,
	                                                      shape_info[c].n_q_points_1d));	      
	    }
	  else
	    {
	      temp1 = scratch_data;
	      temp2 = temp1 + temp_size;
	    }
	  
	  switch(dim)
	  {
	  	  case 1:	        
	  		  if (evaluate_values == true)
	              eval.template values<0,true,false> (values_dofs[c], values_quad[c]);
	            if (evaluate_gradients == true)
	              eval.template gradients<0,true,false>(values_dofs[c], gradients_quad[c][0]);
	            if (evaluate_hessians == true)
	              eval.template hessians<0,true,false> (values_dofs[c], hessians_quad[c][0]);	          
	          break;
	          
	  	  case 2:
	          // grad x
	          if (evaluate_gradients == true)
	            {
	              eval.template gradients<0,true,false> (values_dofs[c], temp1);
	              eval.template values<1,true,false> (temp1, gradients_quad[c][0]);
	            }
	          if (evaluate_hessians == true)
	            {
	              // grad xy
	              if (evaluate_gradients == false)
	                eval.template gradients<0,true,false>(values_dofs[c], temp1);
	              eval.template gradients<1,true,false>  (temp1, hessians_quad[c][d1+d1]);

	              // grad xx
	              eval.template hessians<0,true,false>(values_dofs[c], temp1);
	              eval.template values<1,true,false>  (temp1, hessians_quad[c][0]);
	            }

	          // grad y
	          eval.template values<0,true,false> (values_dofs[c], temp1);
	          if (evaluate_gradients == true)
	            eval.template gradients<1,true,false> (temp1, gradients_quad[c][d1]);

	          // grad yy
	          if (evaluate_hessians == true)
	            eval.template hessians<1,true,false> (temp1, hessians_quad[c][d1]);

	          // val: can use values applied in x
	          if (evaluate_values == true)
	            eval.template values<1,true,false> (temp1, values_quad[c]);	  
	          
	          break;
	      
	  	  case 3:
	          if (evaluate_gradients == true)
	            {
	              // grad x
	              eval.template gradients<0,true,false> (values_dofs[c], temp1);
	              eval.template values<1,true,false> (temp1, temp2);
	              eval.template values<2,true,false> (temp2, gradients_quad[c][0]);
	            }

	          if (evaluate_hessians == true)
	            {
	              // grad xz
	              if (evaluate_gradients == false)
	                {
	                  eval.template gradients<0,true,false> (values_dofs[c], temp1);
	                  eval.template values<1,true,false> (temp1, temp2);
	                }
	              eval.template gradients<2,true,false> (temp2, hessians_quad[c][d4]);

	              // grad xy
	              eval.template gradients<1,true,false> (temp1, temp2);
	              eval.template values<2,true,false> (temp2, hessians_quad[c][d3]);

	              // grad xx
	              eval.template hessians<0,true,false>(values_dofs[c], temp1);
	              eval.template values<1,true,false>  (temp1, temp2);
	              eval.template values<2,true,false>  (temp2, hessians_quad[c][0]);
	            }

	          // grad y
	          eval.template values<0,true,false> (values_dofs[c], temp1);
	          if (evaluate_gradients == true)
	            {
	              eval.template gradients<1,true,false>(temp1, temp2);
	              eval.template values<2,true,false>   (temp2, gradients_quad[c][d1]);
	            }

	          if (evaluate_hessians == true)
	            {
	              // grad yz
	              if (evaluate_gradients == false)
	                eval.template gradients<1,true,false>(temp1, temp2);
	              eval.template gradients<2,true,false>  (temp2, hessians_quad[c][d5]);

	              // grad yy
	              eval.template hessians<1,true,false> (temp1, temp2);
	              eval.template values<2,true,false> (temp2, hessians_quad[c][d1]);
	            }

	          // grad z: can use the values applied in x direction stored in temp1
	          eval.template values<1,true,false> (temp1, temp2);
	          if (evaluate_gradients == true)
	            eval.template gradients<2,true,false> (temp2, gradients_quad[c][d2]);

	          // grad zz: can use the values applied in x and y direction stored
	          // in temp2
	          if (evaluate_hessians == true)
	            eval.template hessians<2,true,false>(temp2, hessians_quad[c][d2]);

	          // val: can use the values applied in x & y direction stored in temp2
	          if (evaluate_values == true)
	            eval.template values<2,true,false> (temp2, values_quad[c]);
	          
	          break;
	          
	  	default:
	  	      AssertThrow(false, ExcNotImplemented());
	  
	  }	  //end of switch	  
//  } //end of for

#if 0 //TBD
  // case additional dof for FE_Q_DG0: add values; gradients and second
  // derivatives evaluate to zero
  if (type == MatrixFreeFunctions::tensor_symmetric_plus_dg0 && evaluate_values)
    for (unsigned int c=0; c<n_components; ++c)
      for (unsigned int q=0; q<shape_info.n_q_points; ++q)
        values_quad[c][q] += values_dofs[c][shape_info.dofs_per_component_on_cell-1];
#endif

  //Call thyself, increase c by 1
  EvaluateAsymmetric<type, dim, fe_degree_1c, fe_degree_2c, fe_degree_3c,
    	  n_q_points_1d_1c, n_q_points_1d_2c, n_q_points_1d_3c,
    	  n_components, c+1, Number>::
    	  evaluate_asymmetric(shape_info, values_dofs_actual, values_quad,
  	            gradients_quad, hessians_quad, scratch_data,
  	            evaluate_values, evaluate_gradients, evaluate_hessians);

}
};



//partial specialization with c = n_components
template <MatrixFreeFunctions::ElementType type, int dim, 
			int fe_degree_1c, int fe_degree_2c, int fe_degree_3c, //fe degree for 3 coordinates
          int n_q_points_1d_1c, int n_q_points_1d_2c, int n_q_points_1d_3c, //quad points in 1-D for 3 coordinates
          int n_components, typename Number> 
struct EvaluateAsymmetric<type,dim,fe_degree_1c,fe_degree_2c,fe_degree_3c,
		n_q_points_1d_1c,n_q_points_1d_2c,n_q_points_1d_3c,
		n_components,n_components,Number>
{ 
  static void
  evaluate_asymmetric (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info[], //in
              VectorizedArray<Number> *values_dofs_actual[], //in
              VectorizedArray<Number> *values_quad[], //out
              VectorizedArray<Number> *gradients_quad[][dim], //out
              VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2], //out
              VectorizedArray<Number> *scratch_data, //intermediate
              const bool               evaluate_values, //in
              const bool               evaluate_gradients, //in
              const bool               evaluate_hessians) //in;
  {
	  //Do nothing	  
  }
  
};


/**
 * First attempt
 * 
 * @in: shape_info 2-D array of [n_components X n_q_points_1d_c1
 * 
 */
template <MatrixFreeFunctions::ElementType type, int dim, 
			int fe_degree_1c, int fe_degree_2c, int fe_degree_3c, //fe degree for 3 coordinates
          int n_q_points_1d_1c, int n_q_points_1d_2c, int n_q_points_1d_3c, //quad points in 1-D for 3 coordinates
          int n_components, typename Number>
void
evaluate_asymmetric (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info[], //in
            VectorizedArray<Number> *values_dofs_actual[], //in
            VectorizedArray<Number> *values_quad[], //out
            VectorizedArray<Number> *gradients_quad[][dim], //out
            VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2], //out
            VectorizedArray<Number> *scratch_data, //intermediate
            const bool               evaluate_values, //in
            const bool               evaluate_gradients, //in
            const bool               evaluate_hessians) //in
{
  if (evaluate_values == false && evaluate_gradients == false && evaluate_hessians == false)
  {
      Assert(evaluate_values == false && evaluate_gradients == false && evaluate_hessians == false,
             ExcInternalError());
    return;
  }
   
  EvaluateAsymmetric<type, dim, fe_degree_1c, fe_degree_2c, fe_degree_3c,
  	  n_q_points_1d_1c, n_q_points_1d_2c, n_q_points_1d_3c,
  	  n_components, 0, Number>::
  	  evaluate_asymmetric(shape_info, values_dofs_actual, values_quad,
	            gradients_quad, hessians_quad, scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);
  
}

int main()
{
	const int dim = 2, fe_degree = 1, n_q_points_1d = 2, n_components = 1;
	//TBD: For n_components > 1, vector-valued problem has to be constructed
	const int fe_degree_1c = 1, fe_degree_2c = 1, fe_degree_3c = 1;
	const int n_q_points_1d_1c = 2, n_q_points_1d_2c = 2, n_q_points_1d_3c = 2;
	using Number = double;
	
	///////////////////
	//in
	internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl;
	internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl[n_components];

	VectorizedArray<Number> *values_dofs_actual[n_components]; //in
	VectorizedArray<Number> *scratch_data; //in

	const bool evaluate_values = true, evaluate_gradients = false, evaluate_hessians = false;

	//out
	VectorizedArray<Number> *values_quad_old_impl[n_components],
							*values_quad_new_impl[n_components];
	VectorizedArray<Number> *gradients_quad_old_impl[n_components][dim],
							*gradients_quad_new_impl[n_components][dim];
	VectorizedArray<Number> *hessians_quad_old_impl[n_components][(dim*(dim+1))/2],
							*hessians_quad_new_impl[n_components][(dim*(dim+1))/2];

	
	/////For simplicity, we use arrays instead of allocating memory to above output variables
	const unsigned int first_selected_component = 0;
	FE_Q<dim> fe(fe_degree);
	QGauss<1> quad(n_q_points_1d);

	shape_info_old_impl.reinit(quad, fe, fe.component_to_base_index(first_selected_component).first);

	//Give known values to values_dofs_actual, i.e. to nodal_values
	const unsigned int dofs_cnt_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;
	const unsigned int n_array_elements = VectorizedArray<Number>::n_array_elements;
	const unsigned int n_elements = static_cast<int>(std::ceil(dofs_cnt_per_cell/static_cast<float>(n_array_elements)));

	using namespace std;
	cout<<endl<<"======= parameters ============"<<endl<<endl;

	cout<<"components = "<<n_components<<" ,dim = "<<dim<<"  ,fe_degree = "<<fe_degree<<"  ,n_q_points_1d = "<<n_q_points_1d<<endl;
	cout<<"DOF per cell = "<<dofs_cnt_per_cell<<" ,length of one VectorArray element = "<<n_array_elements<<endl;
	cout<<"Number of vectorArray elements needed = "<<n_elements<<endl;

	cout<<endl<<"======= ========== ============"<<endl;

	VectorizedArray<Number> nodal_values[n_elements];

	for (int d = 0; d < n_elements; d++)
	{
		for (int n = 0; n < n_array_elements; n++)
		{
			nodal_values[d][n] = n%n_array_elements + 1;
		}
	}

	VectorizedArray<Number> out_values_quad_old_impl[n_components][n_elements],
							out_values_quad_new_impl[n_components][n_elements];
	VectorizedArray<Number> out_gradients_quad_old_impl[n_components][dim][n_elements],
							out_gradients_quad_new_impl[n_components][dim][n_elements];
	VectorizedArray<Number> out_hessians_quad_old_impl[n_components][(dim*(dim+1))/2][n_elements],
							out_hessians_quad_new_impl[n_components][(dim*(dim+1))/2][n_elements];

	for (int c = 0; c < n_components; c++)
	{
		//Same shape info for each component = old impl
		shape_info_new_impl[c].reinit(quad, fe, fe.component_to_base_index(first_selected_component).first);

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

	//THIS IS TBD. So currently, allocate a large memory to scratch_data
	scratch_data = new VectorizedArray<Number> [1000];

#if 0
	std::cout<<"Before =============="<<std::endl;

	for (int c = 0; c < n_components; c++)
	{
		for (int d = 0; d < n_elements; d++)
		{
			for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
			{
				bool res = (std::abs(out_values_quad_old_impl[c][d][n] - out_values_quad_new_impl[c][d][n]) < 10e-4)?true:false;
				std::cout<<"(c,d) = ("<<out_values_quad_old_impl[c][d][n]<<", "<<out_values_quad_new_impl[c][d][n]<<")   AND  before = "<<res<<std::endl;
			}
		}
	}
#endif

	internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general, dim, 
							fe_degree, n_q_points_1d, n_components, Number>
	::evaluate(shape_info_old_impl, values_dofs_actual, values_quad_old_impl,
            gradients_quad_old_impl, hessians_quad_old_impl, scratch_data,
            evaluate_values, evaluate_gradients, evaluate_hessians);
	
	evaluate_asymmetric<internal::MatrixFreeFunctions::tensor_general, dim,
			fe_degree_1c,fe_degree_2c, fe_degree_3c,
			n_q_points_1d_1c, n_q_points_1d_2c, n_q_points_1d_3c,
			n_components, Number>
			(shape_info_new_impl, values_dofs_actual, values_quad_new_impl,
	            gradients_quad_new_impl, hessians_quad_new_impl, scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);

	//Compare output of both and tell result
	std::cout<<"Result =============="<<std::endl;

	for (int c = 0; c < n_components; c++)
	{
		for (int d = 0; d < n_elements; d++)
		{
			for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
			{
				bool res = (std::abs(out_values_quad_old_impl[c][d][n] - out_values_quad_new_impl[c][d][n]) < 10e-4)?true:false;
				std::cout<<"(old,new) = ("<<out_values_quad_old_impl[c][d][n]<<", "<<out_values_quad_new_impl[c][d][n]<<")   AND  result = "<<res<<std::endl;
			}
		}
	}
}
