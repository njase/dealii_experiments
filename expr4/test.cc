#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/evaluation_kernels.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/matrix_free/evaluation_selector.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>


using namespace dealii;
using namespace dealii::internal;
using namespace std;

const int g_dim = 2, g_fe_degree = 1, g_n_q_points_1d = 4, g_n_components = 1;
//TBD: For n_components > 1, vector-valued problem has to be constructed
const int g_fe_degree_1c = g_fe_degree, g_fe_degree_2c = g_fe_degree, g_fe_degree_3c = g_fe_degree;
const int g_n_q_points_1d_1c = g_n_q_points_1d, g_n_q_points_1d_2c = g_n_q_points_1d, g_n_q_points_1d_3c = g_n_q_points_1d;
using Number = double;

//const bool evaluate_values = true, evaluate_gradients = false, evaluate_hessians = false;


////////////////// Traits to get n_components ////////////////
template <typename T, int dim>
struct get_n_comp
{
	static constexpr int n_components = 1;
};


template <int dim>
struct get_n_comp<FE_RaviartThomas<dim>,dim>
{
	static constexpr int n_components = dim+1;
};

template <int dim>
struct get_n_comp<FESystem<dim>, dim>
{
	static constexpr int n_components = g_n_components;
};


//TODO: use CPP style enum
typedef enum
{
  equals_fe_degree,

  fe_degree_plus_one,

  no_policy // Not designed yet !!

}QuadPolicy;

template <QuadPolicy T, int fe_degree>
struct get_quad_1d
{
	static constexpr int n_q_points_1d = fe_degree;
};

template <int fe_degree>
struct get_quad_1d<fe_degree_plus_one, fe_degree>
{
	static constexpr int n_q_points_1d = fe_degree+1;
};


template <typename FEType, int dim, int dir, int base_fe_degree, int n_components, int c>
struct get_FEData
{
	static constexpr int fe_degree_for_component = 0;
	static constexpr int max_fe_degree = 0;
};


template <int dim, int dir, int base_fe_degree, int n_components, int c>
struct get_FEData<FE_RaviartThomas<dim>, dim, dir, base_fe_degree, n_components, c>
{
	static constexpr int fe_degree_for_component = ((dir == c) ? base_fe_degree+1 : base_fe_degree+1);
	static constexpr int max_fe_degree = base_fe_degree+1;
};


////////////////////////////////////////
#if 0
template <typename FEType, typename QuadPolicy, int dim, int fe_degree, int n_components, typename Number>
struct SelectEvaluatorNew
{
	void evaluate()
	{
		if (1) //FEType is RaviartThomas or something
		{
			SelectEvaluatorAsymm<>
			  	  ::evaluate (*this->data, &this->values_dofs[0], this->values_quad,
			              this->gradients_quad, this->hessians_quad, this->scratch_data,
			              evaluate_values, evaluate_gradients, evaluate_hessians);
		}
		else
		{
			//Old style SelectEvaluator
			SelectEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number>
				  ::evaluate (*this->data, &this->values_dofs[0], this->values_quad,
				              this->gradients_quad, this->hessians_quad, this->scratch_data,
				              evaluate_values, evaluate_gradients, evaluate_hessians);
		}
	}
};

template <typename FEType, typename QuadPolicy, int dim, int fe_degree, typename Number >
class FEEvaluationNew : public FEEvaluationAccess<dim,get_n_comp<FEType,dim>::n_components,Number>
{
	using BaseClass =  FEEvaluationAccess<dim,get_n_comp<FEType,dim>::n_components,Number>;
	static const int n_components = BaseClass::n_components;

public:
	FEEvaluationNew():BaseClass (MatrixFree<dim,Number> (), fe_degree, 1, fe_degree, 10)
	{}

	void print_data()
	{
		cout<<"n_components = "<<BaseClass::n_components<<endl;
	}


	void evaluate (const bool evaluate_values,
	            const bool evaluate_gradients,
	            const bool evaluate_hessians)
	{
	  Assert (this->dof_values_initialized == true,
	          internal::ExcAccessToUninitializedField());
	  Assert(this->matrix_info != nullptr ||
	         this->mapped_geometry->is_initialized(), ExcNotInitialized());

	  SelectEvaluatorNew<FEType, QuadPolicy, dim, fe_degree, n_components, Number>::evaluate();


	}
};

#endif

//TODO: The shape_info should have info for all components, all directions
//Currently lets assume that for each component, shape fxns are same in all directions.
// Flexibility needs change in shape_info which is TBD

// @base_fe_degree : e.g. RT0 = 0, RT1 = 1
template <MatrixFreeFunctions::ElementType type, typename FEType, QuadPolicy q_policy,
		  int n_components, int dim, int base_fe_degree, typename Number>
struct FEEvaluationImplGen
{
	//Evaluate for all components
  static
  void evaluate (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> (&shape_info)[n_components],
                 VectorizedArray<Number> *values_dofs_actual[], //in
                 VectorizedArray<Number> *values_quad[], //out
                 VectorizedArray<Number> *gradients_quad[][dim], //out
                 VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2], //out
                 VectorizedArray<Number> *scratch_data, //temp, in
                 const bool               evaluate_values, //in
                 const bool               evaluate_gradients, //in
                 const bool               evaluate_hessians) //in
  {
      if (evaluate_values == false && evaluate_gradients == false && evaluate_hessians == false)
        return;

        const EvaluatorVariant variant =
          EvaluatorSelector<type,(base_fe_degree+get_quad_1d<q_policy,base_fe_degree>::n_q_points_1d>4)>::variant;


      //Here starts the static loop on n_components - Depends on shape_info changes

      for (int c = 0; c < n_components; c++)
      {
      //const int c = 0;
      const int max_fe_degree = get_FEData<FEType, dim, 0 /* any dir */, base_fe_degree, n_components, n_components-1 /* any component */>::max_fe_degree;
      const int max_n_q_points_1d = get_quad_1d<q_policy,max_fe_degree>::n_q_points_1d;

      typedef EvaluatorTensorProduct<variant, dim, max_fe_degree, max_n_q_points_1d,
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
      VectorizedArray<Number> *temp1;
      VectorizedArray<Number> *temp2;
      if (temp_size == 0)
        {
          temp1 = scratch_data;
          //temp2 = temp1 + std::max(Utilities::fixed_power<dim>(shape_info[c].fe_degree+1),
                                   //Utilities::fixed_power<dim>(shape_info[c].n_q_points_1d));
          temp2 = temp1 + Utilities::fixed_power<dim>(std::max(shape_info[c].fe_degree+1,shape_info[c].n_q_points_1d));
        }
      else
        {
          temp1 = scratch_data;
          temp2 = temp1 + temp_size;
        }

      VectorizedArray<Number> **values_dofs = values_dofs_actual;
#if 0
      if (type == MatrixFreeFunctions::truncated_tensor)
        {
    	  VectorizedArray<Number> *expanded_dof_values[n_components];
          values_dofs = expanded_dof_values;
          for (unsigned int c=0; c<n_components; ++c)
            expanded_dof_values[c] = scratch_data+2*(std::max(shape_info.dofs_per_cell,
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

      switch (dim)
        {
        case 1:
          //for (unsigned int c=0; c<n_components; c++)
           // {
              if (evaluate_values == true)
                eval.template values<0,true,false> (values_dofs[c], values_quad[c]);
              if (evaluate_gradients == true)
                eval.template gradients<0,true,false>(values_dofs[c], gradients_quad[c][0]);
              if (evaluate_hessians == true)
                eval.template hessians<0,true,false> (values_dofs[c], hessians_quad[c][0]);
            //}
          break;

        case 2:
          //for (unsigned int c=0; c<n_components; c++)
           // {
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
            //}
          break;

        case 3:
          //for (unsigned int c=0; c<n_components; c++)
           // {
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
            //}
          break;

        default:
          AssertThrow(false, ExcNotImplemented());
        }

#if 0
      // case additional dof for FE_Q_DG0: add values; gradients and second
      // derivatives evaluate to zero
      if (type == MatrixFreeFunctions::tensor_symmetric_plus_dg0 && evaluate_values)
        for (unsigned int c=0; c<n_components; ++c)
          for (unsigned int q=0; q<shape_info.n_q_points; ++q)
            values_quad[c][q] += values_dofs[c][shape_info.dofs_per_cell-1];
    }
#endif
  }//end of for loop
}

};

int main()
{
	//FEEvaluationNew<FE_RaviartThomas<2>, int, 2, 4, double> obj1;
	//obj1.print_data();

}
