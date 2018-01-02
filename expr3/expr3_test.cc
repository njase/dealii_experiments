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

const int g_dim = 2, g_fe_degree = 1, g_n_q_points_1d = g_fe_degree+1, g_n_components = 2;
const int g_base_fe_degree = g_fe_degree;

//TBD: For n_components > 1, vector-valued problem has to be constructed
//const int g_fe_degree_1c = g_fe_degree, g_fe_degree_2c = g_fe_degree, g_fe_degree_3c = g_fe_degree;
using Number = double;

const bool evaluate_values = true, evaluate_gradients = false, evaluate_hessians = false;

//template <typename Number, int n_components, int dim, int fe_degree, int n_q_points_1d, int base_fe_degree=fe
int main()
{
	//////////////////

	VectorizedArray<Number> *values_dofs_actual[g_n_components]; //in
	VectorizedArray<Number> *scratch_data; //in


	//Just allocate a large memory, actual size unimportant for our test
	scratch_data = new VectorizedArray<Number> [1000];


	//out
	VectorizedArray<Number> *values_quad_old_impl[g_n_components],
							*values_quad_new_impl[g_n_components];
	VectorizedArray<Number> *gradients_quad_old_impl[g_n_components][g_dim],
							*gradients_quad_new_impl[g_n_components][g_dim];
	VectorizedArray<Number> *hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2],
							*hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2];


	/////For simplicity in comparison, we use arrays instead of allocating memory to above output variables
	const unsigned int first_selected_component = 0;
	FE_Q<g_dim> fe_u(g_fe_degree);
	FESystem<g_dim>  fe (fe_u, g_n_components);
	QGauss<1> quad(g_n_q_points_1d);

	//in
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl(quad, fe, fe.component_to_base_index(first_selected_component).first);
    MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl(quad, fe, fe.component_to_base_index(first_selected_component).first,true);

	//FIXME:This count will change when the dofs for each component are different
	//Give known values to values_dofs_actual, i.e. to nodal_values
	const int dofs_cnt_per_cell = fe.n_dofs_per_cell ();
	const int quad_cnt_per_cell = Utilities::fixed_int_power<g_n_q_points_1d,g_dim>::value*g_n_components;

	const int n_array_elements = VectorizedArray<Number>::n_array_elements;
	const int n_dof_elements = 100;
	//const int n_eval_elements = (dofs_cnt_per_cell * quad_cnt_per_cell)/static_cast<float>(n_array_elements);
	const int n_eval_elements = (dofs_cnt_per_cell)/static_cast<float>(n_array_elements);


	using namespace std;
	cout<<endl<<"======= parameters ============"<<endl<<endl;

	cout<<"components = "<<g_n_components<<" ,dim = "<<g_dim<<"  ,fe_degree = "<<g_fe_degree<<"  ,n_q_points_1d = "<<g_n_q_points_1d<<endl;
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

	VectorizedArray<Number> out_values_quad_old_impl[g_n_components][n_eval_elements],
							out_values_quad_new_impl[g_n_components][n_eval_elements];
	VectorizedArray<Number> out_gradients_quad_old_impl[g_n_components][g_dim][n_eval_elements],
							out_gradients_quad_new_impl[g_n_components][g_dim][n_eval_elements];
	VectorizedArray<Number> out_hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2][n_eval_elements],
							out_hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2][n_eval_elements];


	shape_info_old_impl.element_type = internal::MatrixFreeFunctions::tensor_general;

	for (int c = 0; c < g_n_components; c++)
	{
		//Same nodal values for all the components
		values_dofs_actual[c] = nodal_values;

		values_quad_old_impl[c] = out_values_quad_old_impl[c];
		values_quad_new_impl[c] = out_values_quad_new_impl[c];

		for (int d = 0; d < g_dim; d++)
		{
			gradients_quad_old_impl[c][d] = out_gradients_quad_old_impl[c][d];
			gradients_quad_new_impl[c][d] = out_gradients_quad_new_impl[c][d];
		}

		for (int e = 0; e < (g_dim*(g_dim+1))/2; e++)
		{
			hessians_quad_old_impl[c][e] = out_hessians_quad_old_impl[c][e];
			hessians_quad_new_impl[c][e] = out_hessians_quad_new_impl[c][e];
		}
	}


    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
    		g_dim, g_fe_degree, g_n_q_points_1d, g_n_components, Number>
             ::evaluate(shape_info_old_impl, values_dofs_actual, values_quad_old_impl,
            		 gradients_quad_old_impl, hessians_quad_old_impl, scratch_data,
                        evaluate_values, evaluate_gradients, evaluate_hessians);


	internal::FEEvaluationImplGen<internal::MatrixFreeFunctions::tensor_general,
				FE_TaylorHood, g_n_q_points_1d, g_dim, g_base_fe_degree, Number>
           ::evaluate(shape_info_new_impl, values_dofs_actual, values_quad_new_impl,
        		   gradients_quad_new_impl, hessians_quad_new_impl, scratch_data,
                      evaluate_values, evaluate_gradients, evaluate_hessians);



	cout<<"Computation finished...printing results"<<endl;

	//Compare output of both and tell result
	std::cout<<"Result =============="<<std::endl;

	for (int c = 0; c < g_n_components; c++)
	{
		std::cout<<"============================================"<<std::endl;
		std::cout<<"Comparison result for component ("<<c<<")"<<std::endl;
		std::cout<<"============================================"<<std::endl;
		for (int d = 0; d < n_eval_elements; d++)
		{
			for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
			{
				bool res = (std::abs(out_values_quad_old_impl[c][d][n] - out_values_quad_new_impl[c][d][n]) < 10e-4)?true:false;
				std::cout<<"(old,new) = ("<<out_values_quad_old_impl[c][d][n]<<", "<<out_values_quad_new_impl[c][d][n]<<")   AND  result = "<<res<<std::endl;
			}
		}
	}


	delete[] scratch_data;
}

#if 0
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


/////////THIS IS ALL GOING IN WRONG DIRECTION///////////////////
///////// REDO ////////////////////////////////////

using namespace dealii;
using namespace dealii::internal;
using namespace std;

const int g_dim = 2, g_fe_degree = 1, g_n_q_points_1d = 4, g_n_components = 1;
//TBD: For n_components > 1, vector-valued problem has to be constructed
const int g_fe_degree_1c = g_fe_degree, g_fe_degree_2c = g_fe_degree, g_fe_degree_3c = g_fe_degree;
const int g_n_q_points_1d_1c = g_n_q_points_1d, g_n_q_points_1d_2c = g_n_q_points_1d, g_n_q_points_1d_3c = g_n_q_points_1d;
using Number = double;

//const bool evaluate_values = true, evaluate_gradients = false, evaluate_hessians = false;


#if 0
//This should likely fail to compile if unbalanced pair of fe_deg and quad poitns is provided
template <typename Number, int dim, int n_components_, int... Types>
class FEEvaluationNew : public FEEvaluationAccess<dim,n_components_,Number> //Do nothing for zero components
{
	typedef FEEvaluationAccess<dim,n_components_,Number> BaseClass;
	typedef Number                            number_type;
	typedef typename BaseClass::value_type    value_type;
	typedef typename BaseClass::gradient_type gradient_type;
	static const unsigned int dimension     = dim;
	static const unsigned int n_components  = n_components_;
};

#if 0
//This form may be needed if we want to keep that a user may also use FEEvaluationNew
//to indirectly obtain FEEvaluation, as earlier - with same ansatz in each dir
template <typename Number, int dim, int n_components, int fe_degree, int n_q_points_1d>
class FEEvaluationNew<Number, dim, fe_degree, n_q_points_1d> :
			public FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number>
{

};


//When only 1 component is left in recursion
template <typename Number, int dim, int fe_degree, int n_q_points_1d>
class FEEvaluationNew<Number, dim, fe_degree, n_q_points_1d> :
			public FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>
{


};
#endif


//When does the recursion stop? TBD
//This should always be user instansiated with comp = 0
template <typename Number, int dim, int n_components_, int comp, int fe_degree, int n_q_points_1d, int... Types>
class FEEvaluationNew<Number, dim, n_components_, comp, fe_degree, n_q_points_1d, Types...>
		: public FEEvaluationNew<Number, dim, n_components_, comp+1, Types...>
{
	//Now these static variables are
	static const unsigned int static_n_q_points    = Utilities::fixed_int_power<n_q_points_1d,dim>::value;
	static const unsigned int tensor_dofs_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;

public:

  FEEvaluationNew (const MatrixFree<dim,Number> &matrix_free,
                const unsigned int            fe_no   = 0,
                const unsigned int            quad_no = 0);

  FEEvaluationNew (const Mapping<dim>       &mapping,
                const FiniteElement<dim> &fe,
                const Quadrature<1>      &quadrature,
                const UpdateFlags         update_flags,
                const unsigned int        first_selected_component = 0);

  FEEvaluationNew (const FiniteElement<dim> &fe,
                const Quadrature<1>      &quadrature,
                const UpdateFlags         update_flags,
                const unsigned int        first_selected_component = 0);


  template <int n_components_other>
  FEEvaluationNew (const FiniteElement<dim> &fe,
                const FEEvaluationBase<dim,n_components_other,Number> &other,
                const unsigned int        first_selected_component = 0);


  FEEvaluationNew (const FEEvaluation &other);


  FEEvaluation &operator= (const FEEvaluation &other);


  void evaluate (const bool evaluate_values,
                 const bool evaluate_gradients,
                 const bool evaluate_hessians = false);


  void integrate (const bool integrate_values,
                  const bool integrate_gradients);


  Point<dim,VectorizedArray<Number> >
  quadrature_point (const unsigned int q_point) const;


  const unsigned int dofs_per_cell;


  const unsigned int n_q_points;

private:

  void check_template_arguments(const unsigned int fe_no,
                                const unsigned int first_selected_component);
};

#endif

void new_eval(const DoFHandler<g_dim> &dof, const MappingQ<g_dim> & mapping, const unsigned int dofs_per_cell,
				const unsigned int n_q_points, Number *values_dofs_actual[],
		        Number *values_quad[])
{


}

//Referred from matrix_free/assemble_matrix_01.cc
void old_eval(const DoFHandler<g_dim> &dof, const MappingQ<g_dim> & mapping, const unsigned int dofs_per_cell,
				const unsigned int n_q_points, Number *values_dofs_actual[],
		        Number *values_quad[])
{
	using namespace std;

	int c = 0; //Currently for one component..>1 is yet TBD


	FEEvaluation<g_dim,g_fe_degree,g_n_q_points_1d>
	    fe_eval (mapping, dof.get_fe(), QGauss<1>(g_n_q_points_1d),
	             update_values);

    typename DoFHandler<g_dim>::active_cell_iterator
	        cell = dof.begin_active(),
	        endc = dof.end();

    for (; cell!=endc; ++cell)
	{
    	fe_eval.reinit(cell);

        for (unsigned int i=0; i<dofs_per_cell; i += VectorizedArray<double>::n_array_elements)
        {
            const unsigned int n_items =
              i+VectorizedArray<double>::n_array_elements > dofs_per_cell ?
              (dofs_per_cell - i) : VectorizedArray<double>::n_array_elements;

            for (unsigned int j=0; j<dofs_per_cell; ++j)
              fe_eval.begin_dof_values()[j]  = VectorizedArray<double>();

            for (unsigned int v=0; v<n_items; ++v)
            {
            	fe_eval.begin_dof_values()[i+v][v] = values_dofs_actual[c][i+v];
            }
        }

        fe_eval.evaluate(true, false);

#if 0
            for (unsigned int q=0; q<n_q_points; ++q)
              {
                fe_eval.submit_value(10.*fe_eval.get_value(q), q);
                fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
              }
            fe_eval.integrate(true, true);
#endif

    	for (unsigned int q=0; q<n_q_points; q += VectorizedArray<double>::n_array_elements)
        {
            const unsigned int n_items =
              q+VectorizedArray<double>::n_array_elements > n_q_points ?
              (n_q_points - q) : VectorizedArray<double>::n_array_elements;

            for (unsigned int v=0; v<n_items; ++v)
            {
            	values_quad[c][q+v] = fe_eval.get_value(q)[v];
            }

        }
	}
}


int main()
{
	Triangulation<g_dim> triangulation;
	FE_Q<g_dim>          fe(g_fe_degree);
	DoFHandler<g_dim>    dof(triangulation);
	GridGenerator::hyper_cube (triangulation); //, -1, 1);
	//triangulation.refine_global (4);

	dof.distribute_dofs(fe);

	MappingQ<g_dim> mapping(g_fe_degree+1);

	QGauss<g_dim>  quadrature_formula(g_n_q_points_1d);
	const unsigned int   n_q_points    = quadrature_formula.size();
	const unsigned int   dofs_per_cell = dof.get_fe().dofs_per_cell;
	const int n_array_elements = VectorizedArray<Number>::n_array_elements;


	//in
	Number nodal_values[dofs_per_cell];
	Number *values_dofs_actual[g_n_components];

	//out
	/////For simplicity, we use arrays instead of allocating memory to above output variables
	Number *values_quad_old_impl[g_n_components],
							*values_quad_new_impl[g_n_components];
	Number *gradients_quad_old_impl[g_n_components][g_dim],
							*gradients_quad_new_impl[g_n_components][g_dim];
	Number *hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2],
							*hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2];
	Number out_values_quad_old_impl[g_n_components][n_q_points],
							out_values_quad_new_impl[g_n_components][n_q_points];
	Number out_gradients_quad_old_impl[g_n_components][g_dim][n_q_points],
							out_gradients_quad_new_impl[g_n_components][g_dim][n_q_points];
	Number out_hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2][n_q_points],
							out_hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2][n_q_points];

    for (unsigned int d=0; d<dofs_per_cell; d++)
    {
       	nodal_values[d] = d%n_array_elements + 1;
    }

	for (int c = 0; c < g_n_components; c++)
	{
		//Same nodal values for all the components
		values_dofs_actual[c] = nodal_values;

		values_quad_old_impl[c] = out_values_quad_old_impl[c];
		values_quad_new_impl[c] = out_values_quad_new_impl[c];

		for (int d = 0; d < g_dim; d++)
		{
			gradients_quad_old_impl[c][d] = out_gradients_quad_old_impl[c][d];
			gradients_quad_new_impl[c][d] = out_gradients_quad_new_impl[c][d];
		}

		for (int e = 0; e < (g_dim*(g_dim+1))/2; e++)
		{
			hessians_quad_old_impl[c][e] = out_hessians_quad_old_impl[c][e];
			hessians_quad_new_impl[c][e] = out_hessians_quad_new_impl[c][e];
		}
	}

	//take results from old and new implementation
	old_eval(dof,mapping, dofs_per_cell,n_q_points,values_dofs_actual,values_quad_old_impl);

	new_eval(dof,mapping, dofs_per_cell,n_q_points,values_dofs_actual,values_quad_new_impl);

	//Compare output of both and tell result
	cout<<endl<<"======= parameters ============"<<endl<<endl;

	cout<<"components = "<<g_n_components<<" ,dim = "<<g_dim<<"  ,fe_degree = "<<g_fe_degree<<"  ,n_q_points_1d = "<<g_n_q_points_1d<<endl;
	std::cout<<"(dofs_per_cell,n_q_points) = "<<dofs_per_cell<<", "<<n_q_points<<std::endl;
	cout<<"Length of one VectorArray element = "<<n_array_elements<<endl;

	cout<<endl<<"======= ========== ============"<<endl;

	std::cout<<"Result =============="<<std::endl;

	for (int c = 0; c < g_n_components; c++)
	{
		for (int d = 0; d < n_q_points; d++)
		{
				bool res = (std::abs(out_values_quad_old_impl[c][d] - out_values_quad_new_impl[c][d]) < 10e-4)?true:false;
				std::cout<<"(old,new) = ("<<out_values_quad_old_impl[c][d]<<", "<<out_values_quad_new_impl[c][d]<<")   AND  result = "<<res<<std::endl;
		}
	}

}


#if 0  //v0.2
template <MatrixFreeFunctions::ElementType type, int dim, int fe_degree,
            int n_q_points_1d, typename Number>
  struct FEEvaluationImplNew
  {
    static
    void evaluate (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &shape_info,
                   VectorizedArray<Number> *values_dofs_actual,
                   VectorizedArray<Number> *values_quad,
                   VectorizedArray<Number> *gradients_quad[dim],
                   VectorizedArray<Number> *hessians_quad[(dim*(dim+1))/2],
                   VectorizedArray<Number> *scratch_data,
                   const bool               evaluate_values,
                   const bool               evaluate_gradients,
                   const bool               evaluate_hessians)
    {
    if (evaluate_values == false && evaluate_gradients == false && evaluate_hessians == false)
      return;

    const EvaluatorVariant variant =
      EvaluatorSelector<type,(fe_degree+n_q_points_1d>4)>::variant;
    typedef EvaluatorTensorProduct<variant, dim, fe_degree, n_q_points_1d,
            VectorizedArray<Number> > Eval;
    Eval eval (variant == evaluate_evenodd ? shape_info.shape_values_eo :
               shape_info.shape_values,
               variant == evaluate_evenodd ? shape_info.shape_gradients_eo :
               shape_info.shape_gradients,
               variant == evaluate_evenodd ? shape_info.shape_hessians_eo :
               shape_info.shape_hessians,
               shape_info.fe_degree,
               shape_info.n_q_points_1d);

    const unsigned int temp_size = Eval::dofs_per_cell == numbers::invalid_unsigned_int ? 0
                                   : (Eval::dofs_per_cell > Eval::n_q_points ?
                                      Eval::dofs_per_cell : Eval::n_q_points);
    VectorizedArray<Number> *temp1;
    VectorizedArray<Number> *temp2;
    if (temp_size == 0)
      {
        temp1 = scratch_data;
        temp2 = temp1 + std::max(Utilities::fixed_power<dim>(shape_info.fe_degree+1),
                                 Utilities::fixed_power<dim>(shape_info.n_q_points_1d));
      }
    else
      {
        temp1 = scratch_data;
        temp2 = temp1 + temp_size;
      }

    VectorizedArray<Number> *values_dofs = values_dofs_actual;

#if 0 //TBD
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
          //{
            if (evaluate_values == true)
              eval.template values<0,true,false> (values_dofs, values_quad);
            if (evaluate_gradients == true)
              eval.template gradients<0,true,false>(values_dofs, gradients_quad[0]);
            if (evaluate_hessians == true)
              eval.template hessians<0,true,false> (values_dofs, hessians_quad[0]);
          //}
        break;

      case 2:
        //for (unsigned int c=0; c<n_components; c++)
          //{
            // grad x
            if (evaluate_gradients == true)
              {
                eval.template gradients<0,true,false> (values_dofs, temp1);
                eval.template values<1,true,false> (temp1, gradients_quad[0]);
              }
            if (evaluate_hessians == true)
              {
                // grad xy
                if (evaluate_gradients == false)
                  eval.template gradients<0,true,false>(values_dofs, temp1);
                eval.template gradients<1,true,false>  (temp1, hessians_quad[d1+d1]);

                // grad xx
                eval.template hessians<0,true,false>(values_dofs, temp1);
                eval.template values<1,true,false>  (temp1, hessians_quad[0]);
              }

            // grad y
            eval.template values<0,true,false> (values_dofs, temp1);
            if (evaluate_gradients == true)
              eval.template gradients<1,true,false> (temp1, gradients_quad[d1]);

            // grad yy
            if (evaluate_hessians == true)
              eval.template hessians<1,true,false> (temp1, hessians_quad[d1]);

            // val: can use values applied in x
            if (evaluate_values == true)
              eval.template values<1,true,false> (temp1, values_quad);
          //}
        break;

      case 3:
        //for (unsigned int c=0; c<n_components; c++)
        //  {
            if (evaluate_gradients == true)
              {
                // grad x
                eval.template gradients<0,true,false> (values_dofs, temp1);
                eval.template values<1,true,false> (temp1, temp2);
                eval.template values<2,true,false> (temp2, gradients_quad[0]);
              }

            if (evaluate_hessians == true)
              {
                // grad xz
                if (evaluate_gradients == false)
                  {
                    eval.template gradients<0,true,false> (values_dofs, temp1);
                    eval.template values<1,true,false> (temp1, temp2);
                  }
                eval.template gradients<2,true,false> (temp2, hessians_quad[d4]);

                // grad xy
                eval.template gradients<1,true,false> (temp1, temp2);
                eval.template values<2,true,false> (temp2, hessians_quad[d3]);

                // grad xx
                eval.template hessians<0,true,false>(values_dofs, temp1);
                eval.template values<1,true,false>  (temp1, temp2);
                eval.template values<2,true,false>  (temp2, hessians_quad[0]);
              }

            // grad y
            eval.template values<0,true,false> (values_dofs, temp1);
            if (evaluate_gradients == true)
              {
                eval.template gradients<1,true,false>(temp1, temp2);
                eval.template values<2,true,false>   (temp2, gradients_quad[d1]);
              }

            if (evaluate_hessians == true)
              {
                // grad yz
                if (evaluate_gradients == false)
                  eval.template gradients<1,true,false>(temp1, temp2);
                eval.template gradients<2,true,false>  (temp2, hessians_quad[d5]);

                // grad yy
                eval.template hessians<1,true,false> (temp1, temp2);
                eval.template values<2,true,false> (temp2, hessians_quad[d1]);
              }

            // grad z: can use the values applied in x direction stored in temp1
            eval.template values<1,true,false> (temp1, temp2);
            if (evaluate_gradients == true)
              eval.template gradients<2,true,false> (temp2, gradients_quad[d2]);

            // grad zz: can use the values applied in x and y direction stored
            // in temp2
            if (evaluate_hessians == true)
              eval.template hessians<2,true,false>(temp2, hessians_quad[d2]);

            // val: can use the values applied in x & y direction stored in temp2
            if (evaluate_values == true)
              eval.template values<2,true,false> (temp2, values_quad);
         // }
        break;

      default:
        AssertThrow(false, ExcNotImplemented());
      }

    // case additional dof for FE_Q_DG0: add values; gradients and second
    // derivatives evaluate to zero
    if (type == MatrixFreeFunctions::tensor_symmetric_plus_dg0 && evaluate_values)
    {
      //for (unsigned int c=0; c<n_components; ++c)
        for (unsigned int q=0; q<shape_info.n_q_points; ++q)
          values_quad[q] += values_dofs[shape_info.dofs_per_cell-1];
    }

    }//End of static function
};


template <int dim, int fe_degree, int n_q_points_1d, typename Number>
struct SelectEvaluatorNew
{

	 static void evaluate (const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &shape_info,
             VectorizedArray<Number> *values_dofs_actual,
             VectorizedArray<Number> *values_quad,
             VectorizedArray<Number> *gradients_quad[dim],
             VectorizedArray<Number> *hessians_quad[(dim*(dim+1))/2],
             VectorizedArray<Number> *scratch_data,
             const bool               evaluate_values,
             const bool               evaluate_gradients,
             const bool               evaluate_hessians)
{
  Assert(fe_degree>=0  && n_q_points_1d>0, ExcInternalError());


 if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_general)
    {
      FEEvaluationImplNew<internal::MatrixFreeFunctions::tensor_general,
               dim, fe_degree, n_q_points_1d, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else
    AssertThrow(false, ExcNotImplemented());
}
};


void new_evaluate(const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> (&shape_info)[g_n_components], //in
        VectorizedArray<Number> *values_dofs_actual[], //in
        VectorizedArray<Number> *values_quad[],
        VectorizedArray<Number> *gradients_quad[][g_dim],
        VectorizedArray<Number> *hessians_quad[][(g_dim*(g_dim+1))/2],
        VectorizedArray<Number> *scratch_data,
        const bool               evaluate_values,
        const bool               evaluate_gradients,
        const bool               evaluate_hessians)
{
	//Assume that the component wise fe_degree and n_1_points_1d are available at compile time
	//Will see later how this can be achieved in dealii
	constexpr int fe_degree_new[] = {g_fe_degree_1c,g_fe_degree_2c, g_fe_degree_3c};
	constexpr int n_q_points_1d_new[] = {g_n_q_points_1d_1c, g_n_q_points_1d_2c, g_n_q_points_1d_3c};


	//This calling for each component has to come from outside..
	//We don't use any compile time loop since then this class will have to
	//know about fe_degree and n_q_points_1d for all components.
	//And if it does, why are we doing all this!!

	SelectEvaluatorNew<g_dim, fe_degree_new[0], n_q_points_1d_new[0], Number>
	  ::evaluate (shape_info[0], values_dofs_actual[0], values_quad[0],
			  	  gradients_quad[0], hessians_quad[0], scratch_data,
	              evaluate_values, evaluate_gradients, evaluate_hessians);


#if 0
	FEEvaluationImplNew<internal::MatrixFreeFunctions::tensor_general, g_dim,
							fe_degree_new[0], n_q_points_1d_new[0], Number>
		::evaluate(shape_info[0], values_dofs_actual[0], values_quad[0],
				gradients_quad[0], hessians_quad[0], scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);

	//This needs vector values FE, TBD
	FEEvaluationImplNew<internal::MatrixFreeFunctions::tensor_general, dim,
								fe_degree[1], n_q_points_1d[1], Number>
		::evaluate(shape_info[1], values_dofs_actual[1], values_quad[1],
				gradients_quad[1], hessians_quad[1], scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);

	FEEvaluationImplNew<internal::MatrixFreeFunctions::tensor_general, dim,
								fe_degree[2], n_q_points_1d[2], Number>
		::evaluate(shape_info[2], values_dofs_actual[2], values_quad[2],
				gradients_quad[2], hessians_quad[2], scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);
#endif
}

void old_evaluate(const MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> &shape_info,
        VectorizedArray<Number> *values_dofs_actual[],
        VectorizedArray<Number> *values_quad[],
        VectorizedArray<Number> *gradients_quad[][g_dim],
        VectorizedArray<Number> *hessians_quad[][(g_dim*(g_dim+1))/2],
        VectorizedArray<Number> *scratch_data,
        const bool               evaluate_values,
        const bool               evaluate_gradients,
        const bool               evaluate_hessians)
{
	  SelectEvaluator<g_dim, g_fe_degree, g_n_q_points_1d, g_n_components, Number>
	  ::evaluate (shape_info, values_dofs_actual, values_quad,
			  	  gradients_quad, hessians_quad, scratch_data,
	              evaluate_values, evaluate_gradients, evaluate_hessians);
}




int main()
{
	//////////////////
	//in
	MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_old_impl;
	MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> shape_info_new_impl[g_n_components];

	VectorizedArray<Number> *values_dofs_actual[g_n_components]; //in
	VectorizedArray<Number> *scratch_data; //in


	//THIS IS TBD. So currently, allocate a large memory to scratch_data
	scratch_data = new VectorizedArray<Number> [1000];


	//out
	VectorizedArray<Number> *values_quad_old_impl[g_n_components],
							*values_quad_new_impl[g_n_components];
	VectorizedArray<Number> *gradients_quad_old_impl[g_n_components][g_dim],
							*gradients_quad_new_impl[g_n_components][g_dim];
	VectorizedArray<Number> *hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2],
							*hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2];


	/////For simplicity, we use arrays instead of allocating memory to above output variables
	const unsigned int first_selected_component = 0;
	FE_Q<g_dim> fe(g_fe_degree);
	QGauss<1> quad(g_n_q_points_1d);

	shape_info_old_impl.reinit(quad, fe, fe.component_to_base_index(first_selected_component).first);

	//TBD:This count will change when the dofs for each component are different
	//Give known values to values_dofs_actual, i.e. to nodal_values
	const int dofs_cnt_per_cell = Utilities::fixed_int_power<g_fe_degree+1,g_dim>::value;
	const int quad_cnt_per_cell = Utilities::fixed_int_power<g_n_q_points_1d,g_dim>::value;

	const int n_array_elements = VectorizedArray<Number>::n_array_elements;
	//const int n_dof_elements = static_cast<int>(std::ceil(dofs_cnt_per_cell/static_cast<float>(n_array_elements)));
	const int n_dof_elements = 10;
	const int n_eval_elements = (dofs_cnt_per_cell * quad_cnt_per_cell)/static_cast<float>(n_array_elements);


	using namespace std;
	cout<<endl<<"======= parameters ============"<<endl<<endl;

	cout<<"components = "<<g_n_components<<" ,dim = "<<g_dim<<"  ,fe_degree = "<<g_fe_degree<<"  ,n_q_points_1d = "<<g_n_q_points_1d<<endl;
	cout<<"DOF per cell = "<<dofs_cnt_per_cell<<", Quad points per cell = "<<quad_cnt_per_cell<<endl<<"Length of one VectorArray element = "<<n_array_elements<<endl;
	//cout<<"Number of vectorArray elements used for dofs = "<<n_dof_elements<<endl;
	cout<<"Number of vectorArray elements used for result = "<<n_eval_elements<<endl;

	cout<<endl<<"======= ========== ============"<<endl;

	VectorizedArray<Number> nodal_values[n_dof_elements];

	for (int d = 0; d < n_dof_elements; d++)
	{
		for (int n = 0; n < n_array_elements; n++)
		{
			nodal_values[d][n] = n%n_array_elements + 1;
		}
	}

	VectorizedArray<Number> out_values_quad_old_impl[g_n_components][n_eval_elements],
							out_values_quad_new_impl[g_n_components][n_eval_elements];
	VectorizedArray<Number> out_gradients_quad_old_impl[g_n_components][g_dim][n_eval_elements],
							out_gradients_quad_new_impl[g_n_components][g_dim][n_eval_elements];
	VectorizedArray<Number> out_hessians_quad_old_impl[g_n_components][(g_dim*(g_dim+1))/2][n_eval_elements],
							out_hessians_quad_new_impl[g_n_components][(g_dim*(g_dim+1))/2][n_eval_elements];


	shape_info_old_impl.element_type = internal::MatrixFreeFunctions::tensor_general;

	for (int c = 0; c < g_n_components; c++)
	{
		//Same shape info for each component = old impl
		shape_info_new_impl[c].reinit(quad, fe, fe.component_to_base_index(first_selected_component).first);
		shape_info_new_impl[c].element_type = internal::MatrixFreeFunctions::tensor_general;

		//Same nodal values for all the components
		values_dofs_actual[c] = nodal_values;

		values_quad_old_impl[c] = out_values_quad_old_impl[c];
		values_quad_new_impl[c] = out_values_quad_new_impl[c];

		for (int d = 0; d < g_dim; d++)
		{
			gradients_quad_old_impl[c][d] = out_gradients_quad_old_impl[c][d];
			gradients_quad_new_impl[c][d] = out_gradients_quad_new_impl[c][d];
		}

		for (int e = 0; e < (g_dim*(g_dim+1))/2; e++)
		{
			hessians_quad_old_impl[c][e] = out_hessians_quad_old_impl[c][e];
			hessians_quad_new_impl[c][e] = out_hessians_quad_new_impl[c][e];
		}
	}

	old_evaluate(shape_info_old_impl, values_dofs_actual, values_quad_old_impl,
	            gradients_quad_old_impl, hessians_quad_old_impl, scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);

	new_evaluate(shape_info_new_impl, values_dofs_actual, values_quad_new_impl,
	            gradients_quad_new_impl, hessians_quad_new_impl, scratch_data,
	            evaluate_values, evaluate_gradients, evaluate_hessians);

	cout<<"Computation finished...printing results"<<endl;

	//Compare output of both and tell result
	std::cout<<"Result =============="<<std::endl;

	for (int c = 0; c < g_n_components; c++)
	{
		for (int d = 0; d < n_eval_elements; d++)
		{
			for (int n = 0; n < n_array_elements; n++) //index in the vectorizedArray
			{
				bool res = (std::abs(out_values_quad_old_impl[c][d][n] - out_values_quad_new_impl[c][d][n]) < 10e-4)?true:false;
				std::cout<<"(old,new) = ("<<out_values_quad_old_impl[c][d][n]<<", "<<out_values_quad_new_impl[c][d][n]<<")   AND  result = "<<res<<std::endl;
			}
		}
	}


	delete[] scratch_data;
}
#endif
#endif
