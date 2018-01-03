//Purpose: To unit test the lexicographic ordering required in Shape Function evaluation
#include <deal.II/base/utilities.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>


#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/fe/fe_poly.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_dg0.h>

#include <iostream>
#include <complex>
#include <vector>

using namespace dealii;
using namespace dealii::internal;
using namespace std;

enum ElementType
{
  tensor_symmetric_collocation = 0,
  tensor_symmetric_hermite = 1,
  tensor_symmetric = 2,
  tensor_general = 3,
  truncated_tensor = 4,
  tensor_symmetric_plus_dg0 = 5
};

void print_vec(const std::vector<unsigned int> &vec)
{
	for (auto e:vec)
	{
		std::cout<<"  "<<e;
	}

	std::cout<<std::endl;
}

//Trimmed down version of original struct definition, just for UT
template <typename Number>
struct ShapeInfo
{
  /**
   * Empty constructor. Does nothing.
   */
	ShapeInfo () = default;
	
	
	template <int dim> void internal_reinit_scalar (const Quadrature<1> &quad,
	                 const FiniteElement<dim> &fe_in,
	                 const unsigned int base_element_number);
	
	template <int dim> void internal_reinit_vector (const Quadrature<1> &quad,
	                 const FiniteElement<dim> &fe_in,
	                 const unsigned int base_element_number);

	template <int dim> std::vector<unsigned int> lexicographic_renumber(const FiniteElement<dim> &fe, const unsigned int base_element_number);

	 	 ElementType element_type;
	 	 
	    	using ShapeVector = AlignedVector<Number>;
	    	using ShapeIterator = typename ShapeVector::iterator;

	      AlignedVector<Number> shape_values;
	      std::vector<std::array<ShapeIterator,3>> shape_values_vec;

	      AlignedVector<Number> shape_gradients;
	      std::vector<std::array<ShapeIterator,3>> shape_gradients_vec;

	      AlignedVector<Number> shape_hessians;
	      std::vector<std::array<ShapeIterator,3>> shape_hessians_vec;

	      std::vector<unsigned int> lexicographic_numbering;

	      unsigned int fe_degree;

	      unsigned int n_q_points_1d;

	      unsigned int n_q_points;

	      unsigned int dofs_per_component_on_cell;

	      unsigned int n_q_points_face;

	      unsigned int dofs_per_component_on_face;

	      bool nodal_at_cell_boundaries;

	      std::vector<ShapeVector> base_shape_values;
	      std::vector<ShapeVector> base_shape_gradients;
	      std::vector<ShapeVector> base_shape_hessians;


};

template <typename Number>
template <int dim>
std::vector<unsigned int>
ShapeInfo<Number>::lexicographic_renumber(const FiniteElement<dim> &fe_in, const unsigned int base_element_number)
{
	const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

    // renumber (this is necessary for FE_Q, for example, since there the
    // vertex DoFs come first, which is incompatible with the lexicographic
    // ordering necessary to apply tensor products efficiently)
    std::vector<unsigned int> scalar_lexicographic;
    {
      const FE_Poly<TensorProductPolynomials<dim>,dim,dim> *fe_poly =
        dynamic_cast<const FE_Poly<TensorProductPolynomials<dim>,dim,dim>*>(fe);

      const FE_Poly<TensorProductPolynomials<dim,Polynomials::
      PiecewisePolynomial<double> >,dim,dim> *fe_poly_piece =
        dynamic_cast<const FE_Poly<TensorProductPolynomials<dim,
        Polynomials::PiecewisePolynomial<double> >,dim,dim>*> (fe);

      const FE_DGP<dim> *fe_dgp = dynamic_cast<const FE_DGP<dim>*>(fe);

      const FE_Q_DG0<dim> *fe_q_dg0 = dynamic_cast<const FE_Q_DG0<dim>*>(fe);

      element_type = tensor_general;
      if (fe_poly != nullptr)
        scalar_lexicographic = fe_poly->get_poly_space_numbering_inverse();
      else if (fe_poly_piece != nullptr)
        scalar_lexicographic = fe_poly_piece->get_poly_space_numbering_inverse();
      else if (fe_dgp != nullptr)
        {
          scalar_lexicographic.resize(fe_dgp->dofs_per_cell);
          for (unsigned int i=0; i<fe_dgp->dofs_per_cell; ++i)
            scalar_lexicographic[i] = i;
          element_type = truncated_tensor;
        }
      else if (fe_q_dg0 != nullptr)
        {
          scalar_lexicographic = fe_q_dg0->get_poly_space_numbering_inverse();
          element_type = tensor_symmetric_plus_dg0;
        }
      else if (fe->dofs_per_cell == 0)
        {
          // FE_Nothing case -> nothing to do here
        }
      else
        Assert(false, ExcNotImplemented());

      // Finally store the renumbering into the member variable of this
      // class
      if (fe_in.n_components() == 1)
        lexicographic_numbering = scalar_lexicographic;
      else
        {
          // have more than one component, get the inverse
          // permutation, invert it, sort the components one after one,
          // and invert back
          std::vector<unsigned int> scalar_inv =
            Utilities::invert_permutation(scalar_lexicographic);
          std::vector<unsigned int> lexicographic(fe_in.dofs_per_cell,
                                                  numbers::invalid_unsigned_int);
          unsigned int components_before = 0;
          for (unsigned int e=0; e<base_element_number; ++e)
            components_before += fe_in.element_multiplicity(e);
          for (unsigned int comp=0;
               comp<fe_in.element_multiplicity(base_element_number); ++comp)
            for (unsigned int i=0; i<scalar_inv.size(); ++i)
              lexicographic[fe_in.component_to_system_index(comp+components_before,i)]
                = scalar_inv.size () * comp + scalar_inv[i];

          // invert numbering again. Need to do it manually because we might
          // have undefined blocks
          lexicographic_numbering.resize(fe_in.element_multiplicity(base_element_number)*fe->dofs_per_cell, numbers::invalid_unsigned_int);
          for (unsigned int i=0; i<lexicographic.size(); ++i)
            if (lexicographic[i] != numbers::invalid_unsigned_int)
              {
                AssertIndexRange(lexicographic[i],
                                 lexicographic_numbering.size());
                lexicographic_numbering[lexicographic[i]] = i;
              }
        }
    }

    return scalar_lexicographic;
}

template <typename Number>
template <int dim>
void ShapeInfo<Number>::internal_reinit_vector (const Quadrature<1> &quad,
             const FiniteElement<dim> &fe_in,
             const unsigned int base_element_number)
{
	unsigned int vector_n_components = fe_in.n_components();
	if (vector_n_components > 3 || dim > 3)
		Assert (false, ExcNotImplemented());


	//limited support currently
	enum class FEName { FE_Unknown=0, FE_RT=1, FE_Q_TP=2 };
	FEName fe_name = FEName::FE_Unknown;

	typedef std::pair<unsigned int /*comp*/, unsigned int /* dim*/> mindex;
	std::map<mindex, unsigned int> index_map;

	unsigned int base_values_count = 0;
	std::array<unsigned int, 2> n_dofs_1d; //not more than 2 distinct values in case of RT

    /*
     * Algo
     * Using FEType, identify number of components
     * utilize the reinit logic of ShapeInfo (restructure it) to evaluate values, quad and hessians
     *  for the required 1-D quad points and 1-D basis functions as many times as required for the particular
     *  FEType. Store the results in basic_shape_values
     * perform reinit for all the components as:
	 *   Mix and match results from basic_shape_values to shape_values_component as needed
     */

	//Find out type of FE as RT or from 1-D tensor product based
	const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

	//store only one base_shape_value with all directions pointing to it
	//This is true for FE_Q vector valued and will be updated for RT
	for (int c=0; c<vector_n_components; c++)
		for (int d=0; d<dim; d++)
			index_map[mindex(c,d)] = 0;

	if (dynamic_cast<const FE_RaviartThomas<dim> *>(fe))
	{
		if (dim != vector_n_components)
			Assert (false, ExcNotImplemented());

		fe_name = FEName::FE_RT;

		base_values_count = 2;
		n_dofs_1d[0] = fe->degree+1; //For Qk+1
		n_dofs_1d[1] = fe->degree; //For Qk

		if (1 == vector_n_components)
		{
			Assert (false, ExcNotImplemented());
		}
		else if (2 == vector_n_components)
		{
			index_map[mindex(1,1)] = 1;
			index_map[mindex(2,1)] = 0;
		}
		else
		{
			index_map[mindex(1,1)] = 1;
			index_map[mindex(2,2)] = 1;
			index_map[mindex(3,3)] = 1;
		}
	}
	else if(dynamic_cast<const FE_Q<dim> *>(fe))
	{
		base_values_count = 1;
		n_dofs_1d[0] = fe->degree+1;

		fe_name = FEName::FE_Q_TP;
	}
	else
	{
		Assert (false, ExcNotImplemented());
	}


    fe_degree = fe->degree; //Note that for RT, this is max degree across all components/dimensions
    n_q_points_1d = quad.size();

    std::vector<unsigned int> scalar_lexicographic;
    Point<dim> unit_point;

    if (FEName::FE_Q_TP == fe_name)
    {
        Assert(fe->n_components() == 1,ExcMessage("Expected a scalar element"));
        // find numbering to lexicographic
        scalar_lexicographic = lexicographic_renumber(fe_in, base_element_number);

        // to evaluate 1D polynomials, evaluate along the line with the first
        // unit support point, assuming that fe.shape_value(0,unit_point) ==
        // 1. otherwise, need other entry point (e.g. generating a 1D element
        // by reading the name, as done before r29356)
        if (fe->has_support_points())
          unit_point = fe->get_unit_support_points()[scalar_lexicographic[0]];
        Assert(fe->dofs_per_cell == 0 ||
               std::abs(fe->shape_value(scalar_lexicographic[0],
                                        unit_point)-1) < 1e-13,
               ExcInternalError("Could not decode 1D shape functions for the "
                                "element " + fe->get_name()));
    }

    n_q_points      = Utilities::fixed_power<dim>(n_q_points_1d);

    //For tensor product FE like Lagrangian or RT, this division is exact
    dofs_per_component_on_cell = fe_in.dofs_per_cell/vector_n_components;

    this->base_shape_gradients.resize(base_values_count);
    this->base_shape_values.resize(base_values_count);
    this->base_shape_hessians.resize(base_values_count);

    for (int j=0; j<base_values_count; j++)
    {
    	const unsigned int array_size = n_dofs_1d[j]*n_q_points_1d;

    	this->base_shape_gradients[j].resize_fast (array_size);
    	this->base_shape_values[j].resize_fast (array_size);
    	this->base_shape_hessians[j].resize_fast (array_size);

    	FE_Q<1> temp_fe(n_dofs_1d[j]-1); //fe_degree = dofs_1d-1
    	std::vector<unsigned int >scalar_lexicographic_temp = temp_fe.get_poly_space_numbering_inverse();

    	for (unsigned int i=0; i<n_dofs_1d[j]; ++i)
    	{
    		// need to reorder from hierarchical to lexicographic to get the
    		// DoFs correct
    		const unsigned int my_i = scalar_lexicographic_temp[i];
    		for (unsigned int q=0; q<n_q_points_1d; ++q)
    		{
    			Point<1> q_point = quad.get_points()[q];

    			base_shape_values[j][i*n_q_points_1d+q] = temp_fe.shape_value(my_i,q_point);
    			base_shape_gradients[j][i*n_q_points_1d+q] = temp_fe.shape_grad(my_i,q_point)[0];
    			base_shape_hessians[j][i*n_q_points_1d+q] = temp_fe.shape_grad_grad(my_i,q_point)[0][0];
    		}
    	}
    }

    shape_values_vec.resize(vector_n_components);
	shape_gradients_vec.resize(vector_n_components);
    shape_hessians_vec.resize(vector_n_components);

	for (int c=0; c<vector_n_components; c++)
    {
    	for (int d=0; d<dim; d++)
    	{
		  	shape_gradients_vec[c][d] = this->base_shape_gradients[index_map[mindex(c,d)]].begin();
		   	shape_values_vec[c][d] = this->base_shape_values[index_map[mindex(c,d)]].begin();
	    	shape_hessians_vec[c][d] = this->base_shape_hessians[index_map[mindex(c,d)]].begin();
    	}
    }
}

template <typename Number>
template <int dim>
void ShapeInfo<Number>::internal_reinit_scalar (const Quadrature<1> &quad,
             const FiniteElement<dim> &fe_in,
             const unsigned int base_element_number)
{
    const FiniteElement<dim> *fe = &fe_in.base_element(base_element_number);

    Assert (fe->n_components() == 1,
            ExcMessage("FEEvaluation only works for scalar finite elements."));

    fe_degree = fe->degree;
    n_q_points_1d = quad.size();

    const unsigned int n_dofs_1d = std::min(fe->dofs_per_cell, fe_degree+1);

    // renumber (this is necessary for FE_Q, for example, since there the
    // vertex DoFs come first, which is incompatible with the lexicographic
    // ordering necessary to apply tensor products efficiently)
    std::vector<unsigned int> scalar_lexicographic;
    Point<dim> unit_point;

    // find numbering to lexicographic
    Assert(fe->n_components() == 1,
           ExcMessage("Expected a scalar element"));

    scalar_lexicographic = lexicographic_renumber(fe_in,base_element_number);
    // to evaluate 1D polynomials, evaluate along the line with the first
    // unit support point, assuming that fe.shape_value(0,unit_point) ==
    // 1. otherwise, need other entry point (e.g. generating a 1D element
    // by reading the name, as done before r29356)
    if (fe->has_support_points())
      unit_point = fe->get_unit_support_points()[scalar_lexicographic[0]];
    Assert(fe->dofs_per_cell == 0 ||
           std::abs(fe->shape_value(scalar_lexicographic[0],
                                    unit_point)-1) < 1e-13,
           ExcInternalError("Could not decode 1D shape functions for the "
                            "element " + fe->get_name()));


    n_q_points      = Utilities::fixed_power<dim>(n_q_points_1d);
    n_q_points_face = dim>1?Utilities::fixed_power<dim-1>(n_q_points_1d):1;
    dofs_per_component_on_cell = fe->dofs_per_cell;
    dofs_per_component_on_face = dim>1?Utilities::fixed_power<dim-1>(fe_degree+1):1;

    const unsigned int array_size = n_dofs_1d*n_q_points_1d;

    this->shape_gradients.resize_fast (array_size);
    this->shape_values.resize_fast (array_size);
    this->shape_hessians.resize_fast (array_size);

#if 0
    this->shape_data_on_face[0].resize(3*n_dofs_1d);
    this->shape_data_on_face[1].resize(3*n_dofs_1d);
    this->values_within_subface[0].resize(array_size);
    this->values_within_subface[1].resize(array_size);
    this->gradients_within_subface[0].resize(array_size);
    this->gradients_within_subface[1].resize(array_size);
    this->hessians_within_subface[0].resize(array_size);
    this->hessians_within_subface[1].resize(array_size);
#endif

    for (unsigned int i=0; i<n_dofs_1d; ++i)
      {
        // need to reorder from hierarchical to lexicographic to get the
        // DoFs correct
        const unsigned int my_i = scalar_lexicographic[i];
        for (unsigned int q=0; q<n_q_points_1d; ++q)
          {
            Point<dim> q_point = unit_point;
            q_point[0] = quad.get_points()[q][0];

            shape_values   [i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
            shape_gradients[i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
            shape_hessians [i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];

#if 0
            // evaluate basis functions on the two 1D subfaces (i.e., at the
            // positions divided by one half and shifted by one half,
            // respectively)
            q_point[0] *= 0.5;
            values_within_subface[0][i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
            gradients_within_subface[0][i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
            hessians_within_subface[0][i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];
            q_point[0] += 0.5;
            values_within_subface[1][i*n_q_points_1d+q] = fe->shape_value(my_i,q_point);
            gradients_within_subface[1][i*n_q_points_1d+q] = fe->shape_grad(my_i,q_point)[0];
            hessians_within_subface[1][i*n_q_points_1d+q] = fe->shape_grad_grad(my_i,q_point)[0][0];
#endif
          }
#if 0
        // evaluate basis functions on the 1D faces, i.e., in zero and one
        Point<dim> q_point = unit_point;
        q_point[0] = 0;
        this->shape_data_on_face[0][i] = fe->shape_value(my_i,q_point);
        this->shape_data_on_face[0][i+n_dofs_1d] = fe->shape_grad(my_i,q_point)[0];
        this->shape_data_on_face[0][i+2*n_dofs_1d] = fe->shape_grad_grad(my_i,q_point)[0][0];
        q_point[0] = 1;
        this->shape_data_on_face[1][i] = fe->shape_value(my_i,q_point);
        this->shape_data_on_face[1][i+n_dofs_1d] = fe->shape_grad(my_i,q_point)[0];
        this->shape_data_on_face[1][i+2*n_dofs_1d] = fe->shape_grad_grad(my_i,q_point)[0][0];
#endif
      }

#if 0
    // get gradient and Hessian transformation matrix for the polynomial
    // space associated with the quadrature rule (collocation space)
    {
      const unsigned int stride = (n_q_points_1d+1)/2;
      shape_gradients_collocation_eo.resize(n_q_points_1d*stride);
      shape_hessians_collocation_eo.resize(n_q_points_1d*stride);
      FE_DGQArbitraryNodes<1> fe(quad.get_points());
      for (unsigned int i=0; i<n_q_points_1d/2; ++i)
        for (unsigned int q=0; q<stride; ++q)
          {
            shape_gradients_collocation_eo[i*stride+q] =
              0.5* (fe.shape_grad(i, quad.get_points()[q])[0] +
                    fe.shape_grad(i, quad.get_points()[n_q_points_1d-1-q])[0]);
            shape_gradients_collocation_eo[(n_q_points_1d-1-i)*stride+q] =
              0.5* (fe.shape_grad(i, quad.get_points()[q])[0] -
                    fe.shape_grad(i, quad.get_points()[n_q_points_1d-1-q])[0]);
            shape_hessians_collocation_eo[i*stride+q] =
              0.5* (fe.shape_grad_grad(i, quad.get_points()[q])[0][0] +
                    fe.shape_grad_grad(i, quad.get_points()[n_q_points_1d-1-q])[0][0]);
            shape_hessians_collocation_eo[(n_q_points_1d-1-i)*stride+q] =
              0.5* (fe.shape_grad_grad(i, quad.get_points()[q])[0][0] -
                    fe.shape_grad_grad(i, quad.get_points()[n_q_points_1d-1-q])[0][0]);
          }
      if (n_q_points_1d % 2 == 1)
        for (unsigned int q=0; q<stride; ++q)
          {
            shape_gradients_collocation_eo[n_q_points_1d/2*stride+q] =
              fe.shape_grad(n_q_points_1d/2, quad.get_points()[q])[0];
            shape_hessians_collocation_eo[n_q_points_1d/2*stride+q] =
              fe.shape_grad_grad(n_q_points_1d/2, quad.get_points()[q])[0][0];
          }
    }

    if (element_type == tensor_general &&
        check_1d_shapes_symmetric(n_q_points_1d))
      {
        if (check_1d_shapes_collocation())
          element_type = tensor_symmetric_collocation;
        else
          element_type = tensor_symmetric;
        if (n_dofs_1d > 3 && element_type == tensor_symmetric)
          {
            // check if we are a Hermite type
            element_type = tensor_symmetric_hermite;
            for (unsigned int i=1; i<n_dofs_1d; ++i)
              if (std::abs(get_first_array_element(shape_data_on_face[0][i])) > 1e-12)
                element_type = tensor_symmetric;
            for (unsigned int i=2; i<n_dofs_1d; ++i)
              if (std::abs(get_first_array_element(shape_data_on_face[0][n_dofs_1d+i])) > 1e-12)
                element_type = tensor_symmetric;
          }
      }
    else if (element_type == tensor_symmetric_plus_dg0)
      check_1d_shapes_symmetric(n_q_points_1d);

    nodal_at_cell_boundaries = true;
    for (unsigned int i=1; i<n_dofs_1d; ++i)
      if (std::abs(get_first_array_element(shape_data_on_face[0][i])) > 1e-13 ||
          std::abs(get_first_array_element(shape_data_on_face[1][i-1])) > 1e-13)
        nodal_at_cell_boundaries = false;

    if (nodal_at_cell_boundaries == true)
      {
        face_to_cell_index_nodal.reinit(GeometryInfo<dim>::faces_per_cell,
                                        dofs_per_component_on_face);
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const unsigned int direction = f/2;
            const unsigned int stride = direction < dim-1 ? (fe_degree+1) : 1;
            int shift = 1;
            for (unsigned int d=0; d<direction; ++d)
              shift *= fe_degree+1;
            const unsigned int offset = (f%2)*fe_degree*shift;

            if (direction == 0 || direction == dim-1)
              for (unsigned int i=0; i<dofs_per_component_on_face; ++i)
                face_to_cell_index_nodal(f,i) = offset + i*stride;
            else
              // local coordinate system on faces 2 and 3 is zx in
              // deal.II, not xz as expected for tensor products -> adjust
              // that here
              for (unsigned int j=0; j<=fe_degree; ++j)
                for (unsigned int i=0; i<=fe_degree; ++i)
                  {
                    const unsigned int ind = offset + j*dofs_per_component_on_face + i;
                    AssertIndexRange(ind, dofs_per_component_on_cell);
                    const unsigned int l = i*(fe_degree+1)+j;
                    face_to_cell_index_nodal(f,l) = ind;
                  }
          }
      }

    if (element_type == tensor_symmetric_hermite)
      {
        face_to_cell_index_hermite.reinit(GeometryInfo<dim>::faces_per_cell,
                                          2*dofs_per_component_on_face);
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const unsigned int direction = f/2;
            const unsigned int stride = direction < dim-1 ? (fe_degree+1) : 1;
            int shift = 1;
            for (unsigned int d=0; d<direction; ++d)
              shift *= fe_degree+1;
            const unsigned int offset = (f%2)*fe_degree*shift;
            if (f%2 == 1)
              shift = -shift;

            if (direction == 0 || direction == dim-1)
              for (unsigned int i=0; i<dofs_per_component_on_face; ++i)
                {
                  face_to_cell_index_hermite(f,2*i) = offset + i*stride;
                  face_to_cell_index_hermite(f,2*i+1) = offset + i*stride + shift;
                }
            else
              // local coordinate system on faces 2 and 3 is zx in
              // deal.II, not xz as expected for tensor products -> adjust
              // that here
              for (unsigned int j=0; j<=fe_degree; ++j)
                for (unsigned int i=0; i<=fe_degree; ++i)
                  {
                    const unsigned int ind = offset + j*dofs_per_component_on_face + i;
                    AssertIndexRange(ind, dofs_per_component_on_cell);
                    const unsigned int l = i*(fe_degree+1)+j;
                    face_to_cell_index_hermite(f,2*l) = ind;
                    face_to_cell_index_hermite(f,2*l+1) = ind+shift;
                  }
          }
      }
#endif
}

template <int n_components, int dim, int fe_degree>
bool test ()
{
	bool res = true;
	constexpr int n_q_points_1d = fe_degree+1;
	int n_dofs_1d = 0, array_size = 0;

	FE_Q<dim> fe_u(fe_degree);
	FESystem<dim>  fe (fe_u, n_components);
	QGauss<1> quad(n_q_points_1d);

	ShapeInfo<double> obj;

	obj.internal_reinit_scalar (quad, fe, 0);
	obj.internal_reinit_vector (quad, fe, 0);

	n_dofs_1d = fe_degree + 1;
	array_size = n_dofs_1d*n_q_points_1d;

	for (int c=0; c<n_components; c++)
		for (int d=0; d<dim; d++)
			for (int i=0; i<array_size; i++)
			{
#if 0
				std::cout<<"(orig, New) = ("<<obj.shape_values[i]<<", "<<obj.shape_values_vec[c][d][i]<<") ";
				std::cout<<"("<<obj.shape_gradients[i]<<", "<<obj.shape_gradients_vec[c][d][i]<<") ";
				std::cout<<"("<<obj.shape_hessians[i]<<", "<<obj.shape_hessians_vec[c][d][i]<<") "<<std::endl;
#endif

				if (obj.shape_values_vec[c][d][i] != obj.shape_values[i])
					res = false;
				if (obj.shape_gradients_vec[c][d][i] != obj.shape_gradients[i])
					res = false;
				if (obj.shape_hessians_vec[c][d][i] != obj.shape_hessians[i])
					res = false;
			}

	std::cout<<"Test for (n_comp, dim, fe_deg) = ("<<n_components<<","<<dim<<","<<fe_degree<<"), result = "<<(res==true?"Pass":"Fail")<<std::endl;

	return res;

}



   int main()
   {
	   //comp, dim, deg,

	   //tests for 1-D
	   test<1,1,1>();
	   test<1,1,2>();
	   test<1,1,3>();

	   test<2,1,1>();
	   test<2,1,2>();
	   test<2,1,3>();

	   test<3,1,1>();
	   test<3,1,2>();
	   test<3,1,3>();


	   //tests for 2-D
	   test<1,2,1>();
	   test<1,2,2>();
	   test<1,2,3>();

	   test<2,2,1>();
	   test<2,2,2>();
	   test<2,2,3>();

	   test<3,2,1>();
	   test<3,2,2>();
	   test<3,2,3>();

	   //tests for 3-D
	   test<1,3,1>();
	   test<1,3,2>();
	   test<1,3,3>();

	   test<2,3,1>();
	   test<2,3,2>();
	   test<2,3,3>();

	   test<3,3,1>();
	   test<3,3,2>();
	   test<3,3,3>();

	   return 0;
   }
    
