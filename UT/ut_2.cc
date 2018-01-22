//Purpose: This is not to test any new functionality but just to confirm
// my understanding of generalized support points


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



int main(int argc, char *argv[])
{
	constexpr int dim = 3;
	constexpr int fe_degree=1;

	FE_RaviartThomas<dim> fe_rt(fe_degree);
	const unsigned int n_dofs = fe_rt.dofs_per_cell;

	//Evaluate nodal values (i.e. node functional applied) for basis functions on generalized support points
    const std::vector<Point<dim> > &points = fe_rt.get_generalized_support_points();
    std::vector<Vector<double> > support_point_values (points.size(), Vector<double>(dim));
    std::vector<double> nodal_values(n_dofs);

    //NOTE: We can see that everytime only one nodal value will be 1, rest all are zero
    for (unsigned int i=1; i< 2/*n_dofs*/; ++i)
    {
    	for (unsigned int k=0; k<points.size(); ++k)
    		for (unsigned int d=0; d<dim; ++d)
    		{
    			support_point_values[k][d] = fe_rt.shape_value_component(i, points[k], d);
    		}
    }

	fe_rt.convert_generalized_support_point_values_to_dof_values(support_point_values,nodal_values);

    for (unsigned int j=0; j<n_dofs; ++j)
    {
    	//std::cout<<"(basis evaluation, nodal value) = ("<<support_point_values[j][0]<<" ,"<<nodal_values[j]<<")"<<std::endl;
    	std::cout<<"(nodal value) = ("<<nodal_values[j]<<")"<<std::endl;
    }


	return 0;
}
