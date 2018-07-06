// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will train the venerable LeNet convolutional
    neural network to recognize hand written digits.  The network will take as
    input a small image and classify it as one of the 10 numeric digits between
    0 and 9.

    The specific network we will run is from the paper
        LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE 86.11 (1998): 2278-2324.
    except that we replace the sigmoid non-linearities with rectified linear units. 

    These tools will use CUDA and cuDNN to drastically accelerate network
    training and testing.  CMake should automatically find them if they are
    installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/

#include <typeinfo>

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <data_loader.h>
#include <dlib/matrix.h>
#include <array>
using namespace std;
using namespace dlib;
 
typedef unsigned char UCHAR;

	template <
    typename SUBNET
    > 
	using start_net =
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	dlib::bn_con<
	dlib::con<16,3,3,1,1,
	dlib::relu<
	dlib::con<16,3,3,1,1, SUBNET
	>>>>>>;
	
	template <
	int N,
    typename SUBNET 
    > 
	using skip =
	dlib::relu<
	dlib::bn_con<
	dlib::con<N, 1, 1, 1, 1, SUBNET>>>;
	
	
	template <
	 int N, 
    typename SUBNET
    > 
	using res =
	dlib::relu<
	dlib::bn_con<
	dlib::con<N,3,3,1,1,
	dlib::relu<
	dlib::con<N,3,3,1,1,SUBNET>>>>>;
	
	// template <
	 // int offset, 
    // typename SUBNET
    // > 
	// using regress =
	// dlib::extract<offset,64,6,6,SUBNET>;

	
	template <typename SUBNET> using res1    =  add_prev2<skip<32,skip1<tag2<res<32,tag1<SUBNET>>>>>>;
	template <typename SUBNET> using res2    =  add_prev2<skip<64,skip1<tag2<res<64,tag1<SUBNET>>>>>>;
	// template <typename SUBNET> using regressfull    = regress<3,skip1<regress<2,skip1<regress<1,skip1<regress<0,tag1<SUBNET>;

class PrintNetVistor {
public:


template <typename T, typename U, typename E>
void operator()(size_t i, add_layer<T,U,E>&l)
{
	cout << "layer" ;
	cout << i ;
	cout << " : ";
	cout << l.layer_details() << endl;
}

template <typename tensor>
void operator()(size_t idx, tensor& t)
{
	// int tmp = 0;
	// if (tmp = 0){
		for (int l = 0; l < t.num_samples(); l++){
			for (int i = 0; i < t.k(); i++){
				for (int j = 0; j < t.nr(); j++){
					for (int k = 0; k < t.nc(); k++)
					{
						cout << t.host()[((l*t.k() + i)*t.nr() + j)*t.nc() + k];
						cout << " " ;
					}
					
				}
			}
		}
		// tmp++;
	// }
}

};

int main(int argc, char** argv) try
{

	using net_type = loss_mean_squared_multioutput<
	dlib::fc<63,
	dropout<
	dlib::fc<1024,
	dropout<
	dlib::fc<1024,
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	res2<
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	res1<
	start_net<
	dlib::input<
	dlib::matrix<float
	>>>>>>>>>>>>>>>;
	
    net_type net;
	deserialize("../../results/MSRA_chkpoint.dat") >> net ;
	PrintNetVistor printVisitor;
	//dlib::visit_layers(net.subnet(),printVisitor);
	dlib::visit_layer_parameters(layer<5>(net).subnet(), printVisitor);

}

catch(std::exception& e)
{
    cout << e.what() << endl;
}

