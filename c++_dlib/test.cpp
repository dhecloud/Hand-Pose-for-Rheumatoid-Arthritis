#include <typeinfo>

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <data_loader.h>
#include <dlib/matrix.h>
#include <array>
using namespace std;
using namespace dlib;
 
class PrintNetVistor {
public:

template <typename any_net>
void operator()(size_t idx, any_net& net){}

template <typename T, typename U, typename E>
void operator()(size_t i, add_layer<T,U,E>&l)
{
	cout << "layer" ;
	cout << i ;
	cout << " : ";
	cout << l.layer_details() << endl;
}

// template <typename tensor>
// void operator()(size_t idx, tensor& t)
// {
	
	// if (idx == (size_t)0){
		// cout << t.num_samples() << endl;
		// cout << t.k() << endl;
		// cout <<  t.nc() << endl;
		// cout << t.nr() << endl;
		// cout << "" << endl;
		// for (int l = 0; l < t.num_samples(); l++){
			// for (int i = 0; i < t.k(); i++){
				// for (int j = 0; j < t.nr(); j++){
					// for (int k = 0; k < t.nc(); k++)
					// {
						// cout << t.host()[((l*t.k() + i)*t.nr() + j)*t.nc() + k];
						// cout << " " ;
					// }
					
				// }
			// }
		// }
	// }
// }

};
 
int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.  
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return 1;
    }

    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset("mnist", training_images, training_labels, testing_images, testing_labels);

    using net_type = loss_multiclass_log<
								fc<10,
								extract<0,16,2,2,
								extract<128,16,4,2,
                                max_pool<3,3,3,3,relu<con<16,5,5,1,1,
                                max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                input<matrix<unsigned char>> 
                                >>>>>>>>>>;


    net_type net;
	testing_images.resize(1);
	std::vector<unsigned long> predicted_labels = net(testing_images);
	cout <<net <<endl;
	// cout << ((layer<3>(net).get_output()).k()) <<endl;
	// cout << ((layer<3>(net).get_output()).nr()) <<endl;
	// cout << ((layer<3>(net).get_output()).nc()) <<endl;
	// for (int l = 0; l < ((layer<3>(net).get_output()).num_samples()); l++){
		// for (int i = 0; i < ((layer<3>(net).get_output()).k()); i++){
			// for (int j = 0; j < ((layer<3>(net).get_output()).nr()); j++){
				// for (int k = 0; k < ((layer<3>(net).get_output()).nc()); k++)
						// {
							// cout << (layer<3>(net).get_output()).host()[((l*((layer<3>(net).get_output()).k()) + i)*((layer<3>(net).get_output()).nr()) + j)*((layer<3>(net).get_output()).nc()) + k];
							// cout << " " ;
						// }
						// cout << "" <<endl ;
					// }
					// cout << "" <<endl ;
				// }
			// }
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" <<endl ;
	for (int l = 0; l < ((layer<2>(net).get_output()).num_samples()); l++){
		for (int i = 0; i < ((layer<2>(net).get_output()).k()); i++){
			for (int j = 0; j < ((layer<2>(net).get_output()).nr()); j++){
				for (int k = 0; k < ((layer<2>(net).get_output()).nc()); k++)
						{
							cout << (layer<2>(net).get_output()).host()[((l*((layer<2>(net).get_output()).k()) + i)*((layer<2>(net).get_output()).nr()) + j)*((layer<2>(net).get_output()).nc()) + k];
							cout << " " ;
						}
						cout << "" <<endl ;
					}
					cout << "" <<endl ;
				}
			}
    
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

