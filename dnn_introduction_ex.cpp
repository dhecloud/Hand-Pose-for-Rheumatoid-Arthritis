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
class PrintNetVistor {
public:

// template <typename any_net>
// void operator()(size_t idx, any_net& net){}

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
	
	if (idx == (size_t)0){
		cout << t.num_samples() << endl;
		cout << t.k() << endl;
		cout <<  t.nc() << endl;
		cout << t.nr() << endl;
		cout << "" << endl;
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
	}
}

};

typedef unsigned char UCHAR;

	template <
    typename SUBNET
    > 
	using start_net =
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	dlib::con<16,3,3,1,1,
	dlib::relu<
	dlib::con<16,3,3,1,1, SUBNET
	>>>>>;
	
	template <
	int N,
    typename SUBNET 
    > 
	using diminc =
	dlib::relu<
	dlib::con<N, 1, 1, 1, 1, SUBNET>>;
	
	
	template <
	 int N, 
    typename SUBNET
    > 
	using res =
	dlib::con<N,3,3,1,1,
	dlib::relu<
	dlib::con<N,3,3,1,1,SUBNET>>>;
	
	// template <
	 // int offset, 
    // typename SUBNET
    // > 
	// using regress =
	// dlib::extract<offset,64,6,6,SUBNET>;

	
	template <typename SUBNET> using res1    =  add_prev1<res<32,tag1<diminc<32,SUBNET>>>>;
	template <typename SUBNET> using res2    =  add_prev1<res<64,tag1<diminc<64,SUBNET>>>>;
	// template <typename SUBNET> using regressfull    = regress<3,skip1<regress<2,skip1<regress<1,skip1<regress<0,tag1<SUBNET>;

	
float maxpoint(float* array, int size=NULL){
	float tmpmax = 0.0;
	if (size == NULL){
		int tmp;
		tmp = (sizeof(array)/sizeof(*array));
		size = tmp;
	}
	for (int m = 0 ; m < size ; m++)
	{
    if (array[m] > tmpmax)
        tmpmax = array[m];
	}
	return tmpmax;
}

float minpoint(float* array, int size=NULL){
	float tmpmin = 999999.0;
	if (size == NULL){
		int tmp;
		tmp = (sizeof(array)/sizeof(*array));
		size = tmp;
	}
	for (int m = 0 ; m < size ; m++)
	{
    if (array[m] < tmpmin)
        tmpmin = array[m];
	}
	return tmpmin;
}

float* normalize(float* array, int size=NULL){
	if (size == NULL){
		int tmp;
		tmp = (sizeof(array)/sizeof(*array));
		size = tmp;
	}
	float* norm = new float[size];
	
	float max = maxpoint(array);
	float min = minpoint(array);
	for (int i=0 ; i< size; i++){
		norm[i] = (array[i] - min) / (max - min);
	}
	return norm;
}

std::vector<matrix<UCHAR>> convertfloat2UCHARM(float* array, int size=NULL){
	if (size == NULL){
		int tmp;
		tmp = (sizeof(array)/sizeof(*array));
		size = tmp;
	}
	std::vector<matrix<UCHAR>> uchar_buff;
	//number of samples
	 uchar_buff.resize(1);
	 cout << "passed!" << endl;
	for (int i=0 ;i< size ; i++){

		uchar_buff[0](i)= static_cast<unsigned char>(array[i]*255);
	}
	return uchar_buff;
}

std::vector<matrix<float>> convertFloat2FloatM(float* array, int size=NULL){
	std::vector<matrix<float>> float_buff;
	if (size == NULL){
		int tmp;
		tmp = (sizeof(array)/sizeof(*array));
		size = tmp;
	}
	float_buff.resize(1);
	float_buff[0].set_size(1,size);
	for (int i=0 ;i< size ; i++){

		float tmpfloat = static_cast<float>(array[i]);
		float_buff[0](i)= tmpfloat;
	}
	return float_buff;
}

matrix<float> copyFloat2ColMat(float* array){
	matrix<float> tmp;
	tmp.set_size(63,1);
	for (int i = 0; i <63; i++){
		tmp(i,0) = array[i];
	}
	return tmp;
	
}

std::vector<matrix<float>> cleanMat(std::vector<matrix<float>> mat, int size){
	for (int i = 0; i < mat.size(); i++){
		for (int j = 0; j <size; j++){
			for (int k=0; k<size; k++){
				if ((fabs(mat[i](j,k)) > 1000.0) || (fabs(mat[i](j,k)) < 0.001)){
					mat[i](j,k) = 0;
				}
			}
		}
	}
	return mat;
}


int main(int argc, char** argv) try
{
	    if (argc != 2)
    {
        cout << "1 - train " << endl;
		cout << "2 - test " << endl;
        return 1;
    }
	int mode = argv[1][0] - '0';
	int imgsize = 96;
	MSRALoader msra ("../../data",imgsize); //try to make imgsize even number
	//msra.loadMSRA();
	std::vector<matrix<float>> training_depth;
	std::vector<matrix<float>> training_joints;
	std::vector<matrix<float>> testing_depth;
	std::vector<matrix<float>> testing_joints;
	// cout<< training_depth.size() <<endl;
	// cout<< training_joints.size() <<endl;

	
    // using net_type = loss_mean_squared_multioutput<
                                // fc<63,        
                                // relu<fc<84,   
                                // relu<fc<120,  
                                // max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                // max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                // input<matrix<float>> 
                                // >>>>>>>>>>>>;
								
	
	using net_type = loss_mean_squared_multioutput<
	dlib::fc<63,
	dropout<
	dlib::fc<2048,
	dropout<
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	res2<
	dlib::relu<
	dlib::max_pool<2, 2, 2, 2,
	res1<
	start_net<
	dlib::bn_con<
	dlib::input<
	dlib::matrix<float
	>>>>>>>>>>>>>>>;
	
	;
	// using net_type = loss_mean_squared_multioutput<
	// dlib::fc<63,
	// dlib::prelu<
	// dlib::con<16,3,3,1,1,
	// dlib::input<ewhy
	// dlib::matrix<float
	// >>>>>>;
	

    // This net_type defines the entire network architecture.  For example, the block
    // relu<fc<84,SUBNET>> means we take the output from the subnetwork, pass it through a
    // fully connected layer with 84 outputs, then apply ReLU.  Similarly, a block of
    // max_pool<2,2,2,2,relu<con<16,5,5,1,1,SUBNET>>> means we apply 16 convolutions with a
    // 5x5 filter size and 1x1 stride to the output of a subnetwork, then apply ReLU, then
    // perform max pooling with a 2x2 window and 2x2 stride.  



    // So with that out of the way, we can make a network instance.
    net_type net;
	if (mode == 1){
		for (int subject = 0; subject < 8; subject ++){
		for (int frame = 0; frame < 499; frame++){
			msra.loadDepth(subject,4,frame);
			matrix<float> depth_mat= msra.getDepthImgSize();
			depth_mat.set_size(imgsize, imgsize);
			// depth_mat.set_size(msra.getBBoxWidth(), msra.getBBoxHeight());
			training_depth.push_back(depth_mat);
			matrix<float> NormCurjointFrame = copyFloat2ColMat(msra.jointFrame(frame));
			NormCurjointFrame.set_size(63,1);
			training_joints.push_back(NormCurjointFrame);
		}
		cout<< "subject ";
		cout<< subject <<endl;
	}

	std::vector<matrix<float>> training_depth_1;
	training_depth_1 = cleanMat(training_depth, imgsize);
	training_depth = training_depth_1;
		sgd defsolver(0.1,0.9);
		dnn_trainer<net_type> trainer(net, defsolver);
		trainer.set_learning_rate(0.00000001);
		trainer.set_min_learning_rate(0.00000000000001);
		trainer.set_mini_batch_size(128);
		trainer.be_verbose();
		trainer.set_max_num_epochs(100000);
		trainer.set_synchronization_file("../../results/MSRA_chkpoint", std::chrono::seconds(120));
		trainer.train(training_depth,  training_joints);
		net.clean();
		serialize("../../results/MSRA_chkpoint.dat") << net;
	}
    // Now if we later wanted to recall the network from disk we can simply say:
   
	else if (mode == 2){
		deserialize("../../results/MSRA_chkpoint.dat") >> net;
		for (int frame = 0; frame < 499; frame++){
			msra.loadDepth(8,4,frame);
			matrix<float> depth_mat1= msra.getDepthImgSize();
			depth_mat1.set_size(imgsize, imgsize);
			testing_depth.push_back(depth_mat1);
			matrix<float> NormCurjointFrame1 = copyFloat2ColMat(msra.jointFrame(frame));
			NormCurjointFrame1.set_size(63,1);
			testing_joints.push_back(NormCurjointFrame1);
		}
	std::vector<matrix<float>> testing_depth_1;
	testing_depth_1 = cleanMat(testing_depth, imgsize);
	testing_depth = testing_depth_1;
	cout<< testing_depth.size() <<endl;
	cout<< testing_joints.size() <<endl;
		std::vector<matrix<float>> predicted_labels = net(testing_depth);
		
		cout << predicted_labels.size() <<endl;
		for (int i = 0; i <63; i++){
			cout << predicted_labels[0](i,0);
			cout <<  " ";
		}
		cout <<  " " <<endl;;
		for (int i = 0; i <63; i++){
			cout << testing_joints[0](i,0);
			cout <<  " ";
		}
	

	}
	
	else if (mode == 3){
		
			using test_net_type = loss_mean_squared_multioutput<
			dlib::fc<63,
			dropout<
			dlib::fc<2048,
			dlib::extract<9216,64,6,6,
			dropout<
			dlib::relu<
			dlib::max_pool<2, 2, 2, 2,
			res2<
			dlib::relu<
			dlib::max_pool<2, 2, 2, 2,
			res1<
			start_net<
			dlib::bn_con<
			dlib::input<
			dlib::matrix<float
			>>>>>>>>>>>>>>>>;
			test_net_type test_net;
		// deserialize("../../results/MSRA_chkpoint.dat") >> net;
		msra.loadDepth(8,4,499);
		matrix<float> depth_mat1= msra.getDepthImgSize();
		depth_mat1.set_size(imgsize, imgsize);
		testing_depth.push_back(depth_mat1);
		std::vector<matrix<float>> testing_depth_1;
		testing_depth_1 = cleanMat(testing_depth, imgsize);
		testing_depth = testing_depth_1;
		std::vector<matrix<float>> predicted_labels = test_net(testing_depth);
		
		PrintNetVistor printVisitor;
		cout << " " <<endl;
		cout << " " <<endl;
		cout << " " <<endl;
		cout << " " <<endl;
		// dlib::visit_layers(test_net,printVisitor);
			for (int l = 0; l < ((layer<7>(test_net).get_output()).num_samples()); l++){
				for (int i = 0; i < ((layer<7>(test_net).get_output()).k()); i++){
					for (int j = 0; j < ((layer<7>(test_net).get_output()).nr()); j++){
						for (int k = 0; k < ((layer<7>(test_net).get_output()).nc()); k++)
						{
							cout << (layer<7>(test_net).get_output()).host()[((l*((layer<7>(test_net).get_output()).k()) + i)*((layer<7>(test_net).get_output()).nr()) + j)*((layer<7>(test_net).get_output()).nc()) + k];
							cout << " " ;
						}
						
					}
				}
			}
		cout << " " <<endl;
		cout << " " <<endl;
		cout << " " <<endl;
		cout << " " <<endl;
			for (int l = 0; l < ((layer<4>(test_net).get_output()).num_samples()); l++){
				for (int i = 0; i < ((layer<4>(test_net).get_output()).k()); i++){
					for (int j = 0; j < ((layer<4>(test_net).get_output()).nr()); j++){
						for (int k = 0; k < ((layer<4>(test_net).get_output()).nc()); k++)
						{
							cout << (layer<4>(test_net).get_output()).host()[((l*((layer<4>(test_net).get_output()).k()) + i)*((layer<4>(test_net).get_output()).nr()) + j)*((layer<4>(test_net).get_output()).nc()) + k];
							cout << " " ;
						}
						
					}
				}
			}
		// cout << ((layer<7>(test_net).get_output()).k()) <<endl;
		// cout << ((layer<7>(test_net).get_output()).nr()) <<endl;
		// cout << ((layer<7>(test_net).get_output()).nc()) <<endl;
		// cout << " " <<endl;
		// cout << ((layer<4>(test_net).get_output()).k()) <<endl;
		// cout << ((layer<4>(test_net).get_output()).nr()) <<endl;
		// cout << ((layer<4>(test_net).get_output()).nc()) <<endl;
		
		
		
	}
    // Now let's run the training images through the network.  This statement runs all the
    // images through it and asks the loss layer to convert the network's raw output into
    // labels.  In our case, these labels are the numbers between 0 and 9.
	// std::vector<matrix<float>> predicted_labels = net(training_images);

    // int num_right = 0;
    // int num_wrong = 0;
    // And then let's see if it classified them correctly.
    // for (size_t i = 0; i < training_images.size(); ++i)
    // {
        // if (predicted_labels[i] == training_labels[i])
            // ++num_right;
        // else
            // ++num_wrong;
        
    // }
    // cout << "training num_right: " << num_right << endl;
    // cout << "training num_wrong: " << num_wrong << endl;
    // cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

    // Let's also see if the network can correctly classify the testing images.  Since
    // MNIST is an easy dataset, we should see at least 99% accuracy.
    // predicted_labels = net(testing_images);
    // num_right = 0;
    // num_wrong = 0;
    // for (size_t i = 0; i < testing_images.size(); ++i)
    // {
        // if (predicted_labels[i] == testing_labels[i])
            // ++num_right;
        // else
            // ++num_wrong;
        
    // }
    // cout << "testing num_right: " << num_right << endl;
    // cout << "testing num_wrong: " << num_wrong << endl;
    // cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;


    // Finally, you can also save network parameters to XML files if you want to do
    // something with the network in another tool.  For example, you could use dlib's
    // tools/convert_dlib_nets_to_caffe to convert the network to a caffe model.
    // net_to_xml(net, "lenet.xml");
}

catch(std::exception& e)
{
    cout << e.what() << endl;
}

