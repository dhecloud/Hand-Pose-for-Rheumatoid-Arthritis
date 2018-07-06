#include "stdafx.h"
#include <iostream>
#include "data_loader.h"

using namespace std;
using namespace dlib;

static const char* GESTURE_NAMES[] = {
	"1", "2", "3", "4", "5", "6", "7", "8", "9",
	"I", "IP", "L", "MP", "RP", "T", "TIP", "Y"
};
#define FOCAL_LENGTH 241.42f

MSRALoader::MSRALoader(
	const char* MSRA_folder,
	int size
)
{
	cout << "MSRALoader constructing" << endl;

	m_iCurrentGesture = -1,
	m_iCurrentSubject = -1,
	m_iMaxFrameNo= 0,
	m_iCurrentFrame= -1,
	m_bJointsLoadedForFrame=false,
	m_pointCloud =NULL,
	m_pointCloudSize=0,
	m_depthImageBBox = cv::Rect(0,0,0,0);
	centerBBx = 0;
	centerBBy = 0;
	sprintf(m_szPrefix, "%s", MSRA_folder);
	imgsize = ceil((size/2)*2);

}

MSRALoader::~MSRALoader()
{

	cout << "MSRALoader destructing" << endl;

}

void MSRALoader::loadMSRA(){
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 500 ; j++)
		{
			if ((i== 3) && (j == 499)){	//missing bin for frame 499, subject 3, pose 5
				continue;
		}
			loadDepth(i,4,j);		//void loadDepth(int subject, int gesture, int frame, bool loadJoints = true)
		}
	}
	cout << "MSRA dataset succesfully loaded" <<endl;

}
bool MSRALoader::loadJoints(
	int subject, 
	int gesture
)
{
	if (subject == m_iCurrentSubject && 
		gesture == m_iCurrentGesture)
		return true;

	char joint_path[255];
	sprintf(joint_path,
#ifdef WINDOWS
		"%s\\P%d\\%s\\joint.txt",
#else
		"%s/P%d/%s/joint.txt",
#endif
		m_szPrefix, subject, GESTURE_NAMES[gesture]
	);
	// cout <<  joint_path << endl;
	FILE *pJointFile = fopen(joint_path, "r");
	if (!pJointFile) {
		fprintf(stderr, "Could not open '%s'\n", joint_path);
		return false;
	}
	else{
		//cout << "Passed!" << endl;
	}

	int tmp = 0;
	fscanf(pJointFile, "%d\n", &m_iMaxFrameNo);
	for (int i_image = 0; i_image < m_iMaxFrameNo; ++i_image)
	{
		for (int i_joint = 0; i_joint < JOINT_NUM; ++i_joint)
		{
			tmp = i_joint * 3;
			fscanf(pJointFile, "%f %f %f", &data_joints[i_image][tmp],
				&data_joints[i_image][tmp + 1],
				&data_joints[i_image][tmp + 2]);
			data_joints[i_image][tmp + 2] *= (-1.0);

			if (data_joints[i_image][tmp + 2] < 0)
				printf("==> Negative value of ground truth depth!\n");
			if (i_joint < JOINT_NUM - 1)
				fscanf(pJointFile, " ");
			else
				fscanf(pJointFile, "\n");
		}
	}
	fclose(pJointFile);

	m_iCurrentGesture = gesture;
	m_iCurrentSubject = subject;
	return true;
}

void MSRALoader::loadDepth(
	int subject, 
	int gesture, 
	int frame,
	bool loadJoints
)
{
	m_bJointsLoadedForFrame = false;

	// load joints if needed
	if (loadJoints && 
		(subject != m_iCurrentSubject || gesture != m_iCurrentGesture)
		)
	{
		if (!this->loadJoints(subject, gesture))
			throw MSRALoader::ERR_CANNOT_LOAD_JOINTS;
		else
			m_bJointsLoadedForFrame = true;
	}


	char fname[255];
	sprintf(fname,
#ifdef WINDOWS
		"%s\\P%d\\%s\\%06d_depth.bin",
#else
		"%s/P%d/%s/%06d_depth.bin",
#endif		
		m_szPrefix, subject, gestureName(gesture), frame);

	
	// cout << fname << endl ;
	if (!loadDepthFromFile(fname)) 
		throw MSRALoader::ERR_CANNOT_LOAD_DEPTH;

	m_iCurrentFrame = frame;
}

float* MSRALoader::jointFrame(
	int frame_no
)
{
	// cout << "Returning frame ";
	// cout << frame_no;
	// cout << " of subject ";
	// cout << m_iCurrentSubject <<endl;
	return data_joints[frame_no];
}

int MSRALoader::frameCount()
const
{
	return m_iMaxFrameNo;
}


const char* MSRALoader::gestureName(
	int g
)
{
	return GESTURE_NAMES[g];
}

void MSRALoader::printProperties() { 
	cout << m_iCurrentSubject <<endl;
	cout << m_iCurrentGesture <<endl;
	cout << m_iCurrentFrame <<endl;
	cout << m_pointCloud <<endl;
	}

bool MSRALoader::loadDepthFromFile(const char* binFile) {
	// cv::Mat map = cv::imread(binFile, CV_LOAD_IMAGE_ANYDEPTH);
	// cv::imshow("window", map);
	FILE* f;
#ifdef WINDOWS
	fopen_s(&f, binFile, "rb");
#else
	f = fopen(binFile, "rb");
#endif

	if (f) {
		SAFE_DELETE_ARR(m_pointCloud);
		m_pointCloudSize = 0;
		m_centerPoint[0] = m_centerPoint[1] = m_centerPoint[2] = 0;

		uint imgwidth;
		uint imgheight;
		uint bbright;
		uint bbbtm;
		fread(&imgwidth, sizeof(uint), 1, f); // img width
		fread(&imgheight, sizeof(uint), 1, f); // img height
		fread(&m_depthImageBBox.x, sizeof(uint), 1, f); // bbox left
		fread(&m_depthImageBBox.y, sizeof(uint), 1, f); // bbox top
		fread(&bbright, sizeof(uint), 1, f); // bbox right
		m_depthImageBBox.width = bbright - m_depthImageBBox.x;
		fread(&bbbtm, sizeof(uint), 1, f); // bbox bottom
		m_depthImageBBox.height = bbbtm - m_depthImageBBox.y;
		

		matrix<float> depth_mat;
		depth_mat.set_size(m_depthImageBBox.height, m_depthImageBBox.width);
		//depth_mat.set_size(imgheight, imgwidth);
		for (int i = 0; i< m_depthImageBBox.height; i++){
			for (int j = 0; j < m_depthImageBBox.width; j++){
				depth_mat(i,j) = 0;
			}
		}
		
		// read the depth from file and populate point count
		for (int i = 0; i < m_depthImageBBox.height; i++) {
			for (int j = 0; j < m_depthImageBBox.width; j++){
				float z;

				// depth data
				fread(&z, sizeof(float), 1, f); 
				if (fabs(z) < FLT_EPSILON) continue;
				if (fabs(z) < 0.001) continue;
				if (fabs(z) > 1000.0) continue;
				else
					depth_mat(i,j) = z; 

			}
		}
		
		fclose(f);

		//TODO: can refactor
		int imgsizex;
		int imgsizey;
		centerBBx =  ceil(m_depthImageBBox.width/2);
		centerBBy =  ceil(m_depthImageBBox.height/2);
		imgsizex = centerBBx - imgsize/2;
		imgsizey = centerBBy - imgsize/2;
		if (imgsizex < 0){
			imgsizex =0;
		}
		if (imgsizey < 0){
			imgsizey =0;
		}
		matrix<float> depth_mat_imgsize;
		depth_mat_imgsize.set_size(imgsize,imgsize);
		int k = 0;
		int l = 0;

		for (int i = imgsizey; i < imgsizey+imgsize; i++) {
			for (int j = imgsizex; j < imgsizex+imgsize; j++){


				if ((j >= m_depthImageBBox.width) || (i >= m_depthImageBBox.height)){
					depth_mat_imgsize(k,l) = 0;
					continue;
				}
				depth_mat_imgsize(k,l) = depth_mat(i,j); 
				l++;

			}
			k++;
			l = 0;
		}
		return true;
	}
	else 
		return false;
}