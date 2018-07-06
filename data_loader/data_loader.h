#pragma once
#include "constants.h"
#include <iostream>
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>
#define SAFE_DELETE_ARR(x) { if(x) { delete[] x; x=NULL; }}

/*
	provides utility functions to load from
	MSRADataset
*/

class MSRALoader
{
	
	char m_szPrefix[255];
	int m_iCurrentSubject;
	int m_iCurrentGesture;
	int m_iCurrentFrame;
	bool m_bJointsLoadedForFrame;
	int m_iMaxFrameNo;
	int centerBBx;
	int centerBBy;
	float* m_pointCloud;
	uint m_pointCloudSize;
	float data_joints[MAX_IMAGE_NUM][JOINT_NUM * 3];
	cv::Rect m_depthImageBBox;
	cv::Vec3f m_centerPoint;
	dlib::matrix<float> depth_mat;
	dlib::matrix<float> depth_mat_imgsize;
	int imgsize;

public:

	enum ErrorNumbers {
		ERR_CANNOT_LOAD_JOINTS = 0x1000,
		ERR_CANNOT_LOAD_DEPTH = 0x1001,
	};

	static const char* gestureName(int g);

	MSRALoader(const char* MSRA_root, int size);
	~MSRALoader();

	void loadDepth(int subject, int gesture, int frame,
		bool loadJoints = true);

	float* jointFrame(int frame_no);
	int frameCount() const;
	void loadMSRA();
	float* getPointCloud() const { return m_pointCloud; }
	dlib::matrix<float> getDepth() { return depth_mat; }
	dlib::matrix<float> getDepthImgSize() { return depth_mat_imgsize; }
	void printProperties();
	uint getPointCloudSize() const { return m_pointCloudSize; }
	int getBBoxLeft() {return m_depthImageBBox.x;};
	int getBBoxWidth() {return m_depthImageBBox.width;};
	int getBBoxTop() {return m_depthImageBBox.y;};
	int getBBoxHeight() {return m_depthImageBBox.height;};
	int getcenterBBx() {return centerBBx;};
	int getcenterBBy() {return centerBBy;};

private:
	bool loadJoints(int subject, int gesture);
	bool loadCloudFromFile(const char* binFile);
	bool loadDepthFromFile(const char* binFile);
};
