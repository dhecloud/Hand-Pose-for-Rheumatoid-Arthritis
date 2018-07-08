#ifndef _CONSTANTS_H__
#define _CONSTANTS_H__

#if defined(_WIN32) || defined(_WIN64)
	#define WINDOWS
#endif

#define JOINT_NUM 21
#define PROJECTION_DIM (96*96)

#if defined(WINDOWS)
	#define SUBJECT_NUM 9
#else
	#define SUBJECT_NUM 2
#endif

#define GESTURE_NUM 17
#define MAX_IMAGE_NUM 500
#define MAX_DEPTH 2001

#define MAX_PCA_SZ (JOINT_NUM * 3) 
#define PROJ_SZ 96
#define HEAT_SZ 18
#define HEAT_SZ_SQ 324

#define HEAT_NUM 21

#define SRC_WIDTH 320
#define SRC_HEIGHT 240
#define SRC_DIM (SRC_WIDTH*SRC_HEIGHT)


enum ProjectionAxis {
	AXIS_XY = 0,
	AXIS_YZ = 1,
	AXIS_ZX = 2
};

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef UCHAR
typedef uchar UCHAR;
#endif

#ifndef UINT
typedef unsigned int UINT;
#endif

#endif
