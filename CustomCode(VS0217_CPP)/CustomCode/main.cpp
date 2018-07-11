#include "CustomCodeRecognition.h"

void main()
{
	string result = "";
	CustomCode customcode;
	vector<Point2f> markers;

	VideoCapture camera(CV_CAP_ANY);
	if (!camera.isOpened())
	{
		printf("Camera is not opened\n");
		return;
	}

	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1080.0);
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 1920.0);
	namedWindow("frame", CV_WINDOW_AUTOSIZE);

	Mat frame;
	while (1)
	{
		camera >> frame;

		customcode.recognition(&frame, &markers, &result);

		if (waitKey(30) >= 0)
		{
			int num = cv::waitKey(0);
			if (num == 32) continue;
			else
				break;
		}
	}
}