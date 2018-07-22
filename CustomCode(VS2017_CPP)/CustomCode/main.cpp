#include "CustomCodeRecognition.h"

void main()
{
	string result = "";
	CustomCode customcode;
	vector<Point2f> markers;

	Mat img = imread("./img/input.png");
	customcode.recognition(&img, &markers, &result);
	waitKey(0);
	return;

	VideoCapture camera(CV_CAP_ANY);
	if (!camera.isOpened())
	{
		printf("Camera is not opened\n");
		return;
	}

	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 1080.0);
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 1920.0);
	//namedWindow("frame", CV_WINDOW_AUTOSIZE);

	Mat frame;
	while (1)
	{
		camera >> frame;
		Mat intput = frame(Rect(frame.cols * 3 / 8, frame.rows * 3 / 8, frame.cols / 4, frame.rows / 4));

		customcode.recognition(&intput, &markers, &result);
		cout << result << endl;
		if (waitKey(30) >= 0)
		{
			int num = cv::waitKey(0);
			if (num == 32) continue;
			else
				break;
		}
	}
}