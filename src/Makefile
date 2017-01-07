CC = g++
CFLAGS =
#LIBS = /usr/local/Cellar/opencv/2.4.13/lib
LIBS1 = ../cv_static/opencv-2.4.13/lib/ 
LIBS2 = ../cv_static/opencv-2.4.13/3rdparty/lib
INC1 = ../cv_static/opencv-2.4.13/
INC2 = ../cv_static/opencv-2.4.13/include/
INC3 = ../cv_static/opencv-2.4.13/modules/core/include/
INC4 = ../cv_static/opencv-2.4.13/modules/legacy/include/
INC5 = ../cv_static/opencv-2.4.13/modules/ocl/include/
INC6 = ../cv_static/opencv-2.4.13/modules/videostab/include/
INC7 = ../cv_static/opencv-2.4.13/modules/stitching/include/
INC8 = ../cv_static/opencv-2.4.13/modules/calib3d/include/
INC9 = ../cv_static/opencv-2.4.13/modules/highgui/include/
INC10 = ../cv_static/opencv-2.4.13/modules/ml/include/
INC11 = ../cv_static/opencv-2.4.13/modules/photo/include/
INC12 = ../cv_static/opencv-2.4.13/modules/features2d/include/
INC13 = ../cv_static/opencv-2.4.13/modules/imgproc/include/
INC14 = ../cv_static/opencv-2.4.13/modules/nonfree/include/
INC15 = ../cv_static/opencv-2.4.13/modules/ts/include/
INC16  = ../cv_static/opencv-2.4.13/modules/contrib/include/
INC17 = ../cv_static/opencv-2.4.13/modules/flann/include/
INC18 = ../cv_static/opencv-2.4.13/modules/objdetect/include/
INC19 = ../cv_static/opencv-2.4.13/modules/video/include/

LDFLAGS = -I $(INC1) -I $(INC2) -I $(INC3) -I $(INC4) -I $(INC5) -I $(INC6) -I $(INC7) -I $(INC8) -I $(INC9) -I $(INC10) -I $(INC11) -I $(INC12) -I $(INC13) -I $(INC14) -I $(INC15) -I $(INC16) -I $(INC17) -I $(INC18) -I$(INC19)
LDFLAGS += -L$(LIBS1) -L$(LIBS2) -lm -llibjpeg -llibpng -llibtiff -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lopencv_objdetect -lopencv_ml -lopencv_features2d -lopencv_calib3d -lopencv_flann -lopencv_legacy -lopencv_ocl -lopencv_contrib -lopencv_photo -lopencv_videostab -lzlib -lIlmImf -llibjasper

ALL = face_rec_main

all: $(ALL)

face_rec_main: face_rec_main.o
	$(CC) $(LDFLAGS) -o $@ $^

face_rec_main.o: face_rec_main.cpp
	$(CC) $(LDFLAGS) -c $<

.PHONY: clean

clean:
	rm -f face_rec_main *.o	
