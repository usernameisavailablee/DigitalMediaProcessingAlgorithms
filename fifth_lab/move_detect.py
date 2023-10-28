import sys
import os
import cv2
current_dir = os.path.dirname(__file__)

third_lab_path = os.path.join(current_dir, "..", "third_lab")
sys.path.append(third_lab_path)

from gauss_methods import *

def read_write_to_file():

    video = cv2.VideoCapture("input_video/main_video.mov")
    ok, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output_video/output.mov", fourcc, 25, (w, h))

    #И фрейм прочитали
    ok, start_frame = video.read()
    gray_start_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_start_frame = apply_gaussian_blur(gray_start_frame,11,5.0)
    start_frame = blurred_start_frame
    out = cv2.VideoWriter('out_moving.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))

    while True:
        copy_start_frame = start_frame.copy()

        ok, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = apply_gaussian_blur(gray_frame,11,5.0)

        frame_diff = cv2.absdiff(blurred_frame,copy_start_frame);


#        frame_diff = cv2.threshold(frame_diff);

        frame_diff = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)[1]


#        contours_for_frame_diff = cv2.findContours(frame_diff);

        contours_for_frame_diff, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_for_frame_diff, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 200
        for contour in contours_for_frame_diff:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Вы можете выполнить дополнительные действия с контуром,
                # например, нарисовать его или выполнить другую обработку.
                out.write(frame)
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
#        area = cv2.contourArea(contours_for_frame_diff)

#        cv2.imshow("frame_diff",frame_diff)
        cv2.imshow('blurred_frame',blurred_frame)
        cv2.imshow('frame1', frame)

        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


read_write_to_file()
