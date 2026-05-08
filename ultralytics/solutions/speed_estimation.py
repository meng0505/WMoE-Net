



from time import time



import numpy as np



from ultralytics.solutions.solutions import BaseSolution

from ultralytics.utils.plotting import Annotator, colors





class SpeedEstimator(BaseSolution):

    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trk_pt (Dict[int, float]): Dictionary storing previous timestamps for tracked objects.
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        annotator (Annotator): Annotator object for drawing on images.
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        track_line (List[Tuple[float, float]]): List of points representing the object's track.
        r_s (LineString): LineString object representing the speed estimation region.

    Methods:
        initialize_region: Initializes the speed estimation region.
        estimate_speed: Estimates the speed of objects based on tracking data.
        store_tracking_history: Stores the tracking history for an object.
        extract_tracks: Extracts tracks from the current frame.
        display_output: Displays the output with annotations.

    Examples:
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = estimator.estimate_speed(frame)
        >>> cv2.imshow("Speed Estimation", processed_frame)
    """



    def __init__(self, **kwargs):

        """Initializes the SpeedEstimator object with speed estimation parameters and data structures."""

        super().__init__(**kwargs)



        self.initialize_region()



        self.spd = {}

        self.trkd_ids = []

        self.trk_pt = {}

        self.trk_pp = {}



    def estimate_speed(self, im0):

        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing. Shape is typically (H, W, C) for RGB images.

        Returns:
            (np.ndarray): Processed image with speed estimations and annotations.

        Examples:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> processed_image = estimator.estimate_speed(image)
        """

        self.annotator = Annotator(im0, line_width=self.line_width)

        self.extract_tracks(im0)



        self.annotator.draw_region(

            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2

        )



        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):

            self.store_tracking_history(track_id, box)





            if track_id not in self.trk_pt:

                self.trk_pt[track_id] = 0

            if track_id not in self.trk_pp:

                self.trk_pp[track_id] = self.track_line[-1]



            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]

            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))





            self.annotator.draw_centroid_and_tracks(

                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width

            )





            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):

                direction = "known"

            else:

                direction = "unknown"





            if direction == "known" and track_id not in self.trkd_ids:

                self.trkd_ids.append(track_id)

                time_difference = time() - self.trk_pt[track_id]

                if time_difference > 0:

                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference



            self.trk_pt[track_id] = time()

            self.trk_pp[track_id] = self.track_line[-1]



        self.display_output(im0)



        return im0

