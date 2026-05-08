



from ultralytics.solutions.solutions import BaseSolution

from ultralytics.utils.plotting import Annotator





class AIGym(BaseSolution):

    """
    A class to manage gym steps of people in a real-time video stream based on their poses.

    This class extends BaseSolution to monitor workouts using YOLO pose estimation models. It tracks and counts
    repetitions of exercises based on predefined angle thresholds for up and down positions.

    Attributes:
        count (List[int]): Repetition counts for each detected person.
        angle (List[float]): Current angle of the tracked body part for each person.
        stage (List[str]): Current exercise stage ('up', 'down', or '-') for each person.
        initial_stage (str | None): Initial stage of the exercise.
        up_angle (float): Angle threshold for considering the 'up' position of an exercise.
        down_angle (float): Angle threshold for considering the 'down' position of an exercise.
        kpts (List[int]): Indices of keypoints used for angle calculation.
        annotator (Annotator): Object for drawing annotations on the image.

    Methods:
        monitor: Processes a frame to detect poses, calculate angles, and count repetitions.

    Examples:
        >>> gym = AIGym(model="yolo11n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> processed_image = gym.monitor(image)
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """



    def __init__(self, **kwargs):

        """Initializes AIGym for workout monitoring using pose estimation and predefined angles."""



        if "model" in kwargs and "-pose" not in kwargs["model"]:

            kwargs["model"] = "yolo11n-pose.pt"

        elif "model" not in kwargs:

            kwargs["model"] = "yolo11n-pose.pt"



        super().__init__(**kwargs)

        self.count = []

        self.angle = []

        self.stage = []





        self.initial_stage = None

        self.up_angle = float(self.CFG["up_angle"])

        self.down_angle = float(self.CFG["down_angle"])

        self.kpts = self.CFG["kpts"]



    def monitor(self, im0):

        """
        Monitors workouts using Ultralytics YOLO Pose Model.

        This function processes an input image to track and analyze human poses for workout monitoring. It uses
        the YOLO Pose model to detect keypoints, estimate angles, and count repetitions based on predefined
        angle thresholds.

        Args:
            im0 (ndarray): Input image for processing.

        Returns:
            (ndarray): Processed image with annotations for workout monitoring.

        Examples:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> processed_image = gym.monitor(image)
        """



        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)[0]



        if tracks.boxes.id is not None:



            if len(tracks) > len(self.count):

                new_human = len(tracks) - len(self.count)

                self.angle += [0] * new_human

                self.count += [0] * new_human

                self.stage += ["-"] * new_human





            self.annotator = Annotator(im0, line_width=self.line_width)





            for ind, k in enumerate(reversed(tracks.keypoints.data)):



                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]

                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)

                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)





                if self.angle[ind] < self.down_angle:

                    if self.stage[ind] == "up":

                        self.count[ind] += 1

                    self.stage[ind] = "down"

                elif self.angle[ind] > self.up_angle:

                    self.stage[ind] = "up"





                self.annotator.plot_angle_and_count_and_stage(

                    angle_text=self.angle[ind],

                    count_text=self.count[ind],

                    stage_text=self.stage[ind],

                    center_kpt=k[int(self.kpts[1])],

                )



        self.display_output(im0)

        return im0

