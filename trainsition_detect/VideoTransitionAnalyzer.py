import os
import cv2
import numpy as np
from collections import defaultdict
from skimage.metrics import structural_similarity


class VideoTransitionAnalyzer:
    def __init__(self):
        pass

    def __call__(self, video_path, output_path):
        """
        Analyzes the transition frames in a video and saves the results to the specified directory.

        :param video_path: Path to the input video file
        :param output_path: Path to the output directory
        :return: A dictionary containing transition frame information and a list of cut points
        """
        transition_scores = defaultdict(list)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        previous_frame = None
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.preprocess_frame(frame)
            if previous_frame is not None:
                probabilities = self.calculate_transition_probabilities(previous_frame, processed_frame)
                for key in probabilities:
                    transition_scores[key].append(probabilities[key])
                transition_scores['frame_number'].append(frame_number)

            previous_frame = processed_frame
            frame_number += 1

        for key in transition_scores:
            transition_scores[key] = np.asarray(transition_scores[key])

        cuts = self.detect_cuts(transition_scores)
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        save_dir = os.path.join(output_path, video_name)
        self.extract_clips(cap, save_dir, fps, cuts)

        cap.release()

        return transition_scores, cuts

    def preprocess_frame(self, frame):
        """
        Preprocesses a video frame to extract features.

        :param frame: Input video frame
        :return: A dictionary containing the extracted features
        """
        resized_frame = cv2.resize(frame, (256, 256))
        channels = cv2.split(resized_frame)
        histograms = []
        for channel in channels:
            histograms.append(cv2.calcHist([channel], [0], None, [254], [1, 255]))

        gray_frame = cv2.resize(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY), (128, 128))
        edge_map = np.maximum(gray_frame, cv2.Canny(gray_frame, 100, 200))

        return {
            'bgr_hist': np.asarray(histograms),
            'edge_map': edge_map,
        }

    def calculate_transition_probabilities(self, frame0, frame1):
        """
        Calculates the transition probabilities between two frames.

        :param frame0: Features of the first frame
        :param frame1: Features of the second frame
        :return: A dictionary containing the transition probabilities
        """
        bgr_similarity = cv2.compareHist(frame0['bgr_hist'], frame1['bgr_hist'], cv2.HISTCMP_CORREL)
        canny_similarity = structural_similarity(frame0['edge_map'], frame1['edge_map'], data_range=255)
        svm_score = 4.61480465 * bgr_similarity + 3.75211168 * canny_similarity - 5.485968377115124

        return {
            'bgr': bgr_similarity,
            'canny': canny_similarity,
            'svm': svm_score,
        }

    def detect_cuts(self, transition_scores):
        """
        Detects cut points in the video based on transition scores.

        :param transition_scores: A dictionary containing transition scores
        :return: A list of cut points
        """
        num_frames = len(transition_scores['bgr'])
        transition_scores['conv_svm'] = transition_scores['svm'].copy()
        transition_scores['conv_svm'][1:-1] = np.convolve(transition_scores['conv_svm'], np.array([1, 1, 1]) / 3.0, mode='valid')

        svm_scores = transition_scores['svm']
        conv_svm_scores = transition_scores['conv_svm']
        bgr_scores = transition_scores['bgr']
        visibility = np.ones(num_frames + 1, bool)
        visibility[-1] = False

        for i in range(num_frames):
            if svm_scores[i] < 0:
                visibility[i] = False
            elif i >= 8:
                start = max(i - 8, 0)
                if conv_svm_scores[i] < 0.75:
                    mu, std = self.get_mu_std(conv_svm_scores[start:i])
                    if conv_svm_scores[i] < mu - 3 * max(0.2, std):
                        visibility[i] = False

        cuts = []
        start = 0
        for i in range(len(visibility)):
            if not visibility[i]:
                if (i - start) > 8:
                    cuts.append((start + 4, i - 4))
                start = i + 1

        transition_scores['visibility'] = visibility[:-1]

        return cuts

    def extract_clips(self, cap, save_dir, fps, cuts):
        """
        Extracts clips from the video and saves them to the specified directory.

        :param cap: Video capture object
        :param save_dir: Path to the save directory
        :param fps: Video frame rate
        :param cuts: List of cut points
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(save_dir, exist_ok=True)

        for i, (start_frame, end_frame) in enumerate(cuts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            save_file_path = os.path.join(save_dir, f"clip_{i}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_file_path, fourcc, fps, (width, height))

            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

    def get_mu_std(self, arr):
        """
        Calculates the mean and standard deviation of an array, excluding the top and bottom 20% of values.

        :param arr: Input array
        :return: Mean and standard deviation of the array
        """
        M = len(arr)
        arr = np.sort(arr)
        arr = arr[int(M * 0.2): int(M * 0.8)]
        mu = arr.mean()
        std = arr.std()
        return mu, std


if __name__ == "__main__":
    video_transition = VideoTransitionAnalyzer()
    transition_scores, cuts = video_transition(
        video_path="videos/test.mp4",
        output_path="videos/outputs"
    )
