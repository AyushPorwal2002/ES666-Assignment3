import cv2
import numpy as np
import random
import os
import glob

class PanaromaStitcher:

    def make_panaroma_for_images_in(self, path):
        # Step 1: Load images from the path using glob to find all .jpg images
        image_paths = sorted(glob.glob(f'{path}{os.sep}*.jpg'))

        images = []
        for image_path in image_paths:
            img = cv2.imread(image_path)

            if img is None:
                print(f"Error reading image {image_path}, skipping...")
                continue

            # Resize image to half its original size to speed up processing
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            images.append(img)

        if len(images) < 2:
            raise ValueError("At least two images are required to stitch a panorama.")

        # Step 2: Initialize the center image as the reference
        center_index = len(images) // 2
        stitched_image = images[center_index]  # Start with the center image as the reference
        homography_matrices = []

        # Step 3: Stitch images to the right of the center image
        for i in range(center_index + 1, len(images)):
            left_img = stitched_image
            right_img = images[i]

            # Step 4: Detect keypoints and descriptors for both images
            key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)

            # Step 5: Match the keypoints between the two images
            good_matches = self.match_keypoint(key_points1, key_points2, descriptor1, descriptor2)

            # Step 6: Compute the homography matrix using RANSAC
            H = self.ransac(good_matches)
            if H is None:
                print("Homography could not be computed, skipping image pair.")
                continue
            homography_matrices.append(H)

            # Step 7: Stitch the images together using the computed homography
            stitched_image = self.stitch_images(left_img, right_img, H)

        # Step 8: Stitch images to the left of the center image
        for i in range(center_index - 1, -1, -1):
            left_img = images[i]
            right_img = stitched_image

            # Step 4: Detect keypoints and descriptors for both images
            key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)

            # Step 5: Match the keypoints between the two images
            good_matches = self.match_keypoint(key_points1, key_points2, descriptor1, descriptor2)

            # Step 6: Compute the homography matrix using RANSAC
            H = self.ransac(good_matches)
            if H is None:
                print("Homography could not be computed, skipping image pair.")
                continue
            homography_matrices.append(H)

            # Step 7: Stitch the images together using the computed homography
            stitched_image = self.stitch_images(left_img, right_img, H)

        return stitched_image, homography_matrices

    def get_keypoint(self, left_img, right_img):
        """Detect keypoints and compute descriptors using SIFT."""
        l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        key_points1 = sift.detect(l_img, None)
        key_points1, descriptor1 = sift.compute(l_img, key_points1)

        key_points2 = sift.detect(r_img, None)
        key_points2, descriptor2 = sift.compute(r_img, key_points2)

        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoint(self, key_points1, key_points2, descriptor1, descriptor2):
        """Match keypoints using the Brute-force matcher and apply a ratio test."""
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                left_pt = key_points1[m.queryIdx].pt
                right_pt = key_points2[m.trainIdx].pt
                good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

        return good_matches

    def homography(self, points):
        """Compute the homography matrix using the points."""
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        u, s, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H = H / H[2, 2]
        return H

    def ransac(self, good_pts):
        """Apply RANSAC to estimate the best homography."""
        best_inliers = []
        final_H = None
        threshold = 5
        iterations = 2000

        # Ensure there are enough points to compute homography
        if len(good_pts) < 4:
            print("Not enough matching points to compute homography.")
            return None

        for _ in range(iterations):
            if len(good_pts) < 4:
                continue

            random_pts = random.sample(good_pts, 4)
            H = self.homography(random_pts)
            inliers = []

            for pt in good_pts:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)

                if Hp[2] != 0:
                    Hp /= Hp[2]
                    dist = np.linalg.norm(p_1 - Hp)

                    if dist < threshold:
                        inliers.append(pt)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                final_H = H

        return final_H

    def stitch_images(self, left_img, right_img, H):
        """Stitch two images using the computed homography."""
        rows1, cols1 = right_img.shape[:2]
        rows2, cols2 = left_img.shape[:2]

        # Calculate the four corners of both images
        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 2)
        points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 2)

        points2_transformed = np.array([self.apply_homography(H, p) for p in points2])

        # Concatenate points to determine the bounding box
        points = np.vstack((points1, points2_transformed))
        [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        stitched_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=left_img.dtype)

        # Apply homography to the left image
        for y in range(rows2):
            for x in range(cols2):
                new_x, new_y = self.apply_homography(translation.dot(H), [x, y])
                new_x, new_y = int(new_x), int(new_y)

                if 0 <= new_x < stitched_img.shape[1] and 0 <= new_y < stitched_img.shape[0]:
                    stitched_img[new_y, new_x] = left_img[y, x]

        # Copy pixels from the right image
        for y in range(rows1):
            for x in range(cols1):
                new_x, new_y = x - x_min, y - y_min

                if 0 <= new_x < stitched_img.shape[1] and 0 <= new_y < stitched_img.shape[0]:
                    stitched_img[new_y, new_x] = right_img[y, x]

        return stitched_img

    def apply_homography(self, H, point):
        """Apply homography transformation to a single point."""
        x, y = point
        transformed_point = np.dot(H, [x, y, 1])
        transformed_point /= transformed_point[2]
        return transformed_point[:2]