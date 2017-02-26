import cv2

matches_matrix = []
pts1 = pts2 = []


def get_aligned_points_from_match(pts1, pts2, matches):
    for match in matches:
        set1 = []
        set2 = []
        assert (match.queryIdx < len(pts1))
        set1.append(pts1[match].queryIdx)
        assert (match.trainIdx < len(pts2))
        set2.append(pts2[match].trainIdx)
    return set1, set2


def key_points_to_points(kps):
    ps = []
    for kp in kps:
        ps.append(kp.pt)
    return ps


def find_homography_inliers_2_views(vi, vj):
    xy = get_aligned_points_from_match(pts1, pts2, matches_matrix[(vi, vj)])
    k = key_points_to_points(xy)
    k = key_points_to_points(k)
    min_v, max_v = cv2.minMaxLoc(k)
    h = cv2.findHomography(*k, ransacReprojThreshold=max_v * .004)
    return cv2.countNonZero(h)
