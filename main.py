import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")
# sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

# cv2.imshow("Sample", sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

filename = None
image = None
kp1 = None
kp2 = None
mp = None
best_score = 0

for file in [file for file in os.listdir("SOCOFing/Real")]:
    fingerprint_image = cv2.imread("SOCOFing/Real/" + file)

    # scale-invariant feature transform (SIFT)
    sift = cv2.SIFT_create()

    keypoint1, descriptor1 = sift.detectAndCompute(sample, None)
    keypoint2, descriptor2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                    {}).knnMatch(descriptor1, descriptor2, k=2)

    matchpoints = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            matchpoints.append(p)

    keypoints = 0

    if len(keypoint1) < len(keypoint2):
        keypoints = len(keypoint1)
    else:
        keypoints = len(keypoint2)

    if len(matchpoints) / keypoints * 100 > best_score:
        best_score = len(matchpoints) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1 = keypoint1
        kp2 = keypoint2
        mp = matchpoints

print("BEST MATCH: " + filename)
print("SCORE: " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
