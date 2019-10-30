import cv2 as cv
import numpy as np
from numpy.linalg import norm

svm_params = dict(kernel_type=cv.ml.SVM_RBF, svm_type=cv.ml.SVM_C_SVC, C=2.67, gamma=5.383)


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):

    def __init__(self, C=1, gamma=0.5):
        self.model = cv.ml.SVM_create()

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:100, :100], bin[100:, :100], bin[:100, 100:], bin[100:, 100:]
        mag_cells = mag[:100, :100], mag[100:, :100], mag[:100, 100:], mag[100:, 100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    samples = np.float32(samples)
    # print(samples)
    return samples


# Here goes my wrappers:
def hog_single(img):
    samples = []
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bin[:100, :100], bin[100:, :100], bin[:100, 100:], bin[100:, 100:]
    mag_cells = mag[:100, :100], mag[100:, :100], mag[:100, 100:], mag[100:, 100:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    samples.append(hist)
    return np.float32(samples)


def trainSVM(num):
    imgs = []
    for i in range(65, num + 65):
        for j in range(1, 401):
            # print ('Class ' + chr(i) + ' is being loaded ')
            imgs.append(cv.imread('TrainData/' + chr(i) + '_' + str(j) + '.jpg', 0))  # all images saved in a list
            #print('TrainData/' + chr(i) + '_' + str(j) + '.jpg')
    labels = np.repeat(np.arange(1, num + 1), 400)  # label for each corresponding image saved above
    # for i in labels:
    #     print(i)
    samples = preprocess_hog(imgs)  # images sent for pre processeing using hog which returns features for the images
    print('SVM is building wait some time ...')
    print(len(labels))
    print(len(samples))
    model = SVM(C=2.67, gamma=5.383)
    # model.train(samples, labels)  # features trained against the labels using svm

    svm = cv.ml.SVM_create();
    svm.setKernel(cv.ml.SVM_RBF);
    svm.setType(cv.ml.SVM_C_SVC);

    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(samples, cv.ml.ROW_SAMPLE, labels)
    return svm
    # return


def predict(model, img):
    samples = hog_single(img)
    # resp=cv.UMat()
    _, resp = model.predict(samples)
    # print(resp)
    return resp.ravel()
