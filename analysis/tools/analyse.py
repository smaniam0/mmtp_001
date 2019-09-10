import numpy as np
import cv2 as cv
from trainutils import unpickleData

class Analyzer:
    def __init__(self, fname, clfname='../../data/class_img.pkl.gz', tstimg='../../data/kf16_tst.pkl.gz'):
        self.missed = unpickleData(fname)[0]
        # Convert Instance index to numpy array (convenience)
        self.missed[0] = np.array(self.missed[0]).flatten()
        ## Prediction is the index of the item with the largest value in array
        self.predidx = np.argmax(self.missed[2], 1) 
        ## Bin to find out the worst performers
        self.freqbin = np.unique(self.missed[1], return_counts=True)
        self.worst_performers = np.flip(np.argsort(self.freqbin[1]))

        self.clsimg = unpickleData(clfname)
        self.tstset = unpickleData(tstimg)[0]
        self.classfreq = np.unique(np.argmax(self.tstset.labels, axis=1), return_counts=True)[1]
        self.tstset = self.tstset.data


    def getTopWrongPredictionForClass(self, classid, N=5):
        idxlist = np.where(self.missed[1] == classid)
        ## Get all the wrong predictioos made for the class
        wrong_prediction = self.predidx[idxlist]
        ## Sort and Frequency Bin the wrong predictions
        wrong_freq_bin = np.unique(wrong_prediction, return_counts=True)
        sorted_bin = np.argsort(wrong_freq_bin[1])

        ## Just in case we have fewer types of wrong classifications
        if len(sorted_bin) < N:
            N = len(sorted_bin)

        ## Slice and Reverse the Order
        top_wrong_idx = np.flip(sorted_bin[-N:])
        top_wrong_prediction = []

        ## Gather the wrong predictions and it's frequency of occurance
        for i in range(N):
            top_wrong_prediction.append(
                (wrong_freq_bin[0][top_wrong_idx[i]],
                 wrong_freq_bin[1][top_wrong_idx[i]]))

        return top_wrong_prediction

    def getTopMisclassifications(self, N=5):
        if len(self.worst_performers) < N:
            N = len(self.worst_performers)
        bad_perf_idx = self.worst_performers[:N]
        bad_stats = []
        for i in range(N):
            idx = bad_perf_idx[i]
            classid = self.freqbin[0][idx]
            bad_stats.append( (classid, self.freqbin[1][idx], 
                self.classfreq[classid],
                self.getTopWrongPredictionForClass(classid)))

        return bad_stats

    def getMisclassImage(self, N=10):
        topmissed = self.getTopMisclassifications(N)
        top = 1
        for d in topmissed:
            miss_idx  = np.where(self.missed[1] == d[0])
            miss_inst = self.missed[0][miss_idx]
            wrong_class_inst = self.predidx[miss_idx]
            misclassified = []
            for wrong_pred in d[3]:
                wrong_pred_inst = miss_inst[np.where(wrong_class_inst == wrong_pred[0])]
                print(wrong_pred_inst)
                misclassified.append([wrong_pred[0], wrong_pred_inst])
            img = self._createImage(d[0], misclassified)
            fname = 'TopMisImg/' + '{0:0>3}'.format(top) + '_' + '{0:0>3}'.format(d[0]) + '.png'
            cv.imwrite(fname, img)
            top += 1

    def _createImage(self, correct, misclassified):
        imgsz = 50
        width = (2 + 2 + 11)*imgsz
        imgset = None
        for baditem in misclassified:
            wrongclass = baditem[0]
            badinst = baditem[1]
            rowcnt = len(badinst)//10
            if len(badinst) % 10 > 0:
                rowcnt += 1
            height = imgsz + rowcnt*imgsz
            #img = np.ones((height, width)) * 0.2
            img = np.zeros((height, width))  # Dark background
            r = np.array([1, 49])
            c = np.array([1, 49])

            r = r + imgsz
            img[r[0]:r[1], c[0]:c[1]] = self.clsimg[correct][0]
            c = c + 2*imgsz
            img[r[0]:r[1], c[0]:c[1]] = self.clsimg[wrongclass][0]
            c = c + 2*imgsz

            for i in range(len(badinst)):
                if (((i % 10)== 0) and ( i != 0)):
                    r = r + imgsz
                    c = np.array([1, 49]) + 4*imgsz
 
                img[r[0]:r[1], c[0]:c[1]] = self.tstset[badinst[i]]
                c = c + imgsz
            if imgset is None:
                imgset = img
            else:
                imgset = np.append(imgset, img, axis=0)
        cv.imshow('img', imgset)
        cv.waitKey(0)
        return 255-255*imgset




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python '+sys.argv[0]+' missed_data_file')
        sys.exit(1)

    tstimg='../../data/tst_dataset.pkl.gz'

    
    tamunic = unpickleData('../../data/tam_class_id.pkl.gz')

    analyzer = Analyzer(sys.argv[1], tstimg=tstimg)
    analyzer.getMisclassImage(12)
    stats = analyzer.getTopMisclassifications(156)
    cv.destroyAllWindows()
    print ('================= Top Misclassifications ===================')
    print ('------------------------------------------------------------')
    print ('Class\tUnic\tMis\tTot\t\tMisclassified as (class, count)')
    print ('------------------------------------------------------------')
    for d in stats:
        print (d[0], '\t', tamunic[d[0]], '\t', d[1], '\t', d[2], '\t', d[3])
