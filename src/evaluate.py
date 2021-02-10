import pylab

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def main():
    for j in range(0, 13):
        print(j)
        #         try:
        dataDir = '..'
        dataType = 'val2014'
        algName = f'img_to_txt_state_{j}.tarseq_to_txt_state_11.tar'
        annFile = '%s/annotations/_captions_%s.json' % (dataDir, dataType)
        subtypes = ['results', 'evalImgs', 'eval']
        [resFile, evalImgsFile, evalFile] = ['%s/results/captions_%s_%s_%s.json' % (dataDir, dataType, algName, subtype)
                                             for subtype in subtypes]
        coco = COCO(annFile)
        cocoRes = coco.loadRes(resFile)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
    #         except:
    #             print "This is an error message!"


if __name__ == '__main__':
    main()
