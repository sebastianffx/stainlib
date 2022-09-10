This project investigates the interplay between color augmentation and adversarial feature learning to address the variability in tissue preparation methods that make substantial changes in the appearance of  digitized histology images and that hampers the performance of deep learning networks when applied to computational pathology tasks.

For this purpose, we design a domain adversarial framework in which color augmentation is used in conjunction with domain invariant training of deep convolutional networks. We test our approach in two open access datasets and provide the steps and code to reproduce each result reported in the paper.

If you find this code useful, consider citing the accompanying article:

Otálora, Sebastian, et al. "([Staining invariant features for improving generalization of deep convolutional neural networks in computational pathology](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00198/full))." Frontiers in Bioengineering and Biotechnology 7 (2019): 198.



Here are the steps to reproduce each set of experiments:

Mitosis detection in TUPAC:
* Have in a local path the ([TUPAC dataset](http://tupac.tue-image.nl/node/3))
* Create the patches using the coordinates located in datasets_utils/tupac

Gleason grading using the subset of Zürich prostate TMA dataset (subset from *) and patches from diagnostic WSI of TCGA:
* Download the [dataset](https://wetransfer.com/downloads/b33c6eda5df597b2fe375a2162be535f20190719142214/25afc2d4546196eb08825d48316c0c8720190719142214/d04b56) (if you have problems downloading it, drop me an email to juan.otalora [AT] etu.unige.ch)

* Run either baseline.py or dann_experiment.py



*Arvaniti, Eirini; Fricker, Kim; Moret, Michael; Rupp, Niels; Hermanns, Thomas; Fankhauser, Christian; Wey, Norbert; Wild, Peter; Rüschoff, Jan Hendrik; Claassen, Manfred, 2018, "Replication Data for: Automated Gleason grading of prostate cancer tissue microarrays via deep learning.", https://doi.org/10.7910/DVN/OCYCMP, Harvard Dataverse, V1
