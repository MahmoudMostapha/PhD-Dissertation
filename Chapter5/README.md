# Learning from Complex Neuroimaging Datasets

This is an archive of the source code and learned models related to my PhD dissertation at UNC-Chapel Hill 

## Abstract

Advancements in Magnetic Resonance Imaging (MRI) allowed for the early diagnosis of neurodevelopmental disorders and neurodegenerative diseases. Neuroanatomical abnormalities in the cerebral cortex are often investigated by examining group-level differences of brain morphometric measures extracted from highly-sampled cortical surfaces. However, group-level differences do not allow for individual-level outcome prediction critical for the application to clinical practice. 

 Despite the success of MRI-based deep learning frameworks, critical issues have been identified: (1) extracting accurate and reliable local features from the cortical surface, (2) determining a parsimonious subset of cortical features for correct disease diagnosis, (3) learning directly from a non-Euclidean high-dimensional feature space, (4) improving the robustness of multi-task multi-modal models, and (5) identifying anomalies in imbalanced and heterogeneous settings.
  
This dissertation describes novel methodological contributions to tackle the challenges above. First, I introduce a Laplacian-based method for quantifying local Extra-Axial Cerebrospinal Fluid (EA-CSF) from structural MRI. Next, I describe a deep learning approach for combining local EA-CSF with other morphometric cortical measures for early disease detection. Then, I propose a data-driven approach for extending convolutional learning to non-Euclidean manifolds such as cortical surfaces. I also present a unified framework for robust multi-task learning from imaging and non-imaging information. Finally, I propose a semi-supervised generative approach for the detection of samples from untrained classes in imbalanced and heterogeneous developmental datasets.
  
The proposed methodological contributions are evaluated by applying them to the early detection of Autism Spectrum Disorder (ASD) in the first year of the infant’s life. Also, the aging human brain is examined in the context of studying different stages of Alzheimer’s Disease (AD).
