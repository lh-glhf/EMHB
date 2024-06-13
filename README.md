# EMHB: Easy Medium Hard Benchmark for Long Sequence Time Series Forecasting
 
This paper introduces **E**asy **M**edium **H**ard **B**enchmark(EMHB), a comprehensive benchmark for evaluating time series forecasting models. This benchmark has been developed in response to recent advancements in deep learning methods in time series forecasting task. EMHB evaluates the performance of 14 major forecasting models on carefully designed dataset collections, including commonly used 8 datasets and 9 new datasets that we have introduced. In addition, we present a novel method to measure the quality of forecasting time series datasets, separating datasets into three collections with different forecasting difficulty: Easy, Medium and Hard. By conducting a thorough analysis of the experimental results, including the overall prediction performance of models and their generalization across different prediction lengths and datasets, we provide a comprehensive measurement of the predictive ability of current time series forecasting models. By demonstrating and analyzing the issue of forecast degradation, our findings show that error metrics alone are insufficient for evaluating these forecasting models. We also discuss the causes of forecast degradation and provide our solutions. Our EMHB will serve as a valuable resource for advancing research in time series forecasting problem.

## Datasets accessibility

Datasets are available at here: [Google Drive](https://drive.google.com/drive/folders/1qeEySmFyE8bDvP4Ab_s1F7bYgKwQcHRe). All datasets are in .csv format and can be easily processed with 
Excel, Python and other programming language. Our Google Drive link will remain open indefinitely. If this link becomes inaccessible due to any unforeseen circumstances, we will provide a download link for the dataset on another sharing website.

Croissant metadata records are in EMHB/datasets/.

Our datasets ETTn and TEC are licensed under the ATTRIBUTION-NONCOMMERCIAL-NODERIVS 4.0 INTERNATIONAL(CC BY-NC-ND 4.0) license.

## Acknowledgement
We are grateful for the code and datasets provided by the following GitHub repositories:

https://github.com/thuml/TimesNet

https://github.com/wanghq21/MICN

https://github.com/luodhhh/ModernTCN

https://github.com/cure-lab/SCINet

https://github.com/philipperemy/n-beats

https://github.com/cure-lab/LTSF-Linear

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/MAZiqing/FEDformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/yuqinie98/patchtst

https://github.com/thuml/iTransformer

https://github.com/zongyi-li/fourier_neural_operator

https://github.com/ts-kim/RevIN

https://github.com/thuml/Flowformer

https://github.com/shreyansh26/FlashAttention-PyTorch

https://github.com/philipperemy/n-beats
