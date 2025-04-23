# EMHB: Easy Medium Hard Benchmark for Long Sequence Time Series Forecasting
 
In this paper, we address the challenges in long-term multivariate time series forecasting, which are the inadequacy of current evaluation metrics and the lack of diverse datasets. Despite advancements in deep learning models for time series prediction, existing error metrics remain insufficient for reflecting the true quality of long-term forecasts, potentially leading to models that underperform in practical applications. Additionally, the scarcity of datasets restricts the comparability of model performance, particularly in emerging areas like time series foundation models. To address these issues, this work introduces a benchmark for long sequence time series forecasting, featuring nine new datasets and proposing novel evaluation criteria that incorporate both error and non-error metrics. We also develop a method for assessing dataset quality, introduce a new dataset collection based on predictive difficulty, and propose a comprehensive benchmark named **E**asy **M**edium **H**ard **B**enchmark (EMHB) to assess the performance of leading forecasting models. Our contributions aim to improve the comparability and efficiency of time series forecasting models, with a focus on supporting further research and practical applications in this field.

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
