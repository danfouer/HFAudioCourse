# Hugging Face Audio course 中文交互式教程
## 前言
我用二十天时间，拿到Hugging Face Audio course (您可以到https://huggingface.co/learn/audio-course/chapter0/introduction 注册该课程）的卓越证书，并将课程转换为中文交互式教程，希望能够帮助到更多的人。
同时我翻译了Transformers in Speech Processing: A Survey 一文
https://s0tiijs5zp.feishu.cn/docx/WE4Jd12DaonhAUx48BScs1wBn5d?from=from_copylink，可供快速的了解此领域的概貌。

## 课程结构
该课程分为几个单元，深入涵盖各种主题：

    第 1 单元：了解处理音频数据的细节，包括音频处理技术和数据准备。

    第 2 单元：了解音频应用程序并学习如何使用 🤗 Transformers 管道执行不同的任务，例如音频分类和语音识别。

    第 3 单元：探索音频转换器架构，了解它们有何不同以及它们最适合执行哪些任务。

    第 4 单元：学习如何构建自己的音乐流派分类器。

    第 5 单元：深入研究语音识别并建立模型来转录会议录音。

    第 6 单元：学习如何从文本生成语音。

    第 7 单元：学习如何使用 Transformer 构建真实世界的音频应用程序。

每个单元都包含一个理论部分，您将深入了解基本概念和技术。在整个课程中，我们提供测验来帮助您测试您的知识并加强您的学习。有些章节还包括实践练习，您将有机会应用所学知识。

在课程结束时，您将在使用音频数据转换器方面打下坚实的基础，并有能力将这些技术应用于广泛的音频相关任务。

## Google Colab安装(推荐)
在Unit1-Unit7*.ipynb文件的第一行代码中，添加如下代码，即可在Google Colab中运行。
```
!pip install --upgrade --quiet pip
!pip install --quiet datasets
!pip install --quiet git+https://github.com/huggingface/transformers.git
!pip install --upgrade --quiet accelerate
!pip install --quiet gradio

后面缺啥就!pip install啥

```
## 本地安装
（本地有GPU,并以安装好CUDA、cudnn,它们的安装参考 https://blog.csdn.net/takedachia/article/details/130375718）
```
conda create --name HFAudio python=3.10.12 -y
conda activate HFAudio
cd Videos/HFAUDIOCOURSE #进入课程文件夹
pip3 install --upgrade pip
pip3 install jupyter
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #根据自己的CUDA版本安装
pip install --upgrade --quiet pip
pip install --quiet datasets
pip install --quiet git+https://github.com/huggingface/transformers.git
pip install --upgrade --quiet accelerate
pip install --quiet gradio
pip install --quiet soundfile==0.12.1
pip install --quiet soundfile librosa==0.10.1
pip install s--quiet sentencepiece
jupyter notebook
后面缺啥就pip install啥
conda env remove --name HFAudio #玩崩了就重来