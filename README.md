# Novel seam carving method

# My method steps:

## 1. Pre-scaling
<p align=center>
<img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/dia3.png">
</p>

## 2. Create energy map

### a- Create gradient map
I used [Bi-Directional Cascade Network for Perceptual Edge Detection](https://arxiv.org/abs/1902.10903), CVPR 2019 and [this](https://github.com/pkuCactus/BDCN) pytorch implementation.

Download the pre-trained model [here](https://drive.google.com/file/d/1CmDMypSlLM6EAvOt5yjwUQ7O5w-xCm1n/view?usp=sharing).


### b- Create saliency map
I used [Pyramid Feature Attention Network for Saliency Detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.html),  CVPR 2019 and [this](https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection) pytorch implementation.

Download the pre-trained model [here](https://drive.google.com/file/d/1Sc7dgXCZjF4wVwBihmIry-Xk7wTqrJdr/view?usp=sharing).

### c- Use depth map
I assumed that the depth map is given as input to the system.

## 3. Create energy map: GM + SM + DM

## 4. Select the appropriate threshold for the transition from seam carving to scaling

## 5. Seam carving

## 6. Post-scaling

# Results

<p align=center>

Original image | 50 % width reduction | 60 % | 70 % | 80 % | 90 %
:-------------:|:--------------------:|:----:|:----:|:----:|:----:
<img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/images/teddy-v.png" width=200 height=170>  | <img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/result/teddy.png" width=100 height=170>  | <img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/result/teddy_60.png" width=80 height=170> | <img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/result/teddy_70.png" width=60 height=170> | <img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/result/teddy_80.png" width=40 height=170>  | <img src="https://github.com/farkoo/novel-seam-carving-method/blob/master/result/teddy_90.png" width=20 height=170>   

</p>

## Support

**Contact me @:**

e-mail:

* farzanehkoohestani2000@gmail.com

Telegram id:

* [@farzaneh_koohestani](https://t.me/farzaneh_koohestani)

## License
[MIT](https://github.com/farkoo/novel-seam-carving-method/blob/master/LICENSE)
&#0169; 
[Farzaneh Koohestani](https://github.com/farkoo)


