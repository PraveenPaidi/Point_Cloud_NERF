# Point_Cloud_NERF


The task of reconstructing 3D scenes from image
 data and producing new photo-realistic perspectives is a crucial
 aspect of computer vision. While classical NeRF [1] uses images
 and pose data for volumetric rendering, our model leverages
 point clouds. However, due to the sparsity of point clouds,
 creating complete images becomes challenging. To overcome this
 challenge, we employ a recurrent neural network pipeline that
 transforms point clouds into dense point clouds, allowing for
 feasible construction even in subpar areas. Additionally, we
 propose point cloud registration using a neural network and
 COLMAP to extract the optimal point cloud in sparse areas.
 To address the training time limitations of Classical NeRF, we
 introduce a RANSAC model that avoids void space training by
 identifying the object’s near space probability from multiview
 stereo probability maps. To complete this pipeline, we implement
 it in a volumetric rendering algorithm to obtain novel views.

 The project pipeline is :

<img width="567" alt="kkk" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/028e6621-2f53-4dbc-9c82-d352ad341809">

The MLP of NERF :

<img width="338" alt="Screenshot 2023-12-19 050416" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/ff129a69-5da8-408a-b8ee-a921eaaa9823">

The depth point cloud representation:

<img width="349" alt="ddd" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/a612210b-2e47-41ca-977c-35d92a4b8971">

The depth loss with RGB :

<img width="410" alt="Screenshot 2023-12-19 050335" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/a6f1720b-8848-437c-9d44-348c4b3be7f6">



The depth loss with RGBD:


<img width="398" alt="Screenshot 2023-12-19 050350" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/d3984d39-a617-4c71-a017-2ea709867ab3">


Diff of registered and simple depth cloud loss function:


<img width="605" alt="Screenshot 2023-12-19 050302" src="https://github.com/PraveenPaidi/Point_Cloud_NERF/assets/120610889/144243df-8a3a-4605-993c-b35912905308">









 
