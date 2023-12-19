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




 
