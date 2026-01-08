# aWIaCSEKOf5feGStf68cBy3t5H08UqiZ
# Xxm3kKCbBGM8
mn4Gexba6Zup3+vy
# pro35:rreJKG7QY4uF2N4Q4ZFJvnlCndY7CntZ
# pro27:6Bo7ecM78dTVJnXaA0BC4g8EDJaRNA6U

ssh：
pro35:
    ssh -p 8322 liyufeng@iGxnRyQHelg9PQBJ@118.89.130.182
    rreJKG7QY4uF2N4Q4ZFJvnlCndY7CntZ

pro27:
    ssh -p 8322 'liyufeng@HkQ1wUnZgCuDY2Yq@118.89.130.182'
    6Bo7ecM78dTVJnXaA0BC4g8EDJaRNA6U

pro13:
    ssh -p 8322 'liyufeng@ISVc3gdW3EA1RO4v@118.89.130.182'
    aWIaCSEKOf5feGStf68cBy3t5H08UqiZ
    
4090:
    ssh -p 5172 liyufeng@222.178.211.73
    mn4Gexba6Zup3+vy



上传到4090:
rsync -avzP -e "ssh -p 5172" /share_data/liyufeng/code/chembench/diffusion_policy/data/outputs/rgb_based/grasp_rgbm/finished_latest  liyufeng@222.178.211.73:/home/liyufeng/chembench/chembench/data/model/dp/rgb_based/grasp/



从4090下载
rsync -avzP  liyufeng@222.178.211.73:/home/liyufeng/chembench /home/psibot/chembench/data/download/final_real/model



##上传文件
rsync -avzP /home/psibot/chembench/data/zarr_gt_point_1024 liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/3D-Diffusion-Policy/data/data/origion/


rsync -avzP /home/psibot/chembench/data/zarr_final/motion_plan/handover liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/final_real/data/


rsync -avzP /home/psibot/chembench/data/zarr_gt_point liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/3D-Diffusion-Policy/data/data/origion/



# /home/psibot/chembench/data/zarr_rgbm_new/motion_plan/grasp/100ml玻璃烧杯_rgbm.zarr
##下载文件
# rsync -avzP liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/1226/100烧杯_无速度_normal_depth/20251226_050936_n50_nd/checkpoints/epoch=5550-train_loss=0.002.ckpt /home/psibot/chembench/data/outputs/download/1226/100烧杯_无速度_normal_depth/20251226_050936_n50_nd/checkpoints/

rsync -avzP liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/act/checkpoints/chunk_other /home/psibot/chembench/act/

# /share_data/liyufeng/code/chembench/data/download/grasp_rgbm/采集
# /share_data/liyufeng/code/chembench/data/download/grasp_rgbm/坩堝
# /share_data/liyufeng/code/chembench/data/zarr_rgb/motion_plan/grasp
# /share_data/liyufeng/code/chembench/data/outputs/download/grasp_rgb/100ml烧杯/epoch=0500-train_loss=0.007.ckpt