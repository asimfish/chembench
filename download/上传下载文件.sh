aWIaCSEKOf5feGStf68cBy3t5H08UqiZ
Xxm3kKCbBGM8
##上传文件
rsync -avzP /home/psibot/chembench/data/zarr_rgbm liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/

##下载文件
rsync -avzP liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/outputs/download/grasp_rgb /home/psibot/chembench/data/outputs/download/grasp_rgb/50ml烧杯/
rsync -avzP liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/outputs/2025.12.18 /home/psibot/chembench/data/outputs/

# /share_data/liyufeng/code/chembench/data/zarr_rgb/motion_plan/grasp
# /share_data/liyufeng/code/chembench/data/outputs/download/grasp_rgb/100ml烧杯/epoch=0500-train_loss=0.007.ckpt

rsync -avzP liyufeng@124.222.194.84:/share_data/liyufeng/code/chembench/data/outputs/grasp_state/50ml玻璃烧杯/20251219_054455_n100/checkpoints/epoch=1700-train_loss=0.001.ckpt /home/psibot/chembench/data/outputs/grasp_state/
