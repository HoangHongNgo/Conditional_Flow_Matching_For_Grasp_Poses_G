CUDA_VISIBLE_DEVICES=0 python test.py --model ecotransgrasp --save_dir results/ecotransgrasp/test_ep10_seen --checkpoint_path results/ecotransgrasp/ecograsp_epoch10.tar --camera kinect --dataset_root /media/dsp520/Grasp_2T/graspnet --test_mode similar --inference --batch_size 32

kinect, AP Seen=0.5758219612363848
seen testing, AP 0.8=0.6879559933002386, AP 0.4=0.4975348317630875

kinect, AP=0.4467528908946281, AP Similar=0.4467528908946281
similar testing, AP 0.8=0.5576409052661799, AP 0.4=0.3353710282698435