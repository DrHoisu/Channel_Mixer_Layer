# 1channel
# train
#python tools/train.py -d iono -c configs/iono/mau/1channel/mau_1imgchannel_0auxchannel.py --ex_name iono/mau/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/mau/1channel/mau_1imgchannel_0auxchannel.py --ex_name iono/mau/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/mau/concatenate_4channel/mau_1imgchannel_3auxchannel.py --ex_name iono/mau/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/mau/concatenate_4channel/mau_1imgchannel_3auxchannel.py --ex_name iono/mau/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/mau/concatenate_9channel/mau_1imgchannel_8auxchannel.py --ex_name iono/mau/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/mau/concatenate_9channel/mau_1imgchannel_8auxchannel.py --ex_name iono/mau/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/mau/concatenate_16channel/mau_1imgchannel_15auxchannel.py --ex_name iono/mau/concatenate_16channel --auto_resume
# test
python tools/test.py --test -d iono -c configs/iono/mau/concatenate_16channel/mau_1imgchannel_15auxchannel.py --ex_name iono/mau/concatenate_16channel/

# mix_4channel
# train
# python tools/train.py -d iono -c configs/iono/mau/mix_4channel/mau_1imgchannel_3auxchannel.py --ex_name iono/mau/mix_4channel --epoch 15
# test
# python tools/test.py --test -d iono -c configs/iono/mau/mix_4channel/mau_1imgchannel_3auxchannel.py --ex_name iono/mau/mix_4channel/

# mix_9channel
# train
# python tools/train.py -d iono -c configs/iono/mau/mix_9channel/mau_1imgchannel_8auxchannel.py --ex_name iono/mau/mix_9channel --epoch 15
# test
# python tools/test.py --test -d iono -c configs/iono/mau/mix_9channel/mau_1imgchannel_8auxchannel.py --ex_name iono/mau/mix_9channel/

# mix_16channel
# train
# python tools/train.py -d iono -c configs/iono/mau/mix_16channel/mau_1imgchannel_15auxchannel.py --ex_name iono/mau/mix_16channel --epoch 15
# test
# python tools/test.py --test -d iono -c configs/iono/mau/mix_16channel/mau_1imgchannel_15auxchannel.py --ex_name iono/mau/mix_16channel/
