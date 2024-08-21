# 1channel
# train
#python tools/train.py -d iono -c configs/iono/mim/1channel/mim_1imgchannel_0auxchannel.py --ex_name iono/mim/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/mim/1channel/mim_1imgchannel_0auxchannel.py --ex_name iono/mim/1channel/

# concatenate_4channel
# train
# python tools/train.py -d iono -c configs/iono/mim/concatenate_4channel/mim_1imgchannel_3auxchannel.py --ex_name iono/mim/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/concatenate_4channel/mim_1imgchannel_3auxchannel.py --ex_name iono/mim/concatenate_4channel/

# concatenate_9channel
# train
# python tools/train.py -d iono -c configs/iono/mim/concatenate_9channel/mim_1imgchannel_8auxchannel.py --ex_name iono/mim/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/concatenate_9channel/mim_1imgchannel_8auxchannel.py --ex_name iono/mim/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/mim/concatenate_16channel/mim_1imgchannel_15auxchannel.py --ex_name iono/mim/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/concatenate_16channel/mim_1imgchannel_15auxchannel.py --ex_name iono/mim/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/mim/mix_4channel/mim_1imgchannel_3auxchannel.py --ex_name iono/mim/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/mix_4channel/mim_1imgchannel_3auxchannel.py --ex_name iono/mim/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/mim/mix_9channel/mim_1imgchannel_8auxchannel.py --ex_name iono/mim/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/mix_9channel/mim_1imgchannel_8auxchannel.py --ex_name iono/mim/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/mim/mix_16channel/mim_1imgchannel_15auxchannel.py --ex_name iono/mim/mix_16channel
# test
python tools/test.py --test -d iono -c configs/iono/mim/mix_16channel/mim_1imgchannel_15auxchannel.py --ex_name iono/mim/mix_16channel/
