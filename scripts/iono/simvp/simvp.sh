# 1channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/1channel/simvp_1imgchannel_0auxchannel.py --ex_name iono/simvp/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/simvp/1channel/simvp_1imgchannel_0auxchannel.py --ex_name iono/simvp/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/concatenate_4channel/simvp_1imgchannel_3auxchannel.py --ex_name iono/simvp/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/concatenate_4channel/simvp_1imgchannel_3auxchannel.py --ex_name iono/simvp/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/concatenate_9channel/simvp_1imgchannel_8auxchannel.py --ex_name iono/simvp/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/concatenate_9channel/simvp_1imgchannel_8auxchannel.py --ex_name iono/simvp/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/concatenate_16channel/simvp_1imgchannel_15auxchannel.py --ex_name iono/simvp/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/concatenate_16channel/simvp_1imgchannel_15auxchannel.py --ex_name iono/simvp/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/mix_4channel/simvp_1imgchannel_3auxchannel.py --ex_name iono/simvp/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/mix_4channel/simvp_1imgchannel_3auxchannel.py --ex_name iono/simvp/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/mix_9channel/simvp_1imgchannel_8auxchannel.py --ex_name iono/simvp/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/mix_9channel/simvp_1imgchannel_8auxchannel.py --ex_name iono/simvp/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/simvp/mix_16channel/simvp_1imgchannel_15auxchannel.py --ex_name iono/simvp/mix_16channel
# test
python tools/test.py --test -d iono -c configs/iono/simvp/mix_16channel/simvp_1imgchannel_15auxchannel.py --ex_name iono/simvp/mix_16channel/
