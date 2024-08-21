# 1channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/1channel/predrnnpp_1imgchannel_0auxchannel.py --ex_name iono/predrnnpp/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/1channel/predrnnpp_1imgchannel_0auxchannel.py --ex_name iono/predrnnpp/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/concatenate_4channel/predrnnpp_1imgchannel_3auxchannel.py --ex_name iono/predrnnpp/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/concatenate_4channel/predrnnpp_1imgchannel_3auxchannel.py --ex_name iono/predrnnpp/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/concatenate_9channel/predrnnpp_1imgchannel_8auxchannel.py --ex_name iono/predrnnpp/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/concatenate_9channel/predrnnpp_1imgchannel_8auxchannel.py --ex_name iono/predrnnpp/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/concatenate_16channel/predrnnpp_1imgchannel_15auxchannel.py --ex_name iono/predrnnpp/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/concatenate_16channel/predrnnpp_1imgchannel_15auxchannel.py --ex_name iono/predrnnpp/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/mix_4channel/predrnnpp_1imgchannel_3auxchannel.py --ex_name iono/predrnnpp/new_mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/mix_4channel/predrnnpp_1imgchannel_3auxchannel.py --ex_name iono/predrnnpp/new_mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnpp/mix_9channel/predrnnpp_1imgchannel_8auxchannel.py --ex_name iono/predrnnpp/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/mix_9channel/predrnnpp_1imgchannel_8auxchannel.py --ex_name iono/predrnnpp/mix_9channel/

# mix_16channel
# train
#CUDA_VISIBLE_DEVICES=1 python tools/train.py -d iono -c configs/iono/predrnnpp/mix_16channel/predrnnpp_1imgchannel_15auxchannel.py --ex_name iono/predrnnpp/mix_16channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnpp/mix_16channel/predrnnpp_1imgchannel_15auxchannel.py --ex_name iono/predrnnpp/mix_16channel/
