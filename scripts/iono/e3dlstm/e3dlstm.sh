# 1channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/1channel/e3dlstm_1imgchannel_0auxchannel.py --ex_name iono/e3dlstm/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/e3dlstm/1channel/e3dlstm_1imgchannel_0auxchannel.py --ex_name iono/e3dlstm/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/concatenate_4channel/e3dlstm_1imgchannel_3auxchannel.py --ex_name iono/e3dlstm/concatenate_4channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/concatenate_4channel/e3dlstm_1imgchannel_3auxchannel.py --ex_name iono/e3dlstm/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/concatenate_9channel/e3dlstm_1imgchannel_8auxchannel.py --ex_name iono/e3dlstm/concatenate_9channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/concatenate_9channel/e3dlstm_1imgchannel_8auxchannel.py --ex_name iono/e3dlstm/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/concatenate_16channel/e3dlstm_1imgchannel_15auxchannel.py --ex_name iono/e3dlstm/concatenate_16channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/concatenate_16channel/e3dlstm_1imgchannel_15auxchannel.py --ex_name iono/e3dlstm/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/mix_4channel/e3dlstm_1imgchannel_3auxchannel.py --ex_name iono/e3dlstm/mix_4channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/mix_4channel/e3dlstm_1imgchannel_3auxchannel.py --ex_name iono/e3dlstm/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/mix_9channel/e3dlstm_1imgchannel_8auxchannel.py --ex_name iono/e3dlstm/mix_9channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/mix_9channel/e3dlstm_1imgchannel_8auxchannel.py --ex_name iono/e3dlstm/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/e3dlstm/mix_16channel/e3dlstm_1imgchannel_15auxchannel.py --ex_name iono/e3dlstm/mix_16channel
# test
#python tools/test.py --test -d iono -c configs/iono/e3dlstm/mix_16channel/e3dlstm_1imgchannel_15auxchannel.py --ex_name iono/e3dlstm/mix_16channel/
