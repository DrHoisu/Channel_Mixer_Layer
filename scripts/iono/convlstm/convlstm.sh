# 1channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/1channel/convlstm_1imgchannel_0auxchannel.py --ex_name iono/convlstm/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/1channel/convlstm_1imgchannel_0auxchannel.py --ex_name iono/convlstm/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/concatenate_4channel/convlstm_1imgchannel_3auxchannel.py --ex_name iono/convlstm/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/concatenate_4channel/convlstm_1imgchannel_3auxchannel.py --ex_name iono/convlstm/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/concatenate_9channel/convlstm_1imgchannel_8auxchannel.py --ex_name iono/convlstm/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/concatenate_9channel/convlstm_1imgchannel_8auxchannel.py --ex_name iono/convlstm/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/concatenate_16channel/convlstm_1imgchannel_15auxchannel.py --ex_name iono/convlstm/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/concatenate_16channel/convlstm_1imgchannel_15auxchannel.py --ex_name iono/convlstm/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/mix_4channel/convlstm_1imgchannel_3auxchannel.py --ex_name iono/convlstm/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/mix_4channel/convlstm_1imgchannel_3auxchannel.py --ex_name iono/convlstm/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/mix_9channel/convlstm_1imgchannel_8auxchannel.py --ex_name iono/convlstm/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/mix_9channel/convlstm_1imgchannel_8auxchannel.py --ex_name iono/convlstm/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/convlstm/mix_16channel/convlstm_1imgchannel_15auxchannel.py --ex_name iono/convlstm/mix_16channel
# test
python tools/test.py --test -d iono -c configs/iono/convlstm/mix_16channel/convlstm_1imgchannel_15auxchannel.py --ex_name iono/convlstm/mix_16channel/
