# 1channel
# train
#python tools/train.py -d iono -c configs/iono/tau/1channel/tau_1imgchannel_0auxchannel.py --ex_name iono/tau/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/tau/1channel/tau_1imgchannel_0auxchannel.py --ex_name iono/tau/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/tau/concatenate_4channel/tau_1imgchannel_3auxchannel.py --ex_name iono/tau/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/tau/concatenate_4channel/tau_1imgchannel_3auxchannel.py --ex_name iono/tau/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/tau/concatenate_9channel/tau_1imgchannel_8auxchannel.py --ex_name iono/tau/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/tau/concatenate_9channel/tau_1imgchannel_8auxchannel.py --ex_name iono/tau/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/tau/concatenate_16channel/tau_1imgchannel_15auxchannel.py --ex_name iono/tau/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/tau/concatenate_16channel/tau_1imgchannel_15auxchannel.py --ex_name iono/tau/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/tau/mix_4channel/tau_1imgchannel_3auxchannel.py --ex_name iono/tau/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/tau/mix_4channel/tau_1imgchannel_3auxchannel.py --ex_name iono/tau/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/tau/mix_9channel/tau_1imgchannel_8auxchannel.py --ex_name iono/tau/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/tau/mix_9channel/tau_1imgchannel_8auxchannel.py --ex_name iono/tau/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/tau/mix_16channel/tau_1imgchannel_15auxchannel.py --ex_name iono/tau/mix_16channel --auto_resume --epoch 15
# test
python tools/test.py --test -d iono -c configs/iono/tau/mix_16channel/tau_1imgchannel_15auxchannel.py --ex_name iono/tau/mix_16channel/
