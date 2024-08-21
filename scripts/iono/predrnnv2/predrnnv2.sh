# 1channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/1channel/predrnnv2_1imgchannel_0auxchannel.py --ex_name iono/predrnnv2/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/1channel/predrnnv2_1imgchannel_0auxchannel.py --ex_name iono/predrnnv2/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/concatenate_4channel/predrnnv2_1imgchannel_3auxchannel.py --ex_name iono/predrnnv2/concatenate_4channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/concatenate_4channel/predrnnv2_1imgchannel_3auxchannel.py --ex_name iono/predrnnv2/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/concatenate_9channel/predrnnv2_1imgchannel_8auxchannel.py --ex_name iono/predrnnv2/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/concatenate_9channel/predrnnv2_1imgchannel_8auxchannel.py --ex_name iono/predrnnv2/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/concatenate_16channel/predrnnv2_1imgchannel_15auxchannel.py --ex_name iono/predrnnv2/concatenate_16channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/concatenate_16channel/predrnnv2_1imgchannel_15auxchannel.py --ex_name iono/predrnnv2/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/mix_4channel/predrnnv2_1imgchannel_3auxchannel.py --ex_name iono/predrnnv2/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/mix_4channel/predrnnv2_1imgchannel_3auxchannel.py --ex_name iono/predrnnv2/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/mix_9channel/predrnnv2_1imgchannel_8auxchannel.py --ex_name iono/predrnnv2/mix_9channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/mix_9channel/predrnnv2_1imgchannel_8auxchannel.py --ex_name iono/predrnnv2/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/predrnnv2/mix_16channel/predrnnv2_1imgchannel_15auxchannel.py --ex_name iono/predrnnv2/mix_16channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnnv2/mix_16channel/predrnnv2_1imgchannel_15auxchannel.py --ex_name iono/predrnnv2/mix_16channel/
