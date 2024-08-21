# 1channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/1channel/predrnn_1imgchannel_0auxchannel.py --ex_name iono/predrnn/1channel/
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/1channel/predrnn_1imgchannel_0auxchannel.py --ex_name iono/predrnn/1channel/

# concatenate_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/concatenate_4channel/predrnn_1imgchannel_3auxchannel.py --ex_name iono/predrnn/concatenate_4channel --auto_resume
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/concatenate_4channel/predrnn_1imgchannel_3auxchannel.py --ex_name iono/predrnn/concatenate_4channel/

# concatenate_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/concatenate_9channel/predrnn_1imgchannel_8auxchannel.py --ex_name iono/predrnn/concatenate_9channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/concatenate_9channel/predrnn_1imgchannel_8auxchannel.py --ex_name iono/predrnn/concatenate_9channel/

# concatenate_16channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/concatenate_16channel/predrnn_1imgchannel_15auxchannel.py --ex_name iono/predrnn/concatenate_16channel --auto_resume
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/concatenate_16channel/predrnn_1imgchannel_15auxchannel.py --ex_name iono/predrnn/concatenate_16channel/

# mix_4channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/mix_4channel/predrnn_1imgchannel_3auxchannel.py --ex_name iono/predrnn/mix_4channel
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/mix_4channel/predrnn_1imgchannel_3auxchannel.py --ex_name iono/predrnn/mix_4channel/

# mix_9channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/mix_9channel/predrnn_1imgchannel_8auxchannel.py --ex_name iono/predrnn/mix_9channel --epoch 20 --test
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/mix_9channel/predrnn_1imgchannel_8auxchannel.py --ex_name iono/predrnn/mix_9channel/

# mix_16channel
# train
#python tools/train.py -d iono -c configs/iono/predrnn/mix_16channel/predrnn_1imgchannel_15auxchannel.py --ex_name iono/predrnn/mix_16channel --epoch 20 --auto_resume
# test
python tools/test.py --test -d iono -c configs/iono/predrnn/mix_16channel/predrnn_1imgchannel_15auxchannel.py --ex_name iono/predrnn/mix_16channel/
