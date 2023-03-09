# Q-Pensieve: Boosting Sample Efficiency of Multi-Objective RL Through Memory Sharing of Q-Snapshots

![](https://i.imgur.com/7Zuv6Jw.png)

**Q-Pensieve: Boosting Sample Efficiency of Multi-Objective RL Through Memory Sharing of Q-Snapshots**

Wei Hung\*, Bo Kai Huang\*, [Ping-Chun Hsieh](https://pinghsieh.github.io/), Xi Liu

*The Eleventh Conference on International Conference on Learning Representations (ICLR 2023)*

[[Paper]](https://openreview.net/pdf?id=AwWaBXLIJE)

## Requirements
- Python 3.7.9
- Pytorch 1.3.1
```
pip install -r requirements.txt
```
## Examples
### Training
You can directly use the following command to train.
```shell
python main.py --seed 1 --prefer 4 --buf_num 4
```
You can also edit the hyperparamter in configs.
### Testing
```
python test.py --prefer 4 --buf_num 4
```
