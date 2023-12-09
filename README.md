## 更新：

2023年12月9日
1. circles_spatial2D模型的python版本有问题，官方还没给出答案。
2. boids_spatial3D_bounded模型的python版本问题7月份已经解决，未记录！

2023年7月27日
完成了pd_punish模型。

2023年7月10日
diffusion模型添加了退出函数。

完成了ensemble模型的python_rtc版本。


2023年6月30日
diffusion模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |3s|3070Ti|
| python rtc和cuda | 10000   |6s|3070Ti|


2023年6月28日
添加并验证了game_of_life模型的python_rtc和c++版本。

game_of_life模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |11.24s|3070Ti|
| python rtc和cuda | 10000   |19s|3070Ti|

2023年6月27日
添加并验证了boids_spatial3D_bounded模型的python_rtc版本和c++版本。python版本有问题！

boids_spatial3D_bounded模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |11.84s|3070Ti|
| python rtc和cuda | 10000   |6.9s|3070Ti|



2023年6月26日
添加并验证了circles_bruteforce模型的python_rtc版本和c++版本。
添加并验证了boids_bruteforce模型的python_rtc版本和c++版本。
添加并验证了boids_spatial3D_wrapped模型的python_rtc版本和python版本。

circles_bruteforce模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 1000   |23s|3070Ti|
| python rtc和cuda | 1000  |7.35s|3070Ti|

boids_bruteforce模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |57.3s|3070Ti|
| python rtc和cuda | 10000   |16.5s|3070Ti|


boids_spatial3D_wrapped模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| python | 10000   |67s|3070Ti|
| python rtc和cuda | 10000   |18.27s|3070Ti|

2023年6月23日
添加并验证了circles_spatial2D模型的python_rtc, c++版本。python版本有问题。
添加并验证了circles_spatial3D模型的python_rtc版本和c++版本。

circles_spatial2D模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |8.65|3070Ti|
| python rtc和cuda | 10000  |7s|3070Ti|
| python | 10000  |7s|3070Ti|

circles_spatial3D模型测速
| 运行组合 | 测试周期数 |消耗时间|显卡|
|------|------|------|------|
| c++和cuda | 10000   |9.68s|3070Ti|
| python rtc和cuda | 10000  |7.5s|3070Ti|
