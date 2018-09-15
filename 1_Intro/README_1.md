# 준비운동
간단한 문제를 Q-Learning을 통해 풀어보는 튜토이알입니다. 이미 RL 기본 컨셉에 대해서 알고 계셔야 하며, 혹시 모르신다면, [여기](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)와 [여기], 또는 [여기]를 읽어 보실 것을 부탁 드립니다. 여기에서는 Neural Network를 사용하지 않는 간단한 예시로 시작하도록 하겠습니다. 
<br />
<br />
Mobile game Numero를 이 단계에서 바로 적용하기엔 난이도가 높을 것 같아서, 여기에선 문제를 간단하게 변형해 보도록 하겠습니다. 아래와 같이 타일이 5 X 5로 있고, 우리의 목표는 

| ------------- | ------------- | ------------- | ------------- |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| ------------- | ------------- | ------------- | ------------- |

## Q-Learning이란?
RL을 할 수 있는 approaches 중 하나는 Q-learning 입니다. 각각의 상황 별로 (`여기서 이렇게 하면` - `이렇게 된다`)를 정리해 놓은 테이블이라고 생각하시면 됩니다. 즉 (`state *s*`, `action *a*`) - (`value estimations *v*`) 으로 표현됩니다. 

## Q table

Q table은 모든 경우의 수를 다 표현 할 수 있어야 합니다. Row는 state이 되고, column은 action이 됩니다. 
+---------+---------+---------+---------+---------+
|         |         |         |         |         |
+---------+---------+---------+---------+---------+
|         |         |         |         |         |
+---------+---------+---------+---------+---------+
|         |         |         |         |         |
+---------+---------+---------+---------+---------+
|         |         |         |         |         |
+---------+---------+---------+---------+---------+
|         |         |         |         |         |
+---------+---------+---------+---------+---------+

### Source
* [An introduction to Reinforcement Learning](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)
* [Reinforcement Learning with Q tables](https://itnext.io/reinforcement-learning-with-q-tables-5f11168862c8)
 