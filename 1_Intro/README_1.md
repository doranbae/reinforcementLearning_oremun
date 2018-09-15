# 준비운동
간단한 문제를 Q-Learning을 통해 풀어보는 tutorial 입니다. 이미 RL 기본 컨셉에 대해서 알고 계셔야 하며, 혹시 모르신다면, [여기](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)와 [여기], 또는 [여기]를 읽어 보실 것을 부탁 드립니다. 그리고 걱정 안 하셔도 됩니다. 이번 tutorial은 Neural Network를 사용하지 않는 간단한 예시니까요.
<br />
<br />
Mobile game Numero를 이 단계에서 바로 적용하기엔 난이도가 높을 것 같아서, 여기에선 Numero를 간단하게 변형해 보도록 하겠습니다. 아래와 같이 보드게임 판이 5 X 5로 있고, 우리의 목표는 출발점에서 목적지까지 도착하는 것입니다. 물론 그냥 가면 심심하니, 가는 길에 함정을 하나씩 파 넣도록 하겠습니다. 함정은 피해서 가도록 하는 것을 룰로 하겠습니다. 
<br />
<br />
![게임 한 판](images/simple5by5.png)

## Q-Learning이란?
RL을 할 수 있는 approaches 중 하나는 Q-learning 입니다. 각각의 상황 별로 (`여기서 이렇게 하면` - `이렇게 된다`)를 정리해 놓은 테이블이라고 생각하시면 됩니다. 즉 (`state`, `action`) - (`value estimations`) 으로 표현됩니다. 
<br />
<br />
잘 안 와닿으시죠? 위 보드게임 판의 작은 타일 하나 하나 (A0, A1, A2, A3.... E4)가 각각 state라고 생각해 봅시다. 그리고 각 state에서 우리가 할 수 있는 움직임들(위, 아래, 오른쪽, 왼쪽으로 움직이기)이 action으로 정의를 해 봅니다. Q-learning에서 각 (s-a) pair 별 value를 계산한 다는 것은, 내가 어느 타일 위에 도착했을 경우, 위/아래/오른쪽/왼쪽으로 갔을 경우의 value를 미리 다 계산해 놓겠다는 거죠. 
<br />
<br />
사람은 딱 눈으로 봐도 대충 보이는 것들이 컴퓨터에게는 매우 힘이 드는 일이 됩니다. 예를 들어, 내가 B0이라는 타일에 도착 했을 때, 오른쪽으로 움직이는 것은 100% 자살행위라고 할 수 있죠 (C0 = Trap). 그러나 이렇게 눈으로 쉽게 보이는 것 외에, 사람도 약간 헷갈리는 경우의 수도 있을 수 있습니다. 예를 들어, 내가 A0에서 출발을 할 때, 오른쪽(B0)으로 움직이는 방법과  아래(A1)로 움직이는 방법 중 어느 action이 더 value가 높을 지는, 당장 숫자로 대답을 할 수 없습니다. 아무튼 이렇게 모든 타일에서의 모든 움직임에 대한 value를 계산하는 것이 바로 Q-learning입니다. 
<br />
<br />
이렇게 계산해서 뭘 하겠다는 건지? 그렇죠. 만약 모든 타일에서 모든 action에 대한 value가 있다면, 내가 어느 타일에 가던, 조금 더 value가 높은 방향으로 움직이면 되겠죠. 완벽한 컨닝 페이퍼는 아니지만, 마치 컨닝페이퍼 같은 역할을 하는 테이블이 될 것입니다. 
<br />
<br />
코딩으로 들어가기 전, 마지막으로 한가지 조건을 꾸역꾸역 넣어보겠습니다. 제가 위에 보여드린 이미지는 사람이 보기 편하라고 올려놓은 것입니다. 컴퓨터는 저런 그림을 줘도 모르니, 컴퓨터가 알아듣는 `numpy`의 array로 표현을 해야 하는 것입니다. 컴퓨터에게 보여주기 위해, state는 row로, action은 column으로 표현합니다. 
![컴퓨터가 알아 들을 수 있도록 변경](images/simple5by5_array.png)
우리가 이제 할 일은 이 array matrix의 값을 구해보는 것입니다.

## 



### Source
* [An introduction to Reinforcement Learning](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)
* [Reinforcement Learning with Q tables](https://itnext.io/reinforcement-learning-with-q-tables-5f11168862c8)
* [Introduction to Q-Learning](https://towardsdatascience.com/introduction-to-q-learning-88d1c4f2b49c)
 