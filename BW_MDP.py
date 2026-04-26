import gym
from sympy import false

P={
    0:{#Hole
        0 : [(1.0,0,0.0,True)] #prob, next state , reward, boolean
       ,1 : [(1.0,0,0.0,True)]
    },
    1:{#Start if terminal true else false
        0: [(1.0, 0, 0.0, True)]  # prob, next state , reward, boolean
       ,1: [(1.0, 2, 1.0, True)]
    }
   ,2:{#goal
        0: [(1.0, 2, 0.0, True)]  # prob, next state , reward, boolean
       ,1: [(1.0, 2, 0.0, True)]
    }
  }
