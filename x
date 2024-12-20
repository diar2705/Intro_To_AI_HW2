


function Minimax(State, Agent, Turn):
    if G(State) then return U(State, Agent)
    Children = Succ(State)
    if Turn == Agent then:
        curMax = -♾️
        for c in Children
            v = Minimax(c, Agent, (Turn + 1) % k)
            curMax = max(curMax, v)
        return curMax
    else
        curMax = -♾️
        for c in Children
            v = Minimax(c, Agent, (Turn + 1) % k)
            curMin = max(curMax, v)
        return curMax






function Minimax(State, Agent, Turn):
    if G(State) then return U(State, Agent)
    Children = Succ(State)
    if Turn == Agent then: 
        curMax = -♾️
        for c in Children
            v = Minimax(c, Agent, (Turn + 1) % k)
            curMax = max(curMax, v)
        return curMax
    else
        curMin = ♾️
        for c in Children
            v = Minimax(c, Agent, (Turn + 1) % k)
            curMin = min(curMin, v)
        return curMin





function Minimax(State, Agent, Turn):
    if G(State) then return U(State, Agent)
    Children = Succ(State)
    curMax = -♾️
    for c in Children
        v = Minimax(c, Agent, (Turn + 1) % k)
        curMax = max(curMax, v)
    return curMax