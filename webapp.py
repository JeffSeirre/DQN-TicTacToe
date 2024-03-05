import streamlit as st
import numpy as np
from webapp_classes import Game_agent

rows, cols = 3, 3

def choose_random():
    emptyFields = np.where(st.session_state.board == 0)
    emptyCoords = list(zip(emptyFields[0], emptyFields[1]))
    randomField = emptyCoords[np.random.choice(len(emptyCoords))]
    return randomField[0], randomField[1]

def opponent_play():
    x,y = st.session_state.agent.play(st.session_state.board.flatten())
    if (st.session_state.board[x, y] == 0):
        return x,y
    else:
        return choose_random()
    
def clicked(x=0, y=0):
    if (st.session_state.board[x, y] != 0) or (st.session_state.terminated):
        return
    st.session_state.board[x, y] = -1
    st.session_state.grid[x, y] = "X"

    if terminated():
        return

    play_x, play_y = opponent_play()
    print(f"Play in {play_x}:{play_y}")
    st.session_state.board[play_x, play_y] = 1
    st.session_state.grid[play_x, play_y] = "O"

    print(st.session_state.board)

    terminated()
    return

def terminated():
    match eval_board():
        case 'win_-1':
            st.session_state.message = 'You won! Click below to restart'
            st.session_state.terminated =  True
        case 'win_1':
            st.session_state.message = 'You lost! Click below to restart'
            st.session_state.terminated =  True
        case 'draw':
            st.session_state.message = 'You made a draw! Click below to restart'
            st.session_state.terminated =  True
        case 'undecided':
            st.session_state.terminated =  False
    return st.session_state.terminated

def reset():
    st.session_state.grid = np.full((rows, cols), "__")
    st.session_state.board = np.zeros((rows, cols))
    st.session_state.message = "Click the empty fields below to place your symbol X."
    st.session_state.terminated = False


def eval_board():
    rows = np.array(np.vsplit(np.transpose(st.session_state.board), 3))[:,0,:]
    columns = np.array(np.vsplit(st.session_state.board, 3))[:,0,:]
    diagonals = np.array([[st.session_state.board[0,0],st.session_state.board[1,1],st.session_state.board[2,2]], 
                          [st.session_state.board[0,2],st.session_state.board[1,1],st.session_state.board[2,0]]])
    lines = np.vstack([rows,columns,diagonals])
    for line in lines:
        if np.all(line == line[0]) and line[0] != 0:
            if line[0] == 1:
                return 'win_1'
            else:
                return 'win_-1'
    if not np.any(st.session_state.board == 0):
        return 'draw'
    return 'undecided'


# Initialize board state, neural network, symbol grid, and displayed message if they don't exist
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((rows, cols))

if 'agent' not in st.session_state:
    st.session_state.agent = Game_agent(9,9,1, "trained_value_network.pth")

if 'grid' not in st.session_state:
    st.session_state.grid = np.full((rows, cols), "__")

if 'message' not in st.session_state:
    st.session_state.message = "Click the empty fields below to place your symbol X."

if 'terminated' not in st.session_state:
    st.session_state.terminated = False




# Web UI        
st.title("Play TicTacToe against an AI agent!")
st.write(st.session_state.message)

# Display grid
for i in range(rows):
    columns = st.columns(cols)
    for j in range(len(columns)):
        with columns[j]:
            # Use button click to trigger cell update
            st.button(st.session_state.grid[i][j], key=f"btn_{i}_{j}", on_click=clicked, args=(i,j))

#st.write(st.session_state.board)
st.button("Click here to reset.", key='bt_reset', on_click=reset)

