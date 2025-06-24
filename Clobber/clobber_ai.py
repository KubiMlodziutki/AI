import argparse
import math
import random
import sys
import time
from typing import List, Tuple, Callable

Move = Tuple[int, int, int, int]


class GameState:
    def __init__(self, board: List[List[str]], player: str):
        self.board = board
        self.player = player
        self.rows = len(board)
        self.cols = len(board[0])

    def clone(self):
        return GameState([row[:] for row in self.board], self.player)

    def opponent(self) -> str:
        return 'W' if self.player == 'B' else 'B'

    def generate_moves(self) -> List[Move]:
        moves = []
        opp = self.opponent()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == self.player:
                    if r > 0 and self.board[r - 1][c] == opp:
                        moves.append((r, c, r - 1, c))
                    if r < self.rows - 1 and self.board[r + 1][c] == opp:
                        moves.append((r, c, r + 1, c))
                    if c > 0 and self.board[r][c - 1] == opp:
                        moves.append((r, c, r, c - 1))
                    if c < self.cols - 1 and self.board[r][c + 1] == opp:
                        moves.append((r, c, r, c + 1))
        return moves

    def apply_move(self, move: Move):
        r1, c1, r2, c2 = move
        self.board[r1][c1] = '_'
        self.board[r2][c2] = self.player
        self.player = self.opponent()

    def is_terminal(self) -> bool:
        return len(self.generate_moves()) == 0

    def pieces(self):
        b = w = 0
        for row in self.board:
            for cell in row:
                if cell == 'B':
                    b += 1
                elif cell == 'W':
                    w += 1
        return b, w


def h_mobility(state, player):
    p_moves = len(state.generate_moves())
    cur = state.player
    state.player = state.opponent()
    o_moves = len(state.generate_moves())
    state.player = cur
    return p_moves - o_moves if player == cur else o_moves - p_moves


def h_piece_count(state, player):
    b, w = state.pieces()
    return b - w if player == 'B' else w - b


def h_hybrid(state, player):
    return h_piece_count(state, player) * 10 + h_mobility(state, player)


HEURISTICS: List[Callable[[GameState, str], int]] = [
    h_mobility,
    h_piece_count,
    h_hybrid
]

visited_nodes = 0


def minimax(state, depth, alpha, beta, maximizing, eval_fn, root_player):
    global visited_nodes
    visited_nodes += 1
    if depth == 0 or state.is_terminal():
        return eval_fn(state, root_player), (-1, -1, -1, -1)
    best_move = (-1, -1, -1, -1)
    if maximizing:
        value = -math.inf
        for mv in state.generate_moves():
            child = state.clone()
            child.apply_move(mv)
            v, _ = minimax(child, depth - 1, alpha, beta, False, eval_fn, root_player)
            if v > value:
                value = v
                best_move = mv
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move
    else:
        value = math.inf
        for mv in state.generate_moves():
            child = state.clone()
            child.apply_move(mv)
            v, _ = minimax(child, depth - 1, alpha, beta, True, eval_fn, root_player)
            if v < value:
                value = v
                best_move = mv
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move


def read_board(stream):
    return [line.strip().split() for line in stream if line.strip()]


def fmt(board):
    return '\n'.join(' '.join(r) for r in board)


def play_full(state, depth, h_b, h_w, adaptive):
    global visited_nodes
    visited_nodes = 0
    rounds = 0
    last = None
    start = time.perf_counter()
    while not state.is_terminal():
        cur = state.player
        h = h_b if cur == 'B' else h_w
        if adaptive and random.random() < 0.3:
            h = random.randrange(len(HEURISTICS))
        _, mv = minimax(state, depth, -math.inf, math.inf, True, HEURISTICS[h], cur)
        if mv == (-1, -1, -1, -1):
            break
        state.apply_move(mv)
        rounds += 1
        last = cur
    elapsed = time.perf_counter() - start
    sys.stderr.write(f"{visited_nodes} {elapsed:.6f}\n")
    print(fmt(state.board))
    print(f"{rounds} {last}")


def agent_move(state, depth, h):
    global visited_nodes
    visited_nodes = 0
    _, mv = minimax(state, depth, -math.inf, math.inf, True, HEURISTICS[h], state.player)
    if mv == (-1, -1, -1, -1):
        print("-1 -1 -1 -1")
    else:
        print(f"{mv[0]} {mv[1]} {mv[2]} {mv[3]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--mode", choices=["full", "agent"], default="full")
    p.add_argument("--heuristic_b", type=int, default=0)
    p.add_argument("--heuristic_w", type=int, default=0)
    p.add_argument("--player", choices=["B", "W"], default="W")
    p.add_argument("--adaptive", action="store_true")
    args = p.parse_args()
    board = read_board(sys.stdin)
    state = GameState(board, args.player)
    if args.mode == "full":
        play_full(state, args.depth, args.heuristic_b, args.heuristic_w, args.adaptive)
    else:
        agent_move(state, args.depth, args.heuristic_b if args.player == 'B' else args.heuristic_w)


if __name__ == "__main__":
    main()
