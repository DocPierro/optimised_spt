from abc import ABC, abstractmethod
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import time
from queue import Queue

from dataclasses import dataclass, field
from typing import Optional, Callable, ClassVar

@dataclass(unsafe_hash=True)
class StateMC:
    id: int = field(init=False)
    name: str
    initial_prob: float
    initial_prob_function: Callable[[list[float]], float]
    is_absorbing: bool

    _counter: ClassVar[int] = 0
    def __post_init__(self):
        self.id = StateMC._counter
        StateMC._counter += 1

@dataclass(unsafe_hash=True)
class TransitionMC:
    source: StateMC
    target: StateMC
    label: Optional[str]
    weight: float
    weight_function: Callable[[list[float]], float]


class MarkovChain(ABC):

    def __init__(self, ev, pt, states, transitions):

        self.ev = ev
        self.ev_language = self.ev.get_language_pm4py()
        self.pt = pt
        self.states = states
        self.transitions = transitions

        self.initial_states = [state for state in self.states if state.initial_prob > 0]
        for state in self.states:
            if state.is_absorbing:
                self.absorbing_state = state
                break
        self.labels = list({transition.label for transition in self.transitions})
        self.S = [state for state in self.states if state != self.absorbing_state]
        self.P = {(t.source.id, t.target.id, t.label): t.weight for t in transitions}
        self.T = {(t.source.id, t.label): t.weight for t in transitions if t.target == self.absorbing_state}
        self.P_function = {(t.source.id, t.target.id, t.label): t.weight_function for t in transitions}
        self.T_function = {(t.source.id, t.label): t.weight_function for t in transitions if t.target == self.absorbing_state}

        self.outgoing = defaultdict(list)
        for transition in self.transitions:
            self.outgoing[transition.source].append(transition)

    def get_ev(self):
        return self.ev

    def get_states(self):
        return self.states

    def get_transitions(self):
        return self.transitions

    def get_initial_states(self):
        return self.initial_states

    def gt_absorbing_state(self):
        return self.absorbing_state

    def get_labels(self):
        return self.labels

    def get_S(self):
        return self.S

    def get_P(self, s, sprime, l):
        return self.P.get((s.id, sprime.id, l),0)

    def get_T(self, s, l):
        return self.T.get((s.id, l),0)

    def get_P_function(self, s, sprime, l):
        return self.P_function.get((s.id, sprime.id, l),lambda p: 0)

    def get_T_function(self, s, l):
        return self.T_function.get((s.id, l),lambda p: 0)

    def get_outgoing(self, state):
        return self.outgoing[state]

    @abstractmethod
    def unfold(self):
        pass

    def to_networkx(self):
        G = nx.MultiDiGraph()
        for state in self.states:
            G.add_node(state)
        for tr in self.transitions:
            G.add_edge(
                tr.source,
                tr.target,
                label=tr.label,
                weight=tr.weight
            )
        return G

    def draw(self):

        G = self.to_networkx()
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))

        node_colors = []
        for s in G.nodes():
            if G.out_degree(s) == 0:
                node_colors.append("#add8e6")
            else:
                node_colors.append("#dddddd")

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=1500,
            node_color=node_colors,
            edgecolors="black"
        )

        node_labels = {s: s.name for s in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11)

        for s in G.nodes():
            if s.initial_prob > 0:
                x, y = pos[s]
                plt.text(
                    x,
                    y + 0.15,
                    f"π = {s.initial_prob:.2f}",
                    fontsize=10,
                    ha="center",
                    color="red"
                )

        # --- ARÊTES ---
        for u, v, key, data in G.edges(keys=True, data=True):
            nx.draw_networkx_edges(
                G,
                pos,
                arrows=True,
                arrowstyle="-|>",  # flèche nette
                arrowsize=20,  # taille visible
                width=1.5,
                min_source_margin=20,
                min_target_margin=20
            )

            # Position du label
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            label = f"{data['weight']:.2f} , {data['label']}"

            plt.text(
                x,
                y,
                label,
                fontsize=9,
                ha="center",
                va="baseline",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5)
            )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return "States: " + str(self.states) + "\n" + "Transitions: " + str(self.transitions)


class DTMC(MarkovChain):

    def __init__(self, ev, pt, states, transitions):
        super().__init__(ev, pt, states, transitions)
        self.next_state_id = 0

    def compute_dtmc(self, p):
        for state in self.states:
            state.initial_prob = state.initial_prob_function(p)
        for transition in self.transitions:
            transition.weight = transition.weight_function(p)

    def unfold(self):

        #start_time = time.time()

        initial_states = [state for state in self.states if state.initial_prob > 0]
        outgoing = defaultdict(list)
        for transition in self.transitions:
            outgoing[transition.source].append(transition)

        max_length = max([len(trace.get_seq()) for trace in self.ev.get_language()])
        prefixes = self.ev.build_prefix_set()

        q = Queue(0)
        triples = {}
        for initial_state in initial_states:
            q.put((initial_state, [], 0))
            triples[(initial_state, tuple([]), 0)] = initial_state.initial_prob

        nb_transition = 0
        traces = {}
        while not q.empty():
            state, trace, level = q.get()
            prob = triples[(state, tuple(trace), level)]
            if len(outgoing[state]) > 0:
                for transition in outgoing[state]:
                    nb_transition += 1
                    new_state = transition.target
                    new_trace = trace.copy()
                    label = transition.label
                    if label is not None:
                        new_trace.append(label)
                    if len(new_trace) <= max_length:
                        if tuple(new_trace) in prefixes:
                            new_prob = prob * transition.weight
                            key = (new_state, tuple(new_trace), level + 1)
                            if key not in triples:
                                triples[key] = new_prob
                                q.put((new_state, new_trace, level + 1))
                            else:
                                triples[key] = triples[key] + new_prob
            else:
                tt = tuple(trace)
                if tt in self.ev_language:
                    if tt not in traces:
                        traces[tt] = prob
                    else:
                        traces[tt] = traces[tt] + prob

        #print("Unfolding")
        #print("\tnumber of nodes of the unfolding: " + str(len(triples)))
        #print("\tnumber of transitions of the unfolding: " + str(nb_transition))
        #print("\ttime of the unfolding: " + str(time.time() - start_time) + " seconds")
        return traces


class CTMC(MarkovChain):

    def __init__(self, ev, pt, states, transitions):
        super().__init__(ev, pt, states, transitions)