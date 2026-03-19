import numpy as np
import pm4py
import math
import itertools
from functools import reduce
from itertools import product

from model.MarkovChain import StateMC, TransitionMC, MarkovChain, DTMC, CTMC


class Node:

    def __init__(self, value, children=None):
        if children is None:
            children = []
        self.value = value
        self.children = children

    def is_leaf(self):
        return len(self.children) == 0

    def to_dtmc(self, ev, pt) -> DTMC:
        if self.value == "":
            return DTMC(ev,pt,[StateMC("se",1,lambda p: 1,True)],[])
        else:
            s0 = StateMC("s0_"+self.value,1,lambda p: 1,False)
            se = StateMC("se",0,lambda p: 0,True)
            return DTMC(ev, pt, [s0,se], [TransitionMC(s0,se,self.value,1,lambda p: 1)])

    def to_string(self, level):
        if self.value == "":
            ret = "  " * level + "tau" + "\n"
        else:
            ret = "  " * level + str(self.value) + "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __str__(self):
        if self.value == "":
            ret = "  " * level + "tau" + "\n"
        else:
            ret = "  " * level + str(self.value) + "\n"

class Seq(Node):

    def __init__(self, node_id, children):
        super().__init__("Seq " + str(node_id), children=children)
        permutations = itertools.permutations(range(len(children)))
        self.O = {"".join(map(str, perm)): 0 for perm in permutations}
        self.O["".join(str(i) for i in range(len(children)))] = 1

    def assign_prob(self, probs):
        pass

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev,pt) for child in self.children]

        def merge(left,right):

            states = []
            state_mapping_left = {}
            state_mapping_right = {}
            for state in left.get_S():
                new_state = StateMC(name=state.name,
                                    initial_prob=state.initial_prob,
                                    initial_prob_function=state.initial_prob_function,
                                    is_absorbing=False)
                states.append(new_state)
                state_mapping_left[state] = new_state
            for state in right.get_S():
                new_state = StateMC(name=state.name,
                                    initial_prob=left.absorbing_state.initial_prob*state.initial_prob,
                                    initial_prob_function=lambda p, left=left, state=state: left.absorbing_state.initial_prob_function(p) * state.initial_prob_function(p),
                                    is_absorbing=False)
                states.append(new_state)
                state_mapping_right[state] = new_state
            se = StateMC(name="se",
                         initial_prob=left.absorbing_state.initial_prob*right.absorbing_state.initial_prob,
                         initial_prob_function=lambda p, left=left, right=right: left.absorbing_state.initial_prob_function(p) * right.absorbing_state.initial_prob_function(p),
                         is_absorbing=True)
            states.append(se)

            transitions = []
            for transition in left.get_transitions():
                new_source = state_mapping_left[transition.source]
                if transition.target != left.absorbing_state:
                    new_target = state_mapping_left[transition.target]
                    transitions.append(TransitionMC(
                        source=new_source,
                        target=new_target,
                        label=transition.label,
                        weight=transition.weight,
                        weight_function=transition.weight_function))
            for transition in right.get_transitions():
                new_source = state_mapping_right[transition.source]
                if transition.target != right.absorbing_state:
                    new_target = state_mapping_right[transition.target]
                    transitions.append(TransitionMC(
                        source=new_source,
                        target=new_target,
                        label=transition.label,
                        weight=transition.weight,
                        weight_function=transition.weight_function))

            for s in left.get_S():
                for sprime in right.get_S():
                    for l in left.get_labels():
                        prob = left.get_T(s,l) * sprime.initial_prob
                        if prob > 0:
                            new_source = state_mapping_left[s]
                            new_target = state_mapping_right[sprime]
                            transitions.append(TransitionMC(
                                source=new_source,
                                target=new_target,
                                label=l,
                                weight=prob,
                                weight_function=lambda p, s=s, l=l, left=left, sprime=sprime: left.get_T_function(s,l)(p) * sprime.initial_prob_function(p)))

            for s in left.get_S():
                for l in left.get_labels():
                    prob = left.get_T(s,l) * right.absorbing_state.initial_prob
                    if prob > 0:
                        new_source = state_mapping_left[s]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=se,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, l=l, left=left, right=right: left.get_T_function(s,l)(p) * right.absorbing_state.initial_prob_function(p)))
            for s in right.get_S():
                for l in right.get_labels():
                    prob = right.get_T(s,l)
                    if prob > 0:
                        new_source = state_mapping_right[s]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=se,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, l=l, right=right: right.get_T_function(s,l)(p)))

            return DTMC(ev, pt, states, transitions)

        return reduce(merge, childs_dtmc)

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.O) + "\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def __str__(self):
        ret = "  " + str(elf.value) + " -> " + str(self.O) + "\n"

class BinaryChoice(Node):

    def __init__(self, node_id, children, wi):
        super().__init__("Binary Choice " + str(node_id), children=children)
        self.node_id = node_id
        self.wi = wi
        self.p = [0.5, 0.5]

    def assign_prob(self, probs):
        self.p = [probs[self.wi], 1-probs[self.wi]]

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev, pt) for child in self.children]

        # S + Pi
        states = []
        state_mapping = {}
        se_prob, se_terms = 0, []
        for i, child_dtmc in enumerate(childs_dtmc):
            for state in child_dtmc.states:
                if state != child_dtmc.absorbing_state:
                    new_state = StateMC(name=state.name,
                                        initial_prob=self.p[i] * state.initial_prob,
                                        initial_prob_function=lambda p, i=i, state=state: (p[self.wi] if i == 0 else 1 - p[self.wi]) * state.initial_prob_function(p),
                                        is_absorbing=False)
                    states.append(new_state)
                    state_mapping[state] = new_state
                else:
                    se_prob += self.p[i] * state.initial_prob
                    se_terms.append(lambda p, i=i, state=state: (p[self.wi] if i==0 else 1-p[self.wi]) * state.initial_prob_function(p))
        se = StateMC(name="se",
                     initial_prob=se_prob,
                     initial_prob_function=lambda p, terms=se_terms: sum(term(p) for term in terms),
                     is_absorbing=True)
        states.append(se)

        # P + L
        transitions = []
        for i, child_dtmc in enumerate(childs_dtmc):
            for transition in child_dtmc.transitions:
                new_source = state_mapping[transition.source]
                if transition.target == child_dtmc.absorbing_state:
                    new_target = se
                else:
                    new_target = state_mapping[transition.target]
                transitions.append(TransitionMC(
                    source=new_source,
                    target=new_target,
                    label=transition.label,
                    weight=transition.weight,
                    weight_function=transition.weight_function))

        return DTMC(ev, pt, states, transitions)

    def __str__(self):
        ret = "  " + str(self.value) + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"

class Choice(Node):

    def __init__(self, node_id, children, wi_start, wi_end):
        super().__init__("Choice " + str(node_id), children=children)
        self.wi_start, self.wi_end = wi_start, wi_end
        self.p = [1/len(children) for _ in range(len(children))]

    def assign_prob(self, probs):
        self.p = [x/sum(probs[self.wi_start:self.wi_end+1]) for x in probs[self.wi_start:self.wi_end+1]]

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev,pt) for child in self.children]

        # S + Pi
        states = []
        state_mapping = {}
        se_prob, se_terms = 0, []
        for i, child_dtmc in enumerate(childs_dtmc):
            for state in child_dtmc.states:
                if state != child_dtmc.absorbing_state:
                    new_state = StateMC(name=state.name,
                                        initial_prob=self.p[i] * state.initial_prob,
                                        initial_prob_function=lambda p, i=i, state=state: p[self.wi_start+i] * state.initial_prob_function(p),
                                        is_absorbing=False)
                    states.append(new_state)
                    state_mapping[state] = new_state
                else:
                    se_prob += self.p[i] * state.initial_prob
                    se_terms.append(lambda p, i=i, state=state: p[self.wi_start+i] * state.initial_prob_function(p))
        se = StateMC(name="se",
                     initial_prob=se_prob,
                     initial_prob_function=lambda p, terms=se_terms: sum(term(p) for term in terms),
                     is_absorbing=True)
        states.append(se)

        # P + L
        transitions = []
        for i, child_dtmc in enumerate(childs_dtmc):
            for transition in child_dtmc.transitions:
                new_source = state_mapping[transition.source]
                if transition.target == child_dtmc.absorbing_state:
                    new_target = se
                else:
                    new_target = state_mapping[transition.target]
                transitions.append(TransitionMC(
                    source=new_source,
                    target=new_target,
                    label=transition.label,
                    weight=transition.weight,
                    weight_function=transition.weight_function))

        return DTMC(ev, pt, states, transitions)

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.p) + " | (" + str(self.wi_start) + "," + str(self.wi_end) + ")\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def __str__(self):
        ret = "  " + str(self.value) + " -> " + str(self.p) + " | (" + str(self.wi_start) + "," + str(self.wi_end) + ")\n"

class BinaryParallel(Node):

    def __init__(self, node_id, children, wi):
        super().__init__("Binary Parallel " + str(node_id), children=children)
        self.wi = wi
        self.p = [0.5, 0.5]

    def assign_prob(self, probs):
        self.p = [probs[self.wi], 1-probs[self.wi]]

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev,pt) for child in self.children]

        state_spaces, end_states = [], []
        for i, child_dtmc in enumerate(childs_dtmc):
            local_states = []
            for state in child_dtmc.get_S():
                local_states.append(state)
            end_state = StateMC(name=f"s{i}_end",
                                initial_prob=0,
                                initial_prob_function=lambda p: 0,
                                is_absorbing=False)
            local_states.append(end_state)
            end_states.append(end_state)
            state_spaces.append(local_states)

        states = []
        state_origin = {}
        L = {}
        for combo in product(*state_spaces):
            if combo == tuple(end_states):
                continue
            funcs = [childs_dtmc[i].absorbing_state.initial_prob_function if s == end_states[i] else s.initial_prob_function for i, s in enumerate(combo)]
            new_state = StateMC(name="__".join(s.name for s in combo),
                                initial_prob=math.prod(childs_dtmc[i].absorbing_state.initial_prob if s == end_states[i] else s.initial_prob for i, s in enumerate(combo)),
                                initial_prob_function=lambda p, fs=funcs: math.prod(f(p) for f in fs),
                                is_absorbing=False)
            states.append(new_state)
            state_origin[new_state] = combo
            L[new_state] = [i for i, s in enumerate(combo) if s != end_states[i]]
        funcs_se = [child_dtmc.absorbing_state.initial_prob_function for child_dtmc in childs_dtmc]
        se = StateMC(name="se",
                     initial_prob=math.prod(child_dtmc.absorbing_state.initial_prob for child_dtmc in childs_dtmc),
                     initial_prob_function=lambda p, fs=funcs_se: math.prod(f(p) for f in fs),
                     is_absorbing=True)

        transitions = []
        for s in states:
            for sprime in states:
                    for l in [l for child_dtmc in childs_dtmc for l in child_dtmc.get_labels()]:

                        if s == sprime:
                            prob, prob_terms = 0, []
                            for i in L[s]:
                                prob += self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_P(state_origin[s][i],state_origin[s][i],l)
                                f = childs_dtmc[i].get_P_function(state_origin[s][i], state_origin[s][i], l)
                                Ls = tuple(L[s])
                                prob_terms.append(lambda p, f=f, i=i, Ls=Ls, wi=self.wi:((p[wi] if i == 0 else 1 - p[wi]) / sum((p[wi] if j == 0 else 1 - p[wi]) for j in Ls)) * f(p))
                                if prob > 0:
                                    transitions.append(TransitionMC(
                                        source=s,
                                        target=s,
                                        label=l,
                                        weight=prob,
                                        weight_function=lambda p, terms=prob_terms: sum(term(p) for term in terms)))

                        for i in range(len(childs_dtmc)):

                            if (state_origin[s][i] != state_origin[sprime][i]
                                    and i in L[s] and i in L[sprime]
                                    and all(state_origin[s][j] == state_origin[sprime][j] for j in range(len(childs_dtmc)) if j != i)):
                                prob = self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_P(state_origin[s][i],state_origin[sprime][i],l)
                                if prob > 0:
                                    transitions.append(TransitionMC(
                                        source=s,
                                        target=sprime,
                                        label=l,
                                        weight=prob,
                                        weight_function=lambda p, i=i, s=s, sprime=sprime, l=l, state_origin=state_origin, Ls=L[s], f=childs_dtmc[i].get_P_function(state_origin[s][i],state_origin[sprime][i],l): ((p[self.wi] if i == 0 else 1 - p[self.wi]) / sum((p[self.wi] if j == 0 else 1 - p[self.wi]) for j in Ls)) * f(p)))

                            if (i in L[s] and i not in L[sprime]
                                    and any(j != i and j in L[s] for j in range(len(childs_dtmc)))
                                    and all(state_origin[s][j] == state_origin[sprime][j] for j in range(len(childs_dtmc)) if j != i)):
                                prob = self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_T(state_origin[s][i],l)
                                if prob > 0:
                                    transitions.append(TransitionMC(
                                        source=s,
                                        target=sprime,
                                        label=l,
                                        weight=prob,
                                        weight_function=lambda p, i=i, s=s, l=l, state_origin=state_origin, Ls=L[s], f=childs_dtmc[i].get_T_function(state_origin[s][i],l): ((p[self.wi] if i == 0 else 1 - p[self.wi]) / sum((p[self.wi] if j == 0 else 1 - p[self.wi]) for j in Ls)) * f(p)))

        for s in states:
            for l in [l for child_dtmc in childs_dtmc for l in child_dtmc.get_labels()]:
                if len(L[s]) == 1:
                    prob = childs_dtmc[L[s][0]].get_T(state_origin[s][L[s][0]],l)
                    i = L[s][0]
                    f = childs_dtmc[i].get_T_function(state_origin[s][i], l)
                    if prob > 0:
                        transitions.append(TransitionMC(
                            source=s,
                            target=se,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, f=f: f(p)))

        states.append(se)

        return DTMC(ev, pt, states, transitions)

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def __str__(self):
        ret = "  " + str(self.value) + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"

class Parallel(Node):

    def __init__(self, node_id, children, wi_start, wi_end):
        super().__init__("Parallel " + str(node_id), children=children)
        self.wi_start, self.wi_end = wi_start, wi_end
        self.p = [1/len(children) for _ in range(len(children))]

    def assign_prob(self, probs):
        self.p = [x/sum(probs[self.wi_start:self.wi_end+1]) for x in probs[self.wi_start:self.wi_end+1]]

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev, pt) for child in self.children]

        state_spaces, end_states = [], []
        for i, child_dtmc in enumerate(childs_dtmc):
            local_states = []
            for state in child_dtmc.get_S():
                local_states.append(state)
            end_state = StateMC(name=f"s{i}_end",
                                initial_prob=0,
                                initial_prob_function=lambda p: 0,
                                is_absorbing=False)
            local_states.append(end_state)
            end_states.append(end_state)
            state_spaces.append(local_states)

        states = []
        state_origin = {}
        L = {}
        for combo in product(*state_spaces):
            if combo == tuple(end_states):
                continue
            funcs = [childs_dtmc[i].absorbing_state.initial_prob_function if s == end_states[i] else s.initial_prob_function for i, s in enumerate(combo)]
            new_state = StateMC(name="__".join(s.name for s in combo),
                                initial_prob=math.prod(childs_dtmc[i].absorbing_state.initial_prob if s == end_states[i] else s.initial_prob for i, s in enumerate(combo)),
                                initial_prob_function=lambda p, fs=funcs: math.prod(f(p) for f in fs),
                                is_absorbing=False)
            states.append(new_state)
            state_origin[new_state] = combo
            L[new_state] = [i for i, s in enumerate(combo) if s != end_states[i]]
        funcs_se = [child_dtmc.absorbing_state.initial_prob_function for child_dtmc in childs_dtmc]
        se = StateMC(name="se",
                     initial_prob=math.prod(child_dtmc.absorbing_state.initial_prob for child_dtmc in childs_dtmc),
                     initial_prob_function=lambda p, fs=funcs_se: math.prod(f(p) for f in fs),
                     is_absorbing=True)

        transitions = []
        for s in states:
            for sprime in states:
                for l in [l for child_dtmc in childs_dtmc for l in child_dtmc.get_labels()]:

                    if s == sprime:
                        prob, prob_terms = 0, []
                        for i in L[s]:
                            prob += self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_P(state_origin[s][i],state_origin[s][i],l)
                            prob_terms.append(lambda p, i=i, s=s, l=l, state_origin=state_origin, Ls=L[s]: ((p[self.wi_start+i]/sum(p[self.wi_start+j] for j in Ls)) * childs_dtmc[i].get_P_function(state_origin[s][i],state_origin[s][i],l)(p)))
                        if prob > 0:
                            transitions.append(TransitionMC(
                                source=s,
                                target=s,
                                label=l,
                                weight=prob,
                                weight_function=lambda p, terms=prob_terms: sum(term(p) for term in terms)))

                    for i in range(len(childs_dtmc)):

                        if (state_origin[s][i] != state_origin[sprime][i]
                                and i in L[s] and i in L[sprime]
                                and all(state_origin[s][j] == state_origin[sprime][j] for j in range(len(childs_dtmc)) if j != i)):
                            prob = self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_P(state_origin[s][i],state_origin[sprime][i],l)
                            if prob > 0:
                                transitions.append(TransitionMC(
                                    source=s,
                                    target=sprime,
                                    label=l,
                                    weight=prob,
                                    weight_function=lambda p, i=i, s=s, sprime=sprime, l=l, state_origin=state_origin, Ls=L[s], f=childs_dtmc[i].get_P_function(state_origin[s][i],state_origin[sprime][i],l): (p[self.wi_start+i]/sum((p[self.wi_start+j]) for j in Ls)) * f(p)))

                        if (i in L[s] and i not in L[sprime]
                                and any(j != i and j in L[s] for j in range(len(childs_dtmc)))
                                and all(state_origin[s][j] == state_origin[sprime][j] for j in range(len(childs_dtmc)) if j != i)):
                            prob = self.p[i] / sum([self.p[j] for j in L[s]]) * childs_dtmc[i].get_T(state_origin[s][i],l)
                            if prob > 0:
                                transitions.append(TransitionMC(
                                    source=s,
                                    target=sprime,
                                    label=l,
                                    weight=prob,
                                    weight_function=lambda p, i=i, s=s, l=l, state_origin=state_origin, Ls=L[s], f=childs_dtmc[i].get_T_function(state_origin[s][i], l): p[self.wi_start+i]/sum((p[self.wi_start+j] for j in Ls)) * f(p)))

        for s in states:
            for l in [l for child_dtmc in childs_dtmc for l in child_dtmc.get_labels()]:
                if len(L[s]) == 1:
                    prob = childs_dtmc[L[s][0]].get_T(state_origin[s][L[s][0]], l)
                    if prob > 0:
                        transitions.append(TransitionMC(
                            source=s,
                            target=se,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, l=l, state_origin=state_origin, Ls=L[s]: childs_dtmc[Ls[0]].get_T_function(state_origin[s][Ls[0]], l)(p)))

        states.append(se)

        return DTMC(ev, pt, states, transitions)

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.p) + " | (" + str(self.wi_start) + "," + str(self.wi_end) + ")\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def __str__(self):
        ret = "  " + str(self.value) + " -> " + str(self.p) + " | (" + str(self.wi_start) + "," + str(self.wi_end) + ")\n"

class Loop(Node):

    def __init__(self, node_id, children, wi):
        super().__init__("Loop " + str(node_id), children=children)
        self.wi = wi
        self.p = 0.5

    def assign_prob(self, probs):
        self.p = probs[self.wi]

    def to_dtmc(self, ev, pt) -> DTMC:

        childs_dtmc = [child.to_dtmc(ev, pt) for child in self.children]

        # S + Pi
        states = []
        state_mapping_1 = {}
        state_mapping_2 = {}

        alpha = childs_dtmc[0].absorbing_state.initial_prob * self.p * childs_dtmc[1].absorbing_state.initial_prob
        alpha_star = 1 / (1 - alpha)

        alpha_function = lambda p, f1=childs_dtmc[0].absorbing_state.initial_prob_function, f2=childs_dtmc[1].absorbing_state.initial_prob_function: f1(p) * p[self.wi] * f2(p)
        alpha_star_function = lambda p, af=alpha_function: 1 / (1 - af(p))

        for state in childs_dtmc[0].get_S():
            new_state = StateMC(name=state.name,
                                initial_prob=alpha_star * state.initial_prob,
                                initial_prob_function=lambda p, state=state, asf=alpha_star_function: asf(p) * state.initial_prob_function(p),
                                is_absorbing=False)
            states.append(new_state)
            state_mapping_1[state] = new_state

        for state in childs_dtmc[1].get_S():
            new_state = StateMC(name=state.name,
                                initial_prob=childs_dtmc[0].absorbing_state.initial_prob * self.p * alpha_star * state.initial_prob,
                                initial_prob_function=lambda p, state=state, asf=alpha_star_function, pi_epsilon_1=childs_dtmc[0].absorbing_state.initial_prob_function: pi_epsilon_1(p) * p[self.wi] * asf(p) * state.initial_prob_function(p),
                                is_absorbing=False)
            states.append(new_state)
            state_mapping_2[state] = new_state

        se = StateMC(name="se",
                     initial_prob=alpha_star * childs_dtmc[0].absorbing_state.initial_prob * (1-self.p),
                     initial_prob_function=lambda p, asf=alpha_star_function, pi_epsilon_1=childs_dtmc[0].absorbing_state.initial_prob_function: asf(p) * pi_epsilon_1(p) * (1-p[self.wi]),
                     is_absorbing=True)
        states.append(se)

        transitions = []
        for s in childs_dtmc[0].get_S():
            for sprime in childs_dtmc[0].get_S():
                for l in childs_dtmc[0].get_labels():
                    prob = (childs_dtmc[0].get_P(s,sprime,l)
                            + childs_dtmc[0].get_T(s,l) * self.p
                            * childs_dtmc[1].absorbing_state.initial_prob
                            * alpha_star * sprime.initial_prob)
                    if prob > 0:
                        new_source = state_mapping_1[s]
                        new_target = state_mapping_1[sprime]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=new_target,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, sprime=sprime, l=l, asf=alpha_star_function:
                                                (childs_dtmc[0].get_P_function(s,sprime,l)(p)
                                                       + childs_dtmc[0].get_T_function(s,l)(p) * p[self.wi]
                                                       * childs_dtmc[1].absorbing_state.initial_prob_function(p)
                                                       * asf(p) * sprime.initial_prob_function(p))))
        for s in childs_dtmc[1].get_S():
            for sprime in childs_dtmc[1].get_S():
                for l in childs_dtmc[1].get_labels():
                    prob = (childs_dtmc[1].get_P(s,sprime,l)
                            + childs_dtmc[1].get_T(s,l) * self.p
                            * childs_dtmc[0].absorbing_state.initial_prob
                            * alpha_star * sprime.initial_prob)
                    if prob > 0:
                        new_source = state_mapping_2[s]
                        new_target = state_mapping_2[sprime]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=new_target,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, sprime=sprime, l=l, asf=alpha_star_function:
                                                (childs_dtmc[1].get_P_function(s,sprime,l)(p)
                                                        + childs_dtmc[1].get_T_function(s,l)(p) * p[self.wi]
                                                        * childs_dtmc[0].absorbing_state.initial_prob_function(p)
                                                        * asf(p) * sprime.initial_prob_function(p))))
        for s in childs_dtmc[1].get_S():
            for sprime in childs_dtmc[0].get_S():
                for l in childs_dtmc[1].get_labels():
                    prob = childs_dtmc[1].get_T(s,l) * alpha_star * sprime.initial_prob
                    if prob > 0:
                        new_source = state_mapping_2[s]
                        new_target = state_mapping_1[sprime]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=new_target,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, sprime=sprime, l=l, asf=alpha_star_function: childs_dtmc[1].get_T_function(s,l)(p) * asf(p) * sprime.initial_prob_function(p)))
        for s in childs_dtmc[0].get_S():
            for sprime in childs_dtmc[1].get_S():
                for l in childs_dtmc[0].get_labels():
                    prob = childs_dtmc[0].get_T(s,l) * alpha_star * self.p * sprime.initial_prob
                    if prob > 0:
                        new_source = state_mapping_1[s]
                        new_target = state_mapping_2[sprime]
                        transitions.append(TransitionMC(
                            source=new_source,
                            target=new_target,
                            label=l,
                            weight=prob,
                            weight_function=lambda p, s=s, sprime=sprime, l=l, asf=alpha_star_function: childs_dtmc[0].get_T_function(s,l)(p) * asf(p) * p[self.wi] * sprime.initial_prob_function(p)))

        for s in childs_dtmc[0].get_S():
            for l in childs_dtmc[0].get_labels():
                prob = childs_dtmc[0].get_T(s,l) * alpha_star * (1-self.p)
                if prob > 0:
                    new_source = state_mapping_1[s]
                    transitions.append(TransitionMC(
                        source=new_source,
                        target=se,
                        label=l,
                        weight=prob,
                        weight_function=lambda p, s=s, sprime=sprime, l=l, asf=alpha_star_function: childs_dtmc[0].get_T_function(s,l)(p) * asf(p) * (1-p[self.wi])))
        for s in childs_dtmc[1].get_S():
            for l in childs_dtmc[1].get_labels():
                prob = childs_dtmc[1].get_T(s,l) * alpha_star * childs_dtmc[0].absorbing_state.initial_prob * (1-self.p)
                if prob > 0:
                    new_source = state_mapping_2[s]
                    transitions.append(TransitionMC(
                        source=new_source,
                        target=se,
                        label=l,
                        weight=prob,
                        weight_function=lambda p, s=s, l=l, asf=alpha_star_function: childs_dtmc[1].get_T_function(s,l)(p) * asf(p) * childs_dtmc[0].absorbing_state.initial_prob_function(p) * (1-p[self[wi]])))

        return DTMC(ev, pt, states, transitions)

    def to_string(self, level):
        ret = "  " * level + self.value + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"
        for child in self.children:
            ret += child.to_string(level+1)
        return ret

    def __str__(self):
        ret = "  " + str(self.value) + " -> " + str(self.p) + " | (" + str(self.wi) + ")\n"

class StochasticProcessTree:

    def __init__(self, ev, tree, root, nodes, size_p, noise_threshold):
        self.ev = ev
        self.tree = tree
        self.root, self.nodes, self.size_p = root, nodes, size_p
        self.noise_threshold = noise_threshold

    def assign_prob(self, probs):
        for node in self.nodes:
            node.assign_prob(probs)

    def get_prob(self):
        probs = [0 for _ in range(self.size_p)]
        for node in self.nodes:
            if type(node) is BinaryChoice or type(node) is BinaryParallel:
                probs[node.wi] = node.p[0]
            if type(node) is Loop:
                probs[node.wi] = node.p
            elif type(node) is Choice or type(node) is Parallel:
                probs[node.wi_start:node.wi_end] = node.p
        return probs

    def normalize_prob(self, p):
        p = list(p)
        for node in self.nodes:
            if isinstance(node, (Choice, Parallel)):
                sub = p[node.wi_start:node.wi_end+1]
                s = sum(sub)
                sub = [x / s for x in sub]
                p[node.wi_start:node.wi_end+1] = sub
        return p

    def to_dtmc(self):
        return self.root.to_dtmc(self.ev, self)

    def __str__(self):
        if self.tree is not None:
            pm4py.view_process_tree(self.tree)
        return self.root.to_string(0)
