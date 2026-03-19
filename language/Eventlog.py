import time
import uuid
import pm4py
import xml.etree.ElementTree as ET

import sympy
from pm4py.statistics.variants.log import get as variants_module

from model.ProcessTree import StochasticProcessTree, Node, Seq, Loop, Parallel, BinaryParallel, Choice, BinaryChoice

from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.petri_utils import remove_transition, add_arc_from_to, remove_place
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.obj import Operator

class Event:

    def __init__(self, case, activity, timestamp):
        self.case = case
        self.activity = activity
        self.timestamp = timestamp

    def get_case(self):
        return self.case

    def get_activity(self):
        return self.activity

    def get_timestamp(self):
        return self.timestamp

    def __repr__(self):
        return ("Case: " + str(self.case) +
                "; Activity: " + str(self.activity) +
                "; Timestamp: " + str(self.timestamp) + ";")


class Trace:

    def __init__(self, id, seq, word, freq, prob, mapping1, mapping3):
        self.id = id
        self.seq = seq
        self.word = word
        self.freq = freq
        self.prob = prob
        self.mapping1 = mapping1
        self.mapping3 = mapping3

    def get_id(self):
        return self.id

    def get_word(self):
        return self.word

    def get_seq(self):
        return self.seq

    def get_freq(self):
        return self.freq

    def get_prob(self):
        return self.prob

    def get_mapping1(self):
        return self.mapping1

    def get_mapping3(self):
        return self.mapping3

    def map1(self, mapping1):
        self.mapping1 = mapping1

    def map3(self, mapping3):
        self.mapping3 = mapping3

    def __str__(self):
        return "(" + str(self.id) + ",<" + str(self.word) + ">," + str(self.freq) + "," + str(self.prob) + "," + str(self.mapping1) + "," + str(self.mapping3) + ")"

    def __repr__(self):
        return self.__str__()


class Eventlog:

    def __init__(self, filename):

        self.log = pm4py.read.read_xes(filename)
        self.activities = {activity: mapping+1 for mapping, activity in
                           enumerate(pm4py.get_event_attribute_values(self.log, "concept:name"))}

        self.language, i = [], 0
        self.language_pm4py = variants_module.get_language(self.log)
        if "case:frequency" in self.log:
            frequencies = [int(trace.find("int").attrib["value"]) for trace in ET.parse(filename).getroot().findall("trace")]
            for i, seq in enumerate(self.language_pm4py):
                self.language.append(Trace(i, seq, "".join(seq), int(frequencies[i]), int(frequencies[i])/sum(frequencies), -1, -1))
        else:
            cases = pm4py.get_event_attribute_values(self.log, "case:concept:name")
            for i, seq in enumerate(self.language_pm4py):
                self.language.append(Trace(i, seq, "".join(seq), self.language_pm4py[seq]*len(cases), self.language_pm4py[seq], -1, -1))

        self.longest_trace_size = max(len(trace.get_seq()) for trace in self.language)
        self.max_activities = self.build_max_activities()
        self.prefixes = self.build_prefix_set()

        for trace in self.language:
            trace_mapping1 = 0
            for c, activity in enumerate(trace.get_seq()):
                trace_mapping1 += self.activities[activity] * pow(len(self.activities), c)
            trace.map1(trace_mapping1)

        p, primes = 0, list(sympy.primerange(len(self.language), 1000000))
        while p <= len(primes):
            mappings3 = []
            for trace in self.language:
                trace_mapping3 = 0
                for c, activity in enumerate(trace.get_seq()):
                    trace_mapping3 = (trace_mapping3 * len(self.activities) + self.activities[activity]) % primes[p]
                mappings3.append(trace_mapping3)
                trace.map3(trace_mapping3)
            if len(mappings3) == len(set(mappings3)):
                self.modulo = primes[p]
                break
            p += 100

    def get_log(self):
        return self.log

    def get_activities(self):
        return self.activities

    def get_max_activities(self):
        return self.max_activities

    def get_prefixes(self):
        return self.prefixes

    def get_language(self):
        return self.language

    def get_language_pm4py(self):
        return self.language_pm4py

    def __str__(self):
        result = "["
        for trace in self.language:
            result += trace.__str__() + ",\n"
        return result[:-2] + "]"

    ####################################################################################################

    def build_prefix_set(self):
        prefixes = set()
        prefixes.add(tuple([]))
        for trace in self.language:
            prefix = []
            for activity in trace.get_seq():
                prefix.append(activity)
                if tuple(prefix) not in prefixes:
                    prefixes.add(tuple(prefix))
        return prefixes

    def build_max_activities(self):
        max_activities = dict(zip(self.activities, [0 for _ in range(len(self.activities))]))
        for trace in self.language:
            for activity in self.activities:
                count = trace.get_seq().count(activity)
                if count > max_activities[activity]:
                    max_activities[activity] = count
        return max_activities

    ####################################################################################################

    def gen_lha_m1(self, pn):

        activities = self.get_activities()

        lha = "\n"

        # NbVariables
        lha += "NbVariables = " + str(4 + len([transition.get_name() for transition in pn.get_transitions()
                                               if transition.label is not None])) + ";\n"
        # NbLocations
        lha += "NbLocations = 3;\n\n"

        # Const
        lha += "const int n = " + str(len(activities)) + ";\n\n"

        # VariablesList
        lha += "VariablesList = {id,word,cpt,c"
        for transition in pn.get_transitions():
            if transition.get_label() is not None:
                lha += ",c_" + str(transition.get_name())
        lha += "};\n"
        # LocationsList
        lha += "LocationsList = {li,lfa,lfr};\n\n"

        # Variables
        lha += "PDF(Last(id), 1, 0, " + str(len(self.language)) + ");\n\n"

        # InitialLocations
        lha += "InitialLocations = {li};\n"
        # FinalLocations
        lha += "FinalLocations = {lfa};\n\n"

        # Locations
        lha += "Locations = {(li,TRUE);(lfa,(end=1));(lfr,TRUE);};\n\n"

        lha += "Edges = {\n"

        # Net edges
        for transition in pn.get_transitions():
            if not transition.is_silent():
                lha += "((li,li),{" + str(transition.get_name()) + "},#,{word=word+" + str(
                    activities[transition.label]) + "*(n^c), c=c+1, c_" + str(transition.get_name()) + "=c_" + str(
                    transition.get_name()) + "+1});\n"
        if len(pn.get_transitions()) != len(activities):
            lha += "((li,li),{"
            for transition in pn.get_transitions():
                if transition.is_silent():
                    lha += str(transition.get_name()) + ","
            lha = lha[:-1] + "},#,#);\n"

        # Accepting Edges
        for trace in self.language:
            lha += "((li,lfa),#,word=" + str(trace.get_mapping1()) + ",{id=" + str(trace.get_id()) + "});\n"

        # Rejecting Edges
        lha += "((li,lfr),#,cpt>=" + str(max([len(trace.get_seq()) for trace in self.language])) + ",#);\n"
        for transition in pn.get_transitions():
            if not transition.is_silent():
                lha += "((li,lfr),#,c_" + str(transition.get_name()) + ">=" + str(
                    self.max_activities[transition.get_label()] + 1) + ",#);\n"

        lha += "};\n"
        return lha

    def gen_lha_m3(self, pn):

        activities = self.get_activities()

        lha = "\n"

        # NbVariables
        lha += "NbVariables = " + str(5 + len([transition.get_name() for transition in pn.get_transitions()
                                               if transition.label is not None])) + ";\n"
        # NbLocations
        lha += "NbLocations = 4;\n\n"

        # Const
        lha += "const int n = " + str(len(activities)) + ";\n"
        lha += "const int modulo = " + str(self.modulo) + ";\n\n"

        # VariablesList
        lha += "VariablesList = {id,word,cpt,temp,quotient"
        for transition in pn.get_transitions():
            if transition.get_label() is not None:
                lha += ",c_" + str(transition.get_name())
        lha += "};\n"
        # LocationsList
        lha += "LocationsList = {li,lmod,lfa,lfr};\n\n"

        # Variables
        lha += "PDF(Last(id), 1, 0, " + str(len(self.language)) + ");\n\n"

        # InitialLocations
        lha += "InitialLocations = {li};\n"
        # FinalLocations
        lha += "FinalLocations = {lfa};\n\n"

        # Locations
        lha += "Locations = {(li,TRUE);(lmod,TRUE);(lfa,(end=1));(lfr,TRUE);};\n\n"

        lha += "Edges = {\n"

        # Modulo Edges
        lha += "((lmod,lmod),#,temp>=modulo,{temp=temp-modulo, quotient=quotient+1});\n"
        lha += "((lmod,li),#,temp<=modulo-1,{word=word-(quotient*modulo)});\n"

        # Net edges
        for transition in pn.get_transitions():
            if not transition.is_silent():
                lha += "((li,lmod),{" + str(transition.get_name()) + "},#,{word=word*n+" + str(
                    activities[transition.label]) + ", temp=word*n+" + str(
                    activities[transition.label]) + ", quotient=0, cpt=cpt+1, c_" + str(
                    transition.get_name()) + "=c_" + str(transition.get_name()) + "+1});\n"
        if len(pn.get_transitions()) != len(activities):
            lha += "((li,li),{"
            for transition in pn.get_transitions():
                if transition.is_silent():
                    lha += str(transition.get_name()) + ","
            lha = lha[:-1] + "},#,#);\n"

        # Accepting Edges
        for trace in self.language:
            lha += "((li,lfa),#,word=" + str(trace.get_mapping3()) + ",{id=" + str(trace.get_id()) + "});\n"

        # Rejecting Edges
        lha += "((li,lfr),#,cpt>=" + str(max([len(trace.get_seq()) for trace in self.language])) + ",#);\n"
        for transition in pn.get_transitions():
            if not transition.is_silent():
                lha += "((li,lfr),#,c_" + str(transition.get_name()) + ">=" + str(
                    self.max_activities[transition.get_label()] + 1) + ",#);\n"

        lha += "};\n"
        return lha

    def discover_pt_inductive(self, noise_threshold=0):
        global node_id, nodes, size_pi

        node_id, nodes, size_pi = -1, [], 0
        tree = pm4py.discover_process_tree_inductive(self.log,
                                   case_id_key="case:concept:name",
                                   activity_key="concept:name",
                                   timestamp_key="time:timestamp",
                                   noise_threshold=noise_threshold)

        def convert_tree(tree_node):
            global node_id, nodes, size_pi

            if tree_node.operator is None:
                return Node(tree_node.label if tree_node.label is not None else "")
            else:
                new_node, children = None, [convert_tree(child) for child in tree_node.children]
                node_id += 1
                if tree_node.operator == Operator.SEQUENCE:
                    new_node = Seq(node_id, children)
                elif tree_node.operator == Operator.LOOP:
                    new_node = Loop(node_id, children, size_pi)
                    size_pi += 1
                elif tree_node.operator == Operator.XOR:
                    if len(children) == 2:
                        new_node = BinaryChoice(node_id, children, size_pi)
                        size_pi += 1
                    else:
                        new_node = Choice(node_id, children, size_pi, size_pi+len(tree_node.children)-1)
                        size_pi += len(tree_node.children)
                elif tree_node.operator == Operator.PARALLEL:
                    if len(children) == 2:
                        new_node = BinaryParallel(node_id, children, size_pi)
                        size_pi += 1
                    else:
                        new_node = Parallel(node_id, children, size_pi, size_pi+len(tree_node.children)-1)
                        size_pi += len(tree_node.children)
                nodes.append(new_node)
                return new_node

        root = convert_tree(tree._get_root())
        return StochasticProcessTree(self, tree, root, nodes, size_pi, noise_threshold)

class Counts(object):

    def __init__(self):
        self.num_places, self.num_hidden, self.num_visible_trans = 0, 0, 0
        self.dict_skips, self.dict_loops = {}, {}

    def inc_places(self):
        self.num_places = self.num_places + 1

    def inc_no_hidden(self):
        self.num_hidden = self.num_hidden + 1

    def inc_no_visible(self):
        self.num_visible_trans = self.num_visible_trans + 1

def get_new_place(counts):
    counts.inc_places()
    return PetriNet.Place('p_' + str(counts.num_places))

def get_new_hidden_trans(counts, type_trans="unknown", weight=1):
    counts.inc_no_hidden()
    return PetriNet.Transition(type_trans + '_' + str(counts.num_hidden), None, properties={"weight":weight})

def get_transition(counts, label, weight=1):
    counts.inc_no_visible()
    return PetriNet.Transition(str(uuid.uuid4()), label, properties={"weight":weight})

def reduce_single_entry_transitions(net):
    cont = True
    while cont:
        cont = False
        single_entry_transitions = [t for t in net.transitions if t.label is None and len(t.in_arcs) == 1]
        for i in range(len(single_entry_transitions)):
            t = single_entry_transitions[i]
            source_place = list(t.in_arcs)[0].source
            target_places = [a.target for a in t.out_arcs]
            if len(source_place.in_arcs) == 1 and len(source_place.out_arcs) == 1:
                source_transition = list(source_place.in_arcs)[0].source
                source_transition.properties["weight"] = t.properties["weight"]
                remove_transition(net, t)
                remove_place(net, source_place)
                for p in target_places:
                    add_arc_from_to(source_transition, p, net)
                cont = True
                break
    return net

def reduce_single_exit_transitions(net):
    cont = True
    while cont:
        cont = False
        single_exit_transitions = [t for t in net.transitions if t.label is None and len(t.out_arcs) == 1]
        for i in range(len(single_exit_transitions)):
            t = single_exit_transitions[i]
            target_place = list(t.out_arcs)[0].target
            source_places = [a.source for a in t.in_arcs]
            if len(target_place.in_arcs) == 1 and len(target_place.out_arcs) == 1:
                target_transition = list(target_place.out_arcs)[0].target
                target_transition.properties["weight"] = t.properties["weight"]
                remove_transition(net, t)
                remove_place(net, target_place)
                for p in source_places:
                    add_arc_from_to(p, target_transition, net)
                cont = True
                break
    return net

def apply_simple_reduction(net):
    reduce_single_entry_transitions(net)
    reduce_single_exit_transitions(net)
    return net

def recursively_add_tree(parent_tree, tree, net, initial_entity_subtree, final_entity_subtree, counts, rec_depth, probs, weight_value,
                         force_add_skip=False):

    if type(initial_entity_subtree) is PetriNet.Transition:
        initial_place = get_new_place(counts)
        net.places.add(initial_place)
        add_arc_from_to(initial_entity_subtree, initial_place, net)
    else:
        initial_place = initial_entity_subtree
    if final_entity_subtree is not None and type(final_entity_subtree) is PetriNet.Place:
        final_place = final_entity_subtree
    else:
        final_place = get_new_place(counts)
        net.places.add(final_place)
        if final_entity_subtree is not None and type(final_entity_subtree) is PetriNet.Transition:
            add_arc_from_to(final_place, final_entity_subtree, net)
    tree_childs = [child for child in tree.children]

    if force_add_skip:
        invisible = get_new_hidden_trans(counts, type_trans="skip", weight=weight_value)
        add_arc_from_to(initial_place, invisible, net)
        add_arc_from_to(invisible, final_place, net)

    if tree.operator is None:
        trans = tree
        if trans.label is None:
            petri_trans = get_new_hidden_trans(counts, type_trans="skip", weight=weight_value)
        else:
            petri_trans = get_transition(counts, trans.label, weight=weight_value)
        net.transitions.add(petri_trans)
        add_arc_from_to(initial_place, petri_trans, net)
        add_arc_from_to(petri_trans, final_place, net)

    if tree.operator == Operator.XOR:
        if len(tree_childs) == 2:
            net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[0], net, initial_place, final_place,
                                                                   counts,
                                                                   rec_depth + 1, probs,
                                                                   weight_value * probs[tree.wi])
            net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[1], net, initial_place, final_place,
                                                                   counts,
                                                                   rec_depth + 1, probs,
                                                                   weight_value * (1-probs[tree.wi]))
        else:
            for i, subtree in enumerate(tree_childs):
                net, counts, intermediate_place = recursively_add_tree(tree, subtree, net, initial_place, final_place,
                                                                       counts,
                                                                       rec_depth + 1, probs,
                                                                       weight_value * probs[tree.wi+i])

    elif tree.operator == Operator.SEQUENCE:
        intermediate_place = initial_place
        for i in range(len(tree_childs)):
            final_connection_place = None
            if i == len(tree_childs) - 1:
                final_connection_place = final_place
            net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[i], net, intermediate_place,
                                                                   final_connection_place, counts,
                                                                   rec_depth + 1, probs, weight_value)

    elif tree.operator == Operator.PARALLEL:
        new_initial_trans = get_new_hidden_trans(counts, type_trans="tauSplit", weight=weight_value)
        net.transitions.add(new_initial_trans)
        add_arc_from_to(initial_place, new_initial_trans, net)
        new_final_trans = get_new_hidden_trans(counts, type_trans="tauJoin", weight=weight_value)
        net.transitions.add(new_final_trans)
        add_arc_from_to(new_final_trans, final_place, net)
        if len(tree_childs) == 2:
            net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[0], net, new_initial_trans,
                                                                   new_final_trans,
                                                                   counts,
                                                                   rec_depth + 1, probs, weight_value * probs[tree.wi])
            net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[1], net, new_initial_trans,
                                                                   new_final_trans,
                                                                   counts,
                                                                   rec_depth + 1, probs, weight_value * (1-probs[tree.wi]))
        else:
            for i, subtree in enumerate(tree_childs):
                net, counts, intermediate_place = recursively_add_tree(tree, tree_childs[1], net, new_initial_trans,
                                                                       new_final_trans,
                                                                       counts,
                                                                       rec_depth + 1, probs,
                                                                       weight_value * probs[tree.wi+i])

    elif tree.operator == Operator.LOOP:
        new_initial_place = get_new_place(counts)
        net.places.add(new_initial_place)
        init_loop_trans = get_new_hidden_trans(counts, type_trans="init_loop", weight=weight_value)
        net.transitions.add(init_loop_trans)
        add_arc_from_to(initial_place, init_loop_trans, net)
        add_arc_from_to(init_loop_trans, new_initial_place, net)
        initial_place = new_initial_place
        loop_trans = get_new_hidden_trans(counts, type_trans="loop", weight=weight_value*probs[tree.wi])
        net.transitions.add(loop_trans)
        net, counts, int1 = recursively_add_tree(tree, tree_childs[0], net, initial_place,
                                                 None, counts,
                                                 rec_depth + 1, probs, weight_value)
        int2 = None
        net, counts, int2 = recursively_add_tree(tree, tree_childs[1], net, int1,
                                                 int2, counts,
                                                 rec_depth + 1, probs, weight_value * probs[tree.wi])
        net, counts, int3 = recursively_add_tree(tree, ProcessTree(), net, int1,
                                                 final_place, counts,
                                                 rec_depth + 1, probs, weight_value * (1-probs[tree.wi]))
        looping_place = int2
        add_arc_from_to(looping_place, loop_trans, net)
        add_arc_from_to(loop_trans, initial_place, net)

    return net, counts, final_place

def convert_stp_to_spn_pm4py(tree, probs):

    counts = Counts()
    net = PetriNet('imdf_net_' + str(time.time()))
    initial_marking, final_marking = Marking(), Marking()

    source = get_new_place(counts)
    source.name = "source"
    net.places.add(source)

    sink = get_new_place(counts)
    sink.name = "sink"
    net.places.add(sink)

    initial_marking[source] = 1
    final_marking[sink] = 1

    net, counts, last_added_place = recursively_add_tree(tree, tree, net, source, sink, counts, 0, probs, 1)

    #apply_simple_reduction(net)

    places = list(net.places)
    for place in places:
        if len(place.out_arcs) == 0 and not place in final_marking:
            remove_place(net, place)
        if len(place.in_arcs) == 0 and not place in initial_marking:
            remove_place(net, place)

    return net, initial_marking, final_marking
