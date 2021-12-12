# Copyright (c) 2021 Michael Auchter
# SPDX-License-Identifier: MIT

from pexpect import fdpexpect
import re
import socket


class Block:
    def __init__(self, pre, parent=None, delim="{}", indent="\t"):
        self.pre = pre
        self.parent = parent
        self.delim = delim
        self.indent = indent
        self.lines = []

        if parent is None:
            self.lines.append(self.pre + " " + self.delim[0])

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.parent is not None:
            self.parent.add(self.pre + " " + self.delim[0])
            for line in self.lines:
                self.parent.add(line)
            self.parent.add(self.delim[1])
        else:
            self.lines.append(self.delim[1])

    def add(self, s):
        self.lines.append(self.indent + s)

    def render(self):
        return "\n".join(self.lines)


class Filter:
    class FilterInOut:
        def __init__(self, index, attenuation, mult=""):
            self.index = int(index)
            self.attenuation = float(attenuation)
            self.mult = "* -1" if mult != "" else ""

        def __repr__(self):
            return f"FilterInOut(index={self.index}, attenuation={self.attenuation})"

    def __init__(self, index, name, coeff_set, inputs, outputs):
        self.index = int(index)
        self.name = name
        self.coeff_set = int(coeff_set)
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f"Filter(index={self.index}, name={self.name}, coeff_set={self.coeff_set}, inputs={self.inputs}, outputs={self.outputs})"

    def parse(s):
        filters = re.findall(r'\s*(\d+):\s+"([^"]+)"', s)
        coeffs = re.findall(r"\s*coeff set: (\d+)", s)
        inputs = re.findall(r"\s*from inputs:\s*(.*)", s)
        outputs = re.findall(r"\s*to outputs:\s*(.*)", s)
        inout_pat = re.compile(r"\s*(\d+)/([0-9\-\.]+)(/-1)?\s*")

        ret = []
        for (f, c, i, o) in zip(filters, coeffs, inputs, outputs):
            ins = [Filter.FilterInOut(*x) for x in re.findall(inout_pat, i)]
            outs = [Filter.FilterInOut(*x) for x in re.findall(inout_pat, o)]
            ret.append(
                Filter(index=f[0], name=f[1], coeff_set=c, inputs=ins, outputs=outs)
            )

        return ret


class InOut:
    def __init__(self, index, name, delay, subdelay, muted):
        self.index = index
        self.name = name
        self.delay = delay
        self.subdelay = subdelay
        self.muted = muted != ""

    def __repr__(self):
        return f'InOut(index={self.index}, name="{self.name}", delay={self.delay}, subdelay={self.subdelay}, muted={self.muted})'

    def parse(s):
        pattern = re.compile(
            r'\s*(\d+):\s*"([^"]+)" \(delay: (\d+):(\d+)\)\s*(\(muted\))?'
        )
        return [InOut(*x) for x in re.findall(pattern, s)]


class CoeffSet:
    def __init__(self, index, name, blocks):
        self.index = int(index)
        self.name = name
        self.blocks = int(blocks)

    def __repr__(self):
        return f'CoeffSet(index={self.index}, name="{self.name}", blocks={self.blocks})'

    def parse(s):
        pattern = re.compile(r'\s*(\d+):\s*"([^"]+)"\s\((\d+) blocks\)')
        return [CoeffSet(*x) for x in re.findall(pattern, s)]


class BruteFIR:
    def __init__(self, path=None, host=None, port=None):
        if path is not None:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(path)
        elif host is not None and port is not None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))
        else:
            raise RuntimeError("must specify either path or host+port")

        self._expect = fdpexpect.fdspawn(self._socket, timeout=10)
        self._expect.expect(">")
        self._update_all()

    def _update_all(self):
        self._filters = self._update("lf", Filter)
        self._inputs = self._update("li", InOut)
        self._outputs = self._update("lo", InOut)
        self._coeff_sets = self._update("lc", CoeffSet)

    def _update(self, cmd, cls):
        self._expect.sendline(cmd)
        self._expect.expect(">")
        s = self._expect.before.decode("ascii")
        return cls.parse(s)

    def _run_command(self, cmd):
        self._expect.sendline(cmd)
        self._expect.expect(">")
        err = self._expect.before.decode("ascii")
        if err != " ":
            raise RuntimeError(err)

    def graph(self):
        """
        Generate a graph of the filter flow, which can be rendered with Graphviz
        """

        self._update_all()

        digraph = Block("digraph")
        with digraph:
            digraph.add("rankdir = LR;")
            with Block("subgraph cluster_pipeline", digraph) as pipeline:
                with Block("subgraph cluster_inputs", pipeline) as inputs:
                    inputs.add('label = "inputs";')
                    inputs.add("graph[style=dotted];")
                    for i in self._inputs:
                        muted = "(muted)" if i.muted else ""
                        inputs.add(f'in_{i.index} [label="{i.name}\\n{muted}"];')
                with Block("subgraph cluster_outputs", pipeline) as outputs:
                    outputs.add('label = "outputs";')
                    outputs.add("graph[style=dotted];")
                    for o in self._outputs:
                        muted = "(muted)" if o.muted else ""
                        outputs.add(f'out_{o.index} [label="{o.name}\\n{muted}"];')
                with Block("subgraph cluster_filters", pipeline) as filters:
                    filters.add('label = "filters";')
                    filters.add("graph[style=dotted];")
                    for f in self._filters:
                        filters.add(f'filt_{f.index} [label="{f.name}"];')
            with Block("subgraph cluster_coeffs", digraph) as coeffs:
                coeffs.add('label = "coeffs";')
                for c in self._coeff_sets:
                    coeffs.add(
                        f'coeff_{c.index} [label="{c.name}\\n{c.blocks} blocks";shape=box;];'
                    )
            for f in self._filters:
                for i in f.inputs:
                    digraph.add(
                        f'in_{i.index} -> filt_{f.index} [label="{i.attenuation} dB {i.mult}"];'
                    )
                for o in f.outputs:
                    digraph.add(
                        f'filt_{f.index} -> out_{o.index} [label="{o.attenuation} dB {o.mult}"];'
                    )
                digraph.add(f"coeff_{f.coeff_set} -> filt_{f.index};")

        return digraph.render()

    def _validate_param(self, value, allowed_values):
        if type(value) is int:
            if value < len(allowed_values):
                return value
        if type(value) is str:
            for i, v in enumerate(allowed_values):
                if v.name == value:
                    return i
        raise RuntimeError("unknown value")

    def _normalize_params(self, values, allowed_values):
        if values is None:
            values = range(len(allowed_values))
        if type(values) in [int, str]:
            values = [values]

        return [self._validate_param(v, allowed_values) for v in values]

    def change_filter_coeffs(self, coeff_set, filters=None):
        """
        configure filters to use the specified coefficient set, atomically.
        by default, this will configure all filters; specify a list of filters to restrict this
        """

        coeff_set = self._validate_param(coeff_set, self._coeff_sets)
        filters = self._normalize_params(filters, self._filters)

        cmd = "; ".join([f"cfc {f} {coeff_set}" for f in filters])
        self._run_command(cmd)

    def toggle_mute_output(self, outputs=None):
        """
        Toggles mute output; by default, this operates on all outputs
        """

        outputs = self._normalize_params(outputs, self._outputs)
        cmd = "; ".join([f"tmo {o}" for o in outputs])
        self._run_command(cmd)

    def toggle_mute_input(self, inputs=None):
        """
        Toggles mute input; by default, this operates on all outputs
        """

        inputs = self._normalize_params(inputs, self._inputs)
        cmd = "; ".join([f"tmi {i}" for i in inputs])
        self._run_command(cmd)

    def change_filter_output_atten(self, atten, filters=None, outputs=None):
        """
        Change output attenutation for filters
        By default, this will operate on all outputs of all filters.
        """

        filters = self._normalize_params(filters, self._filters)

        cmds = []
        # If outputs weren't specified, set attenuation on all outputs of the
        # selected filters
        if outputs is None:
            for idx in filters:
                for out in self._filters[idx].outputs:
                    cmds.append(f"cfoa {idx} {out.index} {atten}")
        else:
            outputs = self._normalize_params(outputs, self._outputs)
            for filt_idx in filters:
                for out_idx in outputs:
                    cmds.append(f"cfoa {filt_idx} {out_idx} {atten}")

        cmd = "; ".join(cmds)
        self._run_command(cmd)

    def change_filter_input_atten(self, atten, filters=None, inputs=None):
        """
        Change input attenutation for filters
        By default, this will operate on all inputs of all filters.
        """

        filters = self._normalize_params(filters, self._filters)

        cmds = []
        # If inputs weren't specified, set attenuation on all outputs of the
        # selected filters
        if inputs is None:
            for idx in filters:
                for i in self._filters[idx].inputs:
                    cmds.append(f"cfia {idx} {i.index} {atten}")
        else:
            inputs = self._normalize_params(inputs, self._inputs)
            for filt_idx in filters:
                for in_idx in inputs:
                    cmds.append(f"cfia {filt_idx} {in_idx} {atten}")

        cmd = "; ".join(cmds)
        self._run_command(cmd)

    def get_filter_coeffs(self, filters=None):
        """
        Output the filter coefficient sets for each filter.
        """

        filters = self._normalize_params(filters, self._filters)
        self._update_all()

        d = {}
        for f in filters:
            d[self._filters[f].name] = self._coeff_sets[self._filters[f].coeff_set].name
        return d

    def get_coeff_sets(self):
        return [c.name for c in self._coeff_sets]

    def get_outputs(self):
        return [o.name for o in self._outputs]

    def get_inputs(self):
        return [i.name for i in self._inputs]

    def get_filters(self):
        return [f.name for f in self._filters]
