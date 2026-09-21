// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"bytes"
	"fmt"
	"html"
	"io"
	"os"
	"os/exec"
	"strings"
)

const maxNodes = 30

type dotEncoder struct {
	w *bytes.Buffer

	idGen    int // Node name generation
	valLimit int // Limit the number of Values in a subgraph

	idp identPrinter
}

func newDotEncoder() *dotEncoder {
	return &dotEncoder{
		w: new(bytes.Buffer),
	}
}

func (enc *dotEncoder) clear() {
	enc.w.Reset()
	enc.idGen = 0
}

func (enc *dotEncoder) writeTo(w io.Writer) {
	fmt.Fprintln(w, "digraph {")
	// Use the "new" ranking algorithm, which lets us put nodes from different
	// clusters in the same rank.
	fmt.Fprintln(w, "newrank=true;")
	fmt.Fprintln(w, "node [shape=box, ordering=out];")

	w.Write(enc.w.Bytes())
	fmt.Fprintln(w, "}")
}

func (enc *dotEncoder) writeSvg(w io.Writer) error {
	cmd := exec.Command("dot", "-Tsvg")
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	enc.writeTo(in)
	in.Close()
	if err := cmd.Wait(); err != nil {
		return err
	}
	// Trim SVG header so the result can be embedded
	//
	// TODO: In Graphviz 10.0.1, we could use -Tsvg_inline.
	svg := out.Bytes()
	if i := bytes.Index(svg, []byte("<svg ")); i >= 0 {
		svg = svg[i:]
	}
	_, err = w.Write(svg)
	return err
}

func (enc *dotEncoder) newID(f string) string {
	id := fmt.Sprintf(f, enc.idGen)
	enc.idGen++
	return id
}

func (enc *dotEncoder) node(label, sublabel string) string {
	id := enc.newID("n%d")
	l := html.EscapeString(label)
	if sublabel != "" {
		l += fmt.Sprintf("<BR ALIGN=\"CENTER\"/><FONT POINT-SIZE=\"10\">%s</FONT>", html.EscapeString(sublabel))
	}
	fmt.Fprintf(enc.w, "%s [label=<%s>];\n", id, l)
	return id
}

func (enc *dotEncoder) edge(from, to string, label string, args ...any) {
	l := fmt.Sprintf(label, args...)
	fmt.Fprintf(enc.w, "%s -> %s [label=%q];\n", from, to, l)
}

func (enc *dotEncoder) valueSubgraph(v *Value) {
	enc.valLimit = maxNodes
	cID := enc.newID("cluster_%d")
	fmt.Fprintf(enc.w, "subgraph %s {\n", cID)
	fmt.Fprintf(enc.w, "style=invis;")
	vID := enc.value(v)
	fmt.Fprintf(enc.w, "}\n")
	// We don't need the IDs right now.
	_, _ = cID, vID
}

func (enc *dotEncoder) value(v *Value) string {
	if enc.valLimit <= 0 {
		id := enc.newID("n%d")
		fmt.Fprintf(enc.w, "%s [label=\"...\", shape=triangle];\n", id)
		return id
	}
	enc.valLimit--

	switch vd := v.Domain.(type) {
	default:
		panic(fmt.Sprintf("unknown domain type %T", vd))

	case nil:
		return enc.node("_|_", "")

	case Top:
		return enc.node("_", "")

		// TODO: Like in YAML, figure out if this is just a sum. In dot, we
		// could say any unentangled variable is a sum, and if it has more than
		// one reference just share the node.

	// case Sum:
	// 	node := enc.node("Sum", "")
	// 	for i, elt := range vd.vs {
	// 		enc.edge(node, enc.value(elt), "%d", i)
	// 		if enc.valLimit <= 0 {
	// 			break
	// 		}
	// 	}
	// 	return node

	case Def:
		node := enc.node("Def", "")
		for k, v := range vd.All() {
			enc.edge(node, enc.value(v), "%s", k)
			if enc.valLimit <= 0 {
				break
			}
		}
		return node

	case Tuple:
		if vd.repeat == nil {
			label := "Tuple"
			node := enc.node(label, "")
			for i, elt := range vd.vs {
				enc.edge(node, enc.value(elt), "%d", i)
				if enc.valLimit <= 0 {
					break
				}
			}
			return node
		} else {
			// TODO
			return enc.node("TODO: Repeat", "")
		}

	case String:
		switch vd.kind {
		case stringExact:
			return enc.node(fmt.Sprintf("%q", vd.exact), "")
		case stringRegex:
			var parts []string
			for _, re := range vd.re {
				parts = append(parts, fmt.Sprintf("%q", re))
			}
			return enc.node(strings.Join(parts, "&"), "")
		}
		panic("bad String kind")

	case Var:
		return enc.node(fmt.Sprintf("Var %s", enc.idp.unique(vd.id)), "")
	}
}

func (enc *dotEncoder) envSubgraph(e envSet) {
	enc.valLimit = maxNodes
	cID := enc.newID("cluster_%d")
	fmt.Fprintf(enc.w, "subgraph %s {\n", cID)
	fmt.Fprintf(enc.w, "style=invis;")
	vID := enc.env(e.root)
	fmt.Fprintf(enc.w, "}\n")
	_, _ = cID, vID
}

func (enc *dotEncoder) env(e *envExpr) string {
	switch e.kind {
	default:
		panic("bad kind")
	case envZero:
		return enc.node("0", "")
	case envUnit:
		return enc.node("1", "")
	case envBinding:
		node := enc.node(fmt.Sprintf("%q :", enc.idp.unique(e.id)), "")
		enc.edge(node, enc.value(e.val), "")
		return node
	case envProduct:
		node := enc.node("⨯", "")
		for _, op := range e.operands {
			enc.edge(node, enc.env(op), "")
		}
		return node
	case envSum:
		node := enc.node("+", "")
		for _, op := range e.operands {
			enc.edge(node, enc.env(op), "")
		}
		return node
	}
}
