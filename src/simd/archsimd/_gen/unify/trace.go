// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

// debugDotInHTML, if true, includes dot code for all graphs in the HTML. Useful
// for debugging the dot output itself.
const debugDotInHTML = false

var Debug struct {
	// UnifyLog, if non-nil, receives a streaming text trace of unification.
	UnifyLog io.Writer

	// HTML, if non-nil, writes an HTML trace of unification to HTML.
	HTML io.Writer
}

type tracer struct {
	logw io.Writer

	enc yamlEncoder // Print consistent idents throughout

	saveTree bool // if set, record tree; required for HTML output

	path []string

	node  *traceTree
	trees []*traceTree
}

type traceTree struct {
	label string // Identifies this node as a child of parent
	v, w  *Value // Unification inputs
	envIn envSet
	res   *Value // Unification result
	env   envSet
	err   error // or error

	parent   *traceTree
	children []*traceTree
}

type tracerExit struct {
	t    *tracer
	len  int
	node *traceTree
}

func (t *tracer) enter(pat string, vals ...any) tracerExit {
	if t == nil {
		return tracerExit{}
	}

	label := fmt.Sprintf(pat, vals...)

	var p *traceTree
	if t.saveTree {
		p = t.node
		if p != nil {
			t.node = &traceTree{label: label, parent: p}
			p.children = append(p.children, t.node)
		}
	}

	t.path = append(t.path, label)
	return tracerExit{t, len(t.path) - 1, p}
}

func (t *tracer) enterVar(id *ident, branch int) tracerExit {
	if t == nil {
		return tracerExit{}
	}

	// Use the tracer's ident printer
	return t.enter("Var %s br %d", t.enc.idp.unique(id), branch)
}

func (te tracerExit) exit() {
	if te.t == nil {
		return
	}
	te.t.path = te.t.path[:te.len]
	te.t.node = te.node
}

func indentf(prefix string, pat string, vals ...any) string {
	s := fmt.Sprintf(pat, vals...)
	if len(prefix) == 0 {
		return s
	}
	if !strings.Contains(s, "\n") {
		return prefix + s
	}

	indent := prefix
	if strings.TrimLeft(prefix, " ") != "" {
		// Prefix has non-space characters in it. Construct an all space-indent.
		indent = strings.Repeat(" ", len(prefix))
	}
	return prefix + strings.ReplaceAll(s, "\n", "\n"+indent)
}

func yamlf(prefix string, node *yaml.Node) string {
	b, err := yaml.Marshal(node)
	if err != nil {
		return fmt.Sprintf("<marshal failed: %s>", err)
	}
	return strings.TrimRight(indentf(prefix, "%s", b), " \n")
}

func (t *tracer) logf(pat string, vals ...any) {
	if t == nil || t.logw == nil {
		return
	}
	prefix := fmt.Sprintf("[%s] ", strings.Join(t.path, "/"))
	s := indentf(prefix, pat, vals...)
	s = strings.TrimRight(s, " \n")
	fmt.Fprintf(t.logw, "%s\n", s)
}

func (t *tracer) traceUnify(v, w *Value, e envSet) {
	if t == nil {
		return
	}

	t.enc.e = e // Interpret values w.r.t. e
	t.logf("Unify\n%s\nwith\n%s\nin\n%s",
		yamlf("  ", t.enc.value(v)),
		yamlf("  ", t.enc.value(w)),
		yamlf("  ", t.enc.env(e)))
	t.enc.e = envSet{}

	if t.saveTree {
		if t.node == nil {
			t.node = &traceTree{}
			t.trees = append(t.trees, t.node)
		}
		t.node.v, t.node.w, t.node.envIn = v, w, e
	}
}

func (t *tracer) traceDone(res *Value, e envSet, err error) {
	if t == nil {
		return
	}

	if err != nil {
		t.logf("==> %s", err)
	} else {
		t.logf("==>\n%s", yamlf("  ", t.enc.closure(Closure{res, e})))
	}

	if t.saveTree {
		node := t.node
		if node == nil {
			panic("popped top of trace stack")
		}
		node.res, node.err = res, err
		node.env = e
	}
}
