// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dag implements a language for expressing directed acyclic
// graphs.
//
// The general syntax of a rule is:
//
//	a, b < c, d;
//
// which means c and d come after a and b in the partial order
// (that is, there are edges from c and d to a and b),
// but doesn't provide a relative order between a vs b or c vs d.
//
// The rules can chain together, as in:
//
//	e < f, g < h;
//
// which is equivalent to
//
//	e < f, g;
//	f, g < h;
//
// Except for the special bottom element "NONE", each name
// must appear exactly once on the right-hand side of any rule.
// That rule serves as the definition of the allowed successor
// for that name. The definition must appear before any uses
// of the name on the left-hand side of a rule. (That is, the
// rules themselves must be ordered according to the partial
// order, for easier reading by people.)
//
// Negative assertions double-check the partial order:
//
//	i !< j
//
// means that it must NOT be the case that i < j.
// Negative assertions may appear anywhere in the rules,
// even before i and j have been defined.
//
// Comments begin with #.
package dag

import (
	"fmt"
	"sort"
	"strings"
)

type Graph struct {
	Nodes   []string
	byLabel map[string]int
	edges   map[string]map[string]bool
}

func newGraph() *Graph {
	return &Graph{byLabel: map[string]int{}, edges: map[string]map[string]bool{}}
}

func (g *Graph) addNode(label string) bool {
	if _, ok := g.byLabel[label]; ok {
		return false
	}
	g.byLabel[label] = len(g.Nodes)
	g.Nodes = append(g.Nodes, label)
	g.edges[label] = map[string]bool{}
	return true
}

func (g *Graph) AddEdge(from, to string) {
	g.edges[from][to] = true
}

func (g *Graph) DelEdge(from, to string) {
	delete(g.edges[from], to)
}

func (g *Graph) HasEdge(from, to string) bool {
	return g.edges[from] != nil && g.edges[from][to]
}

func (g *Graph) Edges(from string) []string {
	edges := make([]string, 0, 16)
	for k := range g.edges[from] {
		edges = append(edges, k)
	}
	sort.Slice(edges, func(i, j int) bool { return g.byLabel[edges[i]] < g.byLabel[edges[j]] })
	return edges
}

// Parse parses the DAG language and returns the transitive closure of
// the described graph. In the returned graph, there is an edge from "b"
// to "a" if b < a (or a > b) in the partial order.
func Parse(dag string) (*Graph, error) {
	g := newGraph()
	disallowed := []rule{}

	rules, err := parseRules(dag)
	if err != nil {
		return nil, err
	}

	// TODO: Add line numbers to errors.
	var errors []string
	errorf := func(format string, a ...any) {
		errors = append(errors, fmt.Sprintf(format, a...))
	}
	for _, r := range rules {
		if r.op == "!<" {
			disallowed = append(disallowed, r)
			continue
		}
		for _, def := range r.def {
			if def == "NONE" {
				errorf("NONE cannot be a predecessor")
				continue
			}
			if !g.addNode(def) {
				errorf("multiple definitions for %s", def)
			}
			for _, less := range r.less {
				if less == "NONE" {
					continue
				}
				if _, ok := g.byLabel[less]; !ok {
					errorf("use of %s before its definition", less)
				} else {
					g.AddEdge(def, less)
				}
			}
		}
	}

	// Check for missing definition.
	for _, tos := range g.edges {
		for to := range tos {
			if g.edges[to] == nil {
				errorf("missing definition for %s", to)
			}
		}
	}

	// Complete transitive closure.
	for _, k := range g.Nodes {
		for _, i := range g.Nodes {
			for _, j := range g.Nodes {
				if i != k && k != j && g.HasEdge(i, k) && g.HasEdge(k, j) {
					if i == j {
						// Can only happen along with a "use of X before deps" error above,
						// but this error is more specific - it makes clear that reordering the
						// rules will not be enough to fix the problem.
						errorf("graph cycle: %s < %s < %s", j, k, i)
					}
					g.AddEdge(i, j)
				}
			}
		}
	}

	// Check negative assertions against completed allowed graph.
	for _, bad := range disallowed {
		for _, less := range bad.less {
			for _, def := range bad.def {
				if g.HasEdge(def, less) {
					errorf("graph edge assertion failed: %s !< %s", less, def)
				}
			}
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("%s", strings.Join(errors, "\n"))
	}

	return g, nil
}

// A rule is a line in the DAG language where "less < def" or "less !< def".
type rule struct {
	less []string
	op   string // Either "<" or "!<"
	def  []string
}

type syntaxError string

func (e syntaxError) Error() string {
	return string(e)
}

// parseRules parses the rules of a DAG.
func parseRules(rules string) (out []rule, err error) {
	defer func() {
		e := recover()
		switch e := e.(type) {
		case nil:
			return
		case syntaxError:
			err = e
		default:
			panic(e)
		}
	}()
	p := &rulesParser{lineno: 1, text: rules}

	var prev []string
	var op string
	for {
		list, tok := p.nextList()
		if tok == "" {
			if prev == nil {
				break
			}
			p.syntaxError("unexpected EOF")
		}
		if prev != nil {
			out = append(out, rule{prev, op, list})
		}
		prev = list
		if tok == ";" {
			prev = nil
			op = ""
			continue
		}
		if tok != "<" && tok != "!<" {
			p.syntaxError("missing <")
		}
		op = tok
	}

	return out, err
}

// A rulesParser parses the depsRules syntax described above.
type rulesParser struct {
	lineno   int
	lastWord string
	text     string
}

// syntaxError reports a parsing error.
func (p *rulesParser) syntaxError(msg string) {
	panic(syntaxError(fmt.Sprintf("parsing graph: line %d: syntax error: %s near %s", p.lineno, msg, p.lastWord)))
}

// nextList parses and returns a comma-separated list of names.
func (p *rulesParser) nextList() (list []string, token string) {
	for {
		tok := p.nextToken()
		switch tok {
		case "":
			if len(list) == 0 {
				return nil, ""
			}
			fallthrough
		case ",", "<", "!<", ";":
			p.syntaxError("bad list syntax")
		}
		list = append(list, tok)

		tok = p.nextToken()
		if tok != "," {
			return list, tok
		}
	}
}

// nextToken returns the next token in the deps rules,
// one of ";" "," "<" "!<" or a name.
func (p *rulesParser) nextToken() string {
	for {
		if p.text == "" {
			return ""
		}
		switch p.text[0] {
		case ';', ',', '<':
			t := p.text[:1]
			p.text = p.text[1:]
			return t

		case '!':
			if len(p.text) < 2 || p.text[1] != '<' {
				p.syntaxError("unexpected token !")
			}
			p.text = p.text[2:]
			return "!<"

		case '#':
			i := strings.Index(p.text, "\n")
			if i < 0 {
				i = len(p.text)
			}
			p.text = p.text[i:]
			continue

		case '\n':
			p.lineno++
			fallthrough
		case ' ', '\t':
			p.text = p.text[1:]
			continue

		default:
			i := strings.IndexAny(p.text, "!;,<#\n \t")
			if i < 0 {
				i = len(p.text)
			}
			t := p.text[:i]
			p.text = p.text[i:]
			p.lastWord = t
			return t
		}
	}
}
