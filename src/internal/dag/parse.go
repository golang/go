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
	"strings"
)

// Parse returns a map m such that m[p][d] == true when there is a
// path from p to d.
func Parse(dag string) (map[string]map[string]bool, error) {
	allowed := map[string]map[string]bool{"NONE": {}}
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
			if allowed[def] != nil {
				errorf("multiple definitions for %s", def)
			}
			allowed[def] = make(map[string]bool)
			for _, less := range r.less {
				if allowed[less] == nil {
					errorf("use of %s before its definition", less)
				}
				allowed[def][less] = true
			}
		}
	}

	// Check for missing definition.
	for _, tos := range allowed {
		for to := range tos {
			if allowed[to] == nil {
				errorf("missing definition for %s", to)
			}
		}
	}

	// Complete transitive closure.
	for k := range allowed {
		for i := range allowed {
			for j := range allowed {
				if i != k && k != j && allowed[i][k] && allowed[k][j] {
					if i == j {
						// Can only happen along with a "use of X before deps" error above,
						// but this error is more specific - it makes clear that reordering the
						// rules will not be enough to fix the problem.
						errorf("graph cycle: %s < %s < %s", j, k, i)
					}
					allowed[i][j] = true
				}
			}
		}
	}

	// Check negative assertions against completed allowed graph.
	for _, bad := range disallowed {
		for _, less := range bad.less {
			for _, def := range bad.def {
				if allowed[def][less] {
					errorf("graph edge assertion failed: %s !< %s", less, def)
				}
			}
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("%s", strings.Join(errors, "\n"))
	}

	return allowed, nil
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
