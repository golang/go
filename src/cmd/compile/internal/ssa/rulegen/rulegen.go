// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program generates Go code that applies rewrite rules to a Value.
// The generated code implements a function of type func (v *Value) bool
// which returns true iff if did something.
// Ideas stolen from Swift: http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-2000-2.html

// Run with something like "go run rulegen.go lower_amd64.rules lowerBlockAmd64 lowerValueAmd64 lowerAmd64.go"

package main

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"fmt"
	"go/format"
	"io"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strings"
)

// rule syntax:
//  sexpr [&& extra conditions] -> sexpr
//
// sexpr are s-expressions (lisp-like parenthesized groupings)
// sexpr ::= (opcode sexpr*)
//         | variable
//         | [aux]
//         | <type>
//         | {code}
//
// aux      ::= variable | {code}
// type     ::= variable | {code}
// variable ::= some token
// opcode   ::= one of the opcodes from ../op.go (without the Op prefix)

// extra conditions is just a chunk of Go that evaluates to a boolean.  It may use
// variables declared in the matching sexpr.  The variable "v" is predefined to be
// the value matched by the entire rule.

// If multiple rules match, the first one in file order is selected.

func main() {
	if len(os.Args) < 4 || len(os.Args) > 5 {
		fmt.Printf("usage: go run rulegen.go <rule file> <block function name> <value function name> [<output file>]")
		os.Exit(1)
	}
	rulefile := os.Args[1]
	blockfn := os.Args[2]
	valuefn := os.Args[3]

	// Open input file.
	text, err := os.Open(rulefile)
	if err != nil {
		log.Fatalf("can't read rule file: %v", err)
	}

	// oprules contains a list of rules for each block and opcode
	blockrules := map[string][]string{}
	oprules := map[string][]string{}

	// read rule file
	scanner := bufio.NewScanner(text)
	for scanner.Scan() {
		line := scanner.Text()
		if i := strings.Index(line, "//"); i >= 0 {
			// Remove comments.  Note that this isn't string safe, so
			// it will truncate lines with // inside strings.  Oh well.
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		op := strings.Split(line, " ")[0][1:]
		if strings.HasPrefix(op, "Block") {
			blockrules[op] = append(blockrules[op], line)
		} else {
			oprules[op] = append(oprules[op], line)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("scanner failed: %v\n", err)
	}

	// Start output buffer, write header.
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// autogenerated from %s: do not edit!\n", rulefile)
	fmt.Fprintf(w, "// generated with: go run rulegen/rulegen.go %s\n", strings.Join(os.Args[1:], " "))
	fmt.Fprintln(w, "package ssa")
	fmt.Fprintf(w, "func %s(v *Value, config *Config) bool {\n", valuefn)

	// generate code for each rule
	fmt.Fprintf(w, "switch v.Op {\n")
	var ops []string
	for op := range oprules {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		fmt.Fprintf(w, "case Op%s:\n", op)
		for _, rule := range oprules[op] {
			// Note: we use a hash to identify the rule so that its
			// identity is invariant to adding/removing rules elsewhere
			// in the rules file.  This is useful to squash spurious
			// diffs that would occur if we used rule index.
			rulehash := fmt.Sprintf("%02x", md5.Sum([]byte(rule)))

			// split at ->
			s := strings.Split(rule, "->")
			if len(s) != 2 {
				log.Fatalf("no arrow in rule %s", rule)
			}
			lhs := strings.TrimSpace(s[0])
			result := strings.TrimSpace(s[1])

			// split match into matching part and additional condition
			match := lhs
			cond := ""
			if i := strings.Index(match, "&&"); i >= 0 {
				cond = strings.TrimSpace(match[i+2:])
				match = strings.TrimSpace(match[:i])
			}

			fmt.Fprintf(w, "// match: %s\n", match)
			fmt.Fprintf(w, "// cond: %s\n", cond)
			fmt.Fprintf(w, "// result: %s\n", result)

			fail := fmt.Sprintf("{\ngoto end%s\n}\n", rulehash)

			fmt.Fprintf(w, "{\n")
			genMatch(w, match, fail)

			if cond != "" {
				fmt.Fprintf(w, "if !(%s) %s", cond, fail)
			}

			genResult(w, result)
			fmt.Fprintf(w, "return true\n")

			fmt.Fprintf(w, "}\n")
			fmt.Fprintf(w, "goto end%s\n", rulehash) // use label
			fmt.Fprintf(w, "end%s:;\n", rulehash)
		}
	}
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "return false\n")
	fmt.Fprintf(w, "}\n")

	// Generate block rewrite function.
	fmt.Fprintf(w, "func %s(b *Block) bool {\n", blockfn)
	fmt.Fprintf(w, "switch b.Kind {\n")
	ops = nil
	for op := range blockrules {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		fmt.Fprintf(w, "case %s:\n", op)
		for _, rule := range blockrules[op] {
			rulehash := fmt.Sprintf("%02x", md5.Sum([]byte(rule)))
			// split at ->
			s := strings.Split(rule, "->")
			if len(s) != 2 {
				log.Fatalf("no arrow in rule %s", rule)
			}
			lhs := strings.TrimSpace(s[0])
			result := strings.TrimSpace(s[1])

			// split match into matching part and additional condition
			match := lhs
			cond := ""
			if i := strings.Index(match, "&&"); i >= 0 {
				cond = strings.TrimSpace(match[i+2:])
				match = strings.TrimSpace(match[:i])
			}

			fmt.Fprintf(w, "// match: %s\n", match)
			fmt.Fprintf(w, "// cond: %s\n", cond)
			fmt.Fprintf(w, "// result: %s\n", result)

			fail := fmt.Sprintf("{\ngoto end%s\n}\n", rulehash)

			fmt.Fprintf(w, "{\n")
			s = split(match[1 : len(match)-1]) // remove parens, then split

			// check match of control value
			if s[1] != "nil" {
				fmt.Fprintf(w, "v := b.Control\n")
				genMatch0(w, s[1], "v", fail, map[string]string{}, false)
			}

			// assign successor names
			succs := s[2:]
			for i, a := range succs {
				if a != "_" {
					fmt.Fprintf(w, "%s := b.Succs[%d]\n", a, i)
				}
			}

			if cond != "" {
				fmt.Fprintf(w, "if !(%s) %s", cond, fail)
			}

			// Rule matches.  Generate result.
			t := split(result[1 : len(result)-1]) // remove parens, then split
			newsuccs := t[2:]

			// Check if newsuccs is a subset of succs.
			m := map[string]bool{}
			for _, succ := range succs {
				if m[succ] {
					log.Fatalf("can't have a repeat successor name %s in %s", succ, rule)
				}
				m[succ] = true
			}
			for _, succ := range newsuccs {
				if !m[succ] {
					log.Fatalf("unknown successor %s in %s", succ, rule)
				}
				delete(m, succ)
			}

			// Modify predecessor lists for no-longer-reachable blocks
			for succ := range m {
				fmt.Fprintf(w, "removePredecessor(b, %s)\n", succ)
			}

			fmt.Fprintf(w, "b.Kind = %s\n", t[0])
			if t[1] == "nil" {
				fmt.Fprintf(w, "b.Control = nil\n")
			} else {
				fmt.Fprintf(w, "b.Control = %s\n", genResult0(w, t[1], new(int), false))
			}
			if len(newsuccs) < len(succs) {
				fmt.Fprintf(w, "b.Succs = b.Succs[:%d]\n", len(newsuccs))
			}
			for i, a := range newsuccs {
				fmt.Fprintf(w, "b.Succs[%d] = %s\n", i, a)
			}

			fmt.Fprintf(w, "return true\n")

			fmt.Fprintf(w, "}\n")
			fmt.Fprintf(w, "goto end%s\n", rulehash) // use label
			fmt.Fprintf(w, "end%s:;\n", rulehash)
		}
	}
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "return false\n")
	fmt.Fprintf(w, "}\n")

	// gofmt result
	b := w.Bytes()
	b, err = format.Source(b)
	if err != nil {
		panic(err)
	}

	// Write to a file if given, otherwise stdout.
	if len(os.Args) >= 5 {
		err = ioutil.WriteFile(os.Args[4], b, 0666)
	} else {
		_, err = os.Stdout.Write(b)
	}
	if err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}
}

func genMatch(w io.Writer, match, fail string) {
	genMatch0(w, match, "v", fail, map[string]string{}, true)
}

func genMatch0(w io.Writer, match, v, fail string, m map[string]string, top bool) {
	if match[0] != '(' {
		if x, ok := m[match]; ok {
			// variable already has a definition.  Check whether
			// the old definition and the new definition match.
			// For example, (add x x).  Equality is just pointer equality
			// on Values (so cse is important to do before lowering).
			fmt.Fprintf(w, "if %s != %s %s", v, x, fail)
			return
		}
		// remember that this variable references the given value
		if match == "_" {
			return
		}
		m[match] = v
		fmt.Fprintf(w, "%s := %s\n", match, v)
		return
	}

	// split body up into regions.  Split by spaces/tabs, except those
	// contained in () or {}.
	s := split(match[1 : len(match)-1]) // remove parens, then split

	// check op
	if !top {
		fmt.Fprintf(w, "if %s.Op != Op%s %s", v, s[0], fail)
	}

	// check type/aux/args
	argnum := 0
	for _, a := range s[1:] {
		if a[0] == '<' {
			// type restriction
			t := a[1 : len(a)-1] // remove <>
			if t[0] == '{' {
				// code.  We must match the results of this code.
				fmt.Fprintf(w, "if %s.Type != %s %s", v, t[1:len(t)-1], fail)
			} else {
				// variable
				if u, ok := m[t]; ok {
					// must match previous variable
					fmt.Fprintf(w, "if %s.Type != %s %s", v, u, fail)
				} else {
					m[t] = v + ".Type"
					fmt.Fprintf(w, "%s := %s.Type\n", t, v)
				}
			}
		} else if a[0] == '[' {
			// aux restriction
			x := a[1 : len(a)-1] // remove []
			if x[0] == '{' {
				// code
				fmt.Fprintf(w, "if %s.Aux != %s %s", v, x[1:len(x)-1], fail)
			} else {
				// variable
				if y, ok := m[x]; ok {
					fmt.Fprintf(w, "if %s.Aux != %s %s", v, y, fail)
				} else {
					m[x] = v + ".Aux"
					fmt.Fprintf(w, "%s := %s.Aux\n", x, v)
				}
			}
		} else if a[0] == '{' {
			fmt.Fprintf(w, "if %s.Args[%d] != %s %s", v, argnum, a[1:len(a)-1], fail)
			argnum++
		} else {
			// variable or sexpr
			genMatch0(w, a, fmt.Sprintf("%s.Args[%d]", v, argnum), fail, m, false)
			argnum++
		}
	}
}

func genResult(w io.Writer, result string) {
	genResult0(w, result, new(int), true)
}
func genResult0(w io.Writer, result string, alloc *int, top bool) string {
	if result[0] != '(' {
		// variable
		if top {
			fmt.Fprintf(w, "v.Op = %s.Op\n", result)
			fmt.Fprintf(w, "v.Aux = %s.Aux\n", result)
			fmt.Fprintf(w, "v.resetArgs()\n")
			fmt.Fprintf(w, "v.AddArgs(%s.Args...)\n", result)
		}
		return result
	}

	s := split(result[1 : len(result)-1]) // remove parens, then split
	var v string
	var hasType bool
	if top {
		v = "v"
		fmt.Fprintf(w, "v.Op = Op%s\n", s[0])
		fmt.Fprintf(w, "v.Aux = nil\n")
		fmt.Fprintf(w, "v.resetArgs()\n")
		hasType = true
	} else {
		v = fmt.Sprintf("v%d", *alloc)
		*alloc++
		fmt.Fprintf(w, "%s := v.Block.NewValue(Op%s, TypeInvalid, nil)\n", v, s[0])
	}
	for _, a := range s[1:] {
		if a[0] == '<' {
			// type restriction
			t := a[1 : len(a)-1] // remove <>
			if t[0] == '{' {
				t = t[1 : len(t)-1] // remove {}
			}
			fmt.Fprintf(w, "%s.Type = %s\n", v, t)
			hasType = true
		} else if a[0] == '[' {
			// aux restriction
			x := a[1 : len(a)-1] // remove []
			if x[0] == '{' {
				x = x[1 : len(x)-1] // remove {}
			}
			fmt.Fprintf(w, "%s.Aux = %s\n", v, x)
		} else if a[0] == '{' {
			fmt.Fprintf(w, "%s.AddArg(%s)\n", v, a[1:len(a)-1])
		} else {
			// regular argument (sexpr or variable)
			x := genResult0(w, a, alloc, false)
			fmt.Fprintf(w, "%s.AddArg(%s)\n", v, x)
		}
	}
	if !hasType {
		log.Fatalf("sub-expression %s must have a type", result)
	}
	return v
}

func split(s string) []string {
	var r []string

outer:
	for s != "" {
		d := 0               // depth of ({[<
		var open, close byte // opening and closing markers ({[< or )}]>
		nonsp := false       // found a non-space char so far
		for i := 0; i < len(s); i++ {
			switch {
			case d == 0 && s[i] == '(':
				open, close = '(', ')'
				d++
			case d == 0 && s[i] == '<':
				open, close = '<', '>'
				d++
			case d == 0 && s[i] == '[':
				open, close = '[', ']'
				d++
			case d == 0 && s[i] == '{':
				open, close = '{', '}'
				d++
			case d == 0 && (s[i] == ' ' || s[i] == '\t'):
				if nonsp {
					r = append(r, strings.TrimSpace(s[:i]))
					s = s[i:]
					continue outer
				}
			case d > 0 && s[i] == open:
				d++
			case d > 0 && s[i] == close:
				d--
			default:
				nonsp = true
			}
		}
		if d != 0 {
			panic("imbalanced expression: " + s)
		}
		if nonsp {
			r = append(r, strings.TrimSpace(s))
		}
		break
	}
	return r
}
