// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gen

// This program generates Go code that applies rewrite rules to a Value.
// The generated code implements a function of type func (v *Value) bool
// which returns true iff if did something.
// Ideas stolen from Swift: http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-2000-2.html

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
)

// rule syntax:
//  sexpr [&& extra conditions] -> [@block] sexpr
//
// sexpr are s-expressions (lisp-like parenthesized groupings)
// sexpr ::= [variable:](opcode sexpr*)
//         | variable
//         | <type>
//         | [auxint]
//         | {aux}
//
// aux      ::= variable | {code}
// type     ::= variable | {code}
// variable ::= some token
// opcode   ::= one of the opcodes from the *Ops.go files

// extra conditions is just a chunk of Go that evaluates to a boolean. It may use
// variables declared in the matching sexpr. The variable "v" is predefined to be
// the value matched by the entire rule.

// If multiple rules match, the first one in file order is selected.

var (
	genLog = flag.Bool("log", false, "generate code that logs; for debugging only")
)

type Rule struct {
	rule string
	loc  string // file name & line number
}

func (r Rule) String() string {
	return fmt.Sprintf("rule %q at %s", r.rule, r.loc)
}

// parse returns the matching part of the rule, additional conditions, and the result.
func (r Rule) parse() (match, cond, result string) {
	s := strings.Split(r.rule, "->")
	if len(s) != 2 {
		log.Fatalf("no arrow in %s", r)
	}
	match = strings.TrimSpace(s[0])
	result = strings.TrimSpace(s[1])
	cond = ""
	if i := strings.Index(match, "&&"); i >= 0 {
		cond = strings.TrimSpace(match[i+2:])
		match = strings.TrimSpace(match[:i])
	}
	return match, cond, result
}

func genRules(arch arch) {
	// Open input file.
	text, err := os.Open(arch.name + ".rules")
	if err != nil {
		log.Fatalf("can't read rule file: %v", err)
	}

	// oprules contains a list of rules for each block and opcode
	blockrules := map[string][]Rule{}
	oprules := map[string][]Rule{}

	// read rule file
	scanner := bufio.NewScanner(text)
	rule := ""
	var lineno int
	var ruleLineno int // line number of "->"
	for scanner.Scan() {
		lineno++
		line := scanner.Text()
		if i := strings.Index(line, "//"); i >= 0 {
			// Remove comments. Note that this isn't string safe, so
			// it will truncate lines with // inside strings. Oh well.
			line = line[:i]
		}
		rule += " " + line
		rule = strings.TrimSpace(rule)
		if rule == "" {
			continue
		}
		if !strings.Contains(rule, "->") {
			continue
		}
		if ruleLineno == 0 {
			ruleLineno = lineno
		}
		if strings.HasSuffix(rule, "->") {
			continue
		}
		if unbalanced(rule) {
			continue
		}

		loc := fmt.Sprintf("%s.rules:%d", arch.name, ruleLineno)
		for _, crule := range commute(rule, arch) {
			r := Rule{rule: crule, loc: loc}
			if rawop := strings.Split(crule, " ")[0][1:]; isBlock(rawop, arch) {
				blockrules[rawop] = append(blockrules[rawop], r)
			} else {
				// Do fancier value op matching.
				match, _, _ := r.parse()
				op, oparch, _, _, _, _ := parseValue(match, arch, loc)
				opname := fmt.Sprintf("Op%s%s", oparch, op.name)
				oprules[opname] = append(oprules[opname], r)
			}
		}
		rule = ""
		ruleLineno = 0
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("scanner failed: %v\n", err)
	}
	if unbalanced(rule) {
		log.Fatalf("%s.rules:%d: unbalanced rule: %v\n", arch.name, lineno, rule)
	}

	// Order all the ops.
	var ops []string
	for op := range oprules {
		ops = append(ops, op)
	}
	sort.Strings(ops)

	// Start output buffer, write header.
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "// Code generated from gen/%s.rules; DO NOT EDIT.\n", arch.name)
	fmt.Fprintln(w, "// generated with: cd gen; go run *.go")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "package ssa")
	fmt.Fprintln(w, "import \"math\"")
	fmt.Fprintln(w, "import \"cmd/internal/obj\"")
	fmt.Fprintln(w, "import \"cmd/internal/objabi\"")
	fmt.Fprintln(w, "import \"cmd/compile/internal/types\"")
	fmt.Fprintln(w, "var _ = math.MinInt8  // in case not otherwise used")
	fmt.Fprintln(w, "var _ = obj.ANOP      // in case not otherwise used")
	fmt.Fprintln(w, "var _ = objabi.GOROOT // in case not otherwise used")
	fmt.Fprintln(w, "var _ = types.TypeMem // in case not otherwise used")
	fmt.Fprintln(w)

	const chunkSize = 10
	// Main rewrite routine is a switch on v.Op.
	fmt.Fprintf(w, "func rewriteValue%s(v *Value) bool {\n", arch.name)
	fmt.Fprintf(w, "switch v.Op {\n")
	for _, op := range ops {
		fmt.Fprintf(w, "case %s:\n", op)
		fmt.Fprint(w, "return ")
		for chunk := 0; chunk < len(oprules[op]); chunk += chunkSize {
			if chunk > 0 {
				fmt.Fprint(w, " || ")
			}
			fmt.Fprintf(w, "rewriteValue%s_%s_%d(v)", arch.name, op, chunk)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "return false\n")
	fmt.Fprintf(w, "}\n")

	// Generate a routine per op. Note that we don't make one giant routine
	// because it is too big for some compilers.
	for _, op := range ops {
		for chunk := 0; chunk < len(oprules[op]); chunk += chunkSize {
			buf := new(bytes.Buffer)
			var canFail bool
			endchunk := chunk + chunkSize
			if endchunk > len(oprules[op]) {
				endchunk = len(oprules[op])
			}
			for i, rule := range oprules[op][chunk:endchunk] {
				match, cond, result := rule.parse()
				fmt.Fprintf(buf, "// match: %s\n", match)
				fmt.Fprintf(buf, "// cond: %s\n", cond)
				fmt.Fprintf(buf, "// result: %s\n", result)

				canFail = false
				fmt.Fprintf(buf, "for {\n")
				if genMatch(buf, arch, match, rule.loc) {
					canFail = true
				}

				if cond != "" {
					fmt.Fprintf(buf, "if !(%s) {\nbreak\n}\n", cond)
					canFail = true
				}
				if !canFail && i+chunk != len(oprules[op])-1 {
					log.Fatalf("unconditional rule %s is followed by other rules", match)
				}

				genResult(buf, arch, result, rule.loc)
				if *genLog {
					fmt.Fprintf(buf, "logRule(\"%s\")\n", rule.loc)
				}
				fmt.Fprintf(buf, "return true\n")

				fmt.Fprintf(buf, "}\n")
			}
			if canFail {
				fmt.Fprintf(buf, "return false\n")
			}

			body := buf.String()
			// Do a rough match to predict whether we need b, config, fe, and/or types.
			// It's not precise--thus the blank assignments--but it's good enough
			// to avoid generating needless code and doing pointless nil checks.
			hasb := strings.Contains(body, "b.")
			hasconfig := strings.Contains(body, "config.") || strings.Contains(body, "config)")
			hasfe := strings.Contains(body, "fe.")
			hastyps := strings.Contains(body, "typ.")
			fmt.Fprintf(w, "func rewriteValue%s_%s_%d(v *Value) bool {\n", arch.name, op, chunk)
			if hasb || hasconfig || hasfe || hastyps {
				fmt.Fprintln(w, "b := v.Block")
				fmt.Fprintln(w, "_ = b")
			}
			if hasconfig {
				fmt.Fprintln(w, "config := b.Func.Config")
				fmt.Fprintln(w, "_ = config")
			}
			if hasfe {
				fmt.Fprintln(w, "fe := b.Func.fe")
				fmt.Fprintln(w, "_ = fe")
			}
			if hastyps {
				fmt.Fprintln(w, "typ := &b.Func.Config.Types")
				fmt.Fprintln(w, "_ = typ")
			}
			fmt.Fprint(w, body)
			fmt.Fprintf(w, "}\n")
		}
	}

	// Generate block rewrite function. There are only a few block types
	// so we can make this one function with a switch.
	fmt.Fprintf(w, "func rewriteBlock%s(b *Block) bool {\n", arch.name)
	fmt.Fprintln(w, "config := b.Func.Config")
	fmt.Fprintln(w, "_ = config")
	fmt.Fprintln(w, "fe := b.Func.fe")
	fmt.Fprintln(w, "_ = fe")
	fmt.Fprintln(w, "typ := &config.Types")
	fmt.Fprintln(w, "_ = typ")
	fmt.Fprintf(w, "switch b.Kind {\n")
	ops = nil
	for op := range blockrules {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		fmt.Fprintf(w, "case %s:\n", blockName(op, arch))
		for _, rule := range blockrules[op] {
			match, cond, result := rule.parse()
			fmt.Fprintf(w, "// match: %s\n", match)
			fmt.Fprintf(w, "// cond: %s\n", cond)
			fmt.Fprintf(w, "// result: %s\n", result)

			fmt.Fprintf(w, "for {\n")

			s := split(match[1 : len(match)-1]) // remove parens, then split

			// check match of control value
			if s[1] != "nil" {
				fmt.Fprintf(w, "v := b.Control\n")
				if strings.Contains(s[1], "(") {
					genMatch0(w, arch, s[1], "v", map[string]struct{}{}, false, rule.loc)
				} else {
					fmt.Fprintf(w, "_ = v\n") // in case we don't use v
					fmt.Fprintf(w, "%s := b.Control\n", s[1])
				}
			}

			if cond != "" {
				fmt.Fprintf(w, "if !(%s) {\nbreak\n}\n", cond)
			}

			// Rule matches. Generate result.
			t := split(result[1 : len(result)-1]) // remove parens, then split
			newsuccs := t[2:]

			// Check if newsuccs is the same set as succs.
			succs := s[2:]
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
			if len(m) != 0 {
				log.Fatalf("unmatched successors %v in %s", m, rule)
			}

			fmt.Fprintf(w, "b.Kind = %s\n", blockName(t[0], arch))
			if t[1] == "nil" {
				fmt.Fprintf(w, "b.SetControl(nil)\n")
			} else {
				fmt.Fprintf(w, "b.SetControl(%s)\n", genResult0(w, arch, t[1], new(int), false, false, rule.loc))
			}

			succChanged := false
			for i := 0; i < len(succs); i++ {
				if succs[i] != newsuccs[i] {
					succChanged = true
				}
			}
			if succChanged {
				if len(succs) != 2 {
					log.Fatalf("changed successors, len!=2 in %s", rule)
				}
				if succs[0] != newsuccs[1] || succs[1] != newsuccs[0] {
					log.Fatalf("can only handle swapped successors in %s", rule)
				}
				fmt.Fprintln(w, "b.swapSuccessors()")
			}

			if *genLog {
				fmt.Fprintf(w, "logRule(\"%s\")\n", rule.loc)
			}
			fmt.Fprintf(w, "return true\n")

			fmt.Fprintf(w, "}\n")
		}
	}
	fmt.Fprintf(w, "}\n")
	fmt.Fprintf(w, "return false\n")
	fmt.Fprintf(w, "}\n")

	// gofmt result
	b := w.Bytes()
	src, err := format.Source(b)
	if err != nil {
		fmt.Printf("%s\n", b)
		panic(err)
	}

	// Write to file
	err = ioutil.WriteFile("../rewrite"+arch.name+".go", src, 0666)
	if err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}
}

// genMatch returns true if the match can fail.
func genMatch(w io.Writer, arch arch, match string, loc string) bool {
	return genMatch0(w, arch, match, "v", map[string]struct{}{}, true, loc)
}

func genMatch0(w io.Writer, arch arch, match, v string, m map[string]struct{}, top bool, loc string) bool {
	if match[0] != '(' || match[len(match)-1] != ')' {
		panic("non-compound expr in genMatch0: " + match)
	}
	canFail := false

	op, oparch, typ, auxint, aux, args := parseValue(match, arch, loc)

	// check op
	if !top {
		fmt.Fprintf(w, "if %s.Op != Op%s%s {\nbreak\n}\n", v, oparch, op.name)
		canFail = true
	}

	if typ != "" {
		if !isVariable(typ) {
			// code. We must match the results of this code.
			fmt.Fprintf(w, "if %s.Type != %s {\nbreak\n}\n", v, typ)
			canFail = true
		} else {
			// variable
			if _, ok := m[typ]; ok {
				// must match previous variable
				fmt.Fprintf(w, "if %s.Type != %s {\nbreak\n}\n", v, typ)
				canFail = true
			} else {
				m[typ] = struct{}{}
				fmt.Fprintf(w, "%s := %s.Type\n", typ, v)
			}
		}
	}

	if auxint != "" {
		if !isVariable(auxint) {
			// code
			fmt.Fprintf(w, "if %s.AuxInt != %s {\nbreak\n}\n", v, auxint)
			canFail = true
		} else {
			// variable
			if _, ok := m[auxint]; ok {
				fmt.Fprintf(w, "if %s.AuxInt != %s {\nbreak\n}\n", v, auxint)
				canFail = true
			} else {
				m[auxint] = struct{}{}
				fmt.Fprintf(w, "%s := %s.AuxInt\n", auxint, v)
			}
		}
	}

	if aux != "" {

		if !isVariable(aux) {
			// code
			fmt.Fprintf(w, "if %s.Aux != %s {\nbreak\n}\n", v, aux)
			canFail = true
		} else {
			// variable
			if _, ok := m[aux]; ok {
				fmt.Fprintf(w, "if %s.Aux != %s {\nbreak\n}\n", v, aux)
				canFail = true
			} else {
				m[aux] = struct{}{}
				fmt.Fprintf(w, "%s := %s.Aux\n", aux, v)
			}
		}
	}

	if n := len(args); n > 1 {
		fmt.Fprintf(w, "_ = %s.Args[%d]\n", v, n-1) // combine some bounds checks
	}
	for i, arg := range args {
		if arg == "_" {
			continue
		}
		if !strings.Contains(arg, "(") {
			// leaf variable
			if _, ok := m[arg]; ok {
				// variable already has a definition. Check whether
				// the old definition and the new definition match.
				// For example, (add x x).  Equality is just pointer equality
				// on Values (so cse is important to do before lowering).
				fmt.Fprintf(w, "if %s != %s.Args[%d] {\nbreak\n}\n", arg, v, i)
				canFail = true
			} else {
				// remember that this variable references the given value
				m[arg] = struct{}{}
				fmt.Fprintf(w, "%s := %s.Args[%d]\n", arg, v, i)
			}
			continue
		}
		// compound sexpr
		var argname string
		colon := strings.Index(arg, ":")
		openparen := strings.Index(arg, "(")
		if colon >= 0 && openparen >= 0 && colon < openparen {
			// rule-specified name
			argname = arg[:colon]
			arg = arg[colon+1:]
		} else {
			// autogenerated name
			argname = fmt.Sprintf("%s_%d", v, i)
		}
		fmt.Fprintf(w, "%s := %s.Args[%d]\n", argname, v, i)
		if genMatch0(w, arch, arg, argname, m, false, loc) {
			canFail = true
		}
	}

	if op.argLength == -1 {
		fmt.Fprintf(w, "if len(%s.Args) != %d {\nbreak\n}\n", v, len(args))
		canFail = true
	}
	return canFail
}

func genResult(w io.Writer, arch arch, result string, loc string) {
	move := false
	if result[0] == '@' {
		// parse @block directive
		s := strings.SplitN(result[1:], " ", 2)
		fmt.Fprintf(w, "b = %s\n", s[0])
		result = s[1]
		move = true
	}
	genResult0(w, arch, result, new(int), true, move, loc)
}
func genResult0(w io.Writer, arch arch, result string, alloc *int, top, move bool, loc string) string {
	// TODO: when generating a constant result, use f.constVal to avoid
	// introducing copies just to clean them up again.
	if result[0] != '(' {
		// variable
		if top {
			// It in not safe in general to move a variable between blocks
			// (and particularly not a phi node).
			// Introduce a copy.
			fmt.Fprintf(w, "v.reset(OpCopy)\n")
			fmt.Fprintf(w, "v.Type = %s.Type\n", result)
			fmt.Fprintf(w, "v.AddArg(%s)\n", result)
		}
		return result
	}

	op, oparch, typ, auxint, aux, args := parseValue(result, arch, loc)

	// Find the type of the variable.
	typeOverride := typ != ""
	if typ == "" && op.typ != "" {
		typ = typeName(op.typ)
	}

	var v string
	if top && !move {
		v = "v"
		fmt.Fprintf(w, "v.reset(Op%s%s)\n", oparch, op.name)
		if typeOverride {
			fmt.Fprintf(w, "v.Type = %s\n", typ)
		}
	} else {
		if typ == "" {
			log.Fatalf("sub-expression %s (op=Op%s%s) must have a type", result, oparch, op.name)
		}
		v = fmt.Sprintf("v%d", *alloc)
		*alloc++
		fmt.Fprintf(w, "%s := b.NewValue0(v.Pos, Op%s%s, %s)\n", v, oparch, op.name, typ)
		if move && top {
			// Rewrite original into a copy
			fmt.Fprintf(w, "v.reset(OpCopy)\n")
			fmt.Fprintf(w, "v.AddArg(%s)\n", v)
		}
	}

	if auxint != "" {
		fmt.Fprintf(w, "%s.AuxInt = %s\n", v, auxint)
	}
	if aux != "" {
		fmt.Fprintf(w, "%s.Aux = %s\n", v, aux)
	}
	for _, arg := range args {
		x := genResult0(w, arch, arg, alloc, false, move, loc)
		fmt.Fprintf(w, "%s.AddArg(%s)\n", v, x)
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

// isBlock returns true if this op is a block opcode.
func isBlock(name string, arch arch) bool {
	for _, b := range genericBlocks {
		if b.name == name {
			return true
		}
	}
	for _, b := range arch.blocks {
		if b.name == name {
			return true
		}
	}
	return false
}

// parseValue parses a parenthesized value from a rule.
// The value can be from the match or the result side.
// It returns the op and unparsed strings for typ, auxint, and aux restrictions and for all args.
// oparch is the architecture that op is located in, or "" for generic.
func parseValue(val string, arch arch, loc string) (op opData, oparch string, typ string, auxint string, aux string, args []string) {
	val = val[1 : len(val)-1] // remove ()

	// Split val up into regions.
	// Split by spaces/tabs, except those contained in (), {}, [], or <>.
	s := split(val)

	// Extract restrictions and args.
	for _, a := range s[1:] {
		switch a[0] {
		case '<':
			typ = a[1 : len(a)-1] // remove <>
		case '[':
			auxint = a[1 : len(a)-1] // remove []
		case '{':
			aux = a[1 : len(a)-1] // remove {}
		default:
			args = append(args, a)
		}
	}

	// Resolve the op.

	// match reports whether x is a good op to select.
	// If strict is true, rule generation might succeed.
	// If strict is false, rule generation has failed,
	// but we're trying to generate a useful error.
	// Doing strict=true then strict=false allows
	// precise op matching while retaining good error messages.
	match := func(x opData, strict bool, archname string) bool {
		if x.name != s[0] {
			return false
		}
		if x.argLength != -1 && int(x.argLength) != len(args) {
			if strict {
				return false
			} else {
				log.Printf("%s: op %s (%s) should have %d args, has %d", loc, s[0], archname, x.argLength, len(args))
			}
		}
		return true
	}

	for _, x := range genericOps {
		if match(x, true, "generic") {
			op = x
			break
		}
	}
	if arch.name != "generic" {
		for _, x := range arch.ops {
			if match(x, true, arch.name) {
				if op.name != "" {
					log.Fatalf("%s: matches for op %s found in both generic and %s", loc, op.name, arch.name)
				}
				op = x
				oparch = arch.name
				break
			}
		}
	}

	if op.name == "" {
		// Failed to find the op.
		// Run through everything again with strict=false
		// to generate useful diagnosic messages before failing.
		for _, x := range genericOps {
			match(x, false, "generic")
		}
		for _, x := range arch.ops {
			match(x, false, arch.name)
		}
		log.Fatalf("%s: unknown op %s", loc, s)
	}

	// Sanity check aux, auxint.
	if auxint != "" {
		switch op.aux {
		case "Bool", "Int8", "Int16", "Int32", "Int64", "Int128", "Float32", "Float64", "SymOff", "SymValAndOff", "SymInt32", "TypSize":
		default:
			log.Fatalf("%s: op %s %s can't have auxint", loc, op.name, op.aux)
		}
	}
	if aux != "" {
		switch op.aux {
		case "String", "Sym", "SymOff", "SymValAndOff", "SymInt32", "Typ", "TypSize":
		default:
			log.Fatalf("%s: op %s %s can't have aux", loc, op.name, op.aux)
		}
	}

	return
}

func blockName(name string, arch arch) string {
	for _, b := range genericBlocks {
		if b.name == name {
			return "Block" + name
		}
	}
	return "Block" + arch.name + name
}

// typeName returns the string to use to generate a type.
func typeName(typ string) string {
	if typ[0] == '(' {
		ts := strings.Split(typ[1:len(typ)-1], ",")
		if len(ts) != 2 {
			panic("Tuple expect 2 arguments")
		}
		return "types.NewTuple(" + typeName(ts[0]) + ", " + typeName(ts[1]) + ")"
	}
	switch typ {
	case "Flags", "Mem", "Void", "Int128":
		return "types.Type" + typ
	default:
		return "typ." + typ
	}
}

// unbalanced returns true if there aren't the same number of ( and ) in the string.
func unbalanced(s string) bool {
	var left, right int
	for _, c := range s {
		if c == '(' {
			left++
		}
		if c == ')' {
			right++
		}
	}
	return left != right
}

// isVariable reports whether s is a single Go alphanumeric identifier.
func isVariable(s string) bool {
	b, err := regexp.MatchString("^[A-Za-z_][A-Za-z_0-9]*$", s)
	if err != nil {
		panic("bad variable regexp")
	}
	return b
}

// commute returns all equivalent rules to r after applying all possible
// argument swaps to the commutable ops in r.
// Potentially exponential, be careful.
func commute(r string, arch arch) []string {
	match, cond, result := Rule{rule: r}.parse()
	a := commute1(match, varCount(match), arch)
	for i, m := range a {
		if cond != "" {
			m += " && " + cond
		}
		m += " -> " + result
		a[i] = m
	}
	if len(a) == 1 && normalizeWhitespace(r) != normalizeWhitespace(a[0]) {
		fmt.Println(normalizeWhitespace(r))
		fmt.Println(normalizeWhitespace(a[0]))
		panic("commute() is not the identity for noncommuting rule")
	}
	if false && len(a) > 1 {
		fmt.Println(r)
		for _, x := range a {
			fmt.Println("  " + x)
		}
	}
	return a
}

func commute1(m string, cnt map[string]int, arch arch) []string {
	if m[0] == '<' || m[0] == '[' || m[0] == '{' || isVariable(m) {
		return []string{m}
	}
	// Split up input.
	var prefix string
	colon := strings.Index(m, ":")
	if colon >= 0 && isVariable(m[:colon]) {
		prefix = m[:colon+1]
		m = m[colon+1:]
	}
	if m[0] != '(' || m[len(m)-1] != ')' {
		panic("non-compound expr in commute1: " + m)
	}
	s := split(m[1 : len(m)-1])
	op := s[0]

	// Figure out if the op is commutative or not.
	commutative := false
	for _, x := range genericOps {
		if op == x.name {
			if x.commutative {
				commutative = true
			}
			break
		}
	}
	if arch.name != "generic" {
		for _, x := range arch.ops {
			if op == x.name {
				if x.commutative {
					commutative = true
				}
				break
			}
		}
	}
	var idx0, idx1 int
	if commutative {
		// Find indexes of two args we can swap.
		for i, arg := range s {
			if i == 0 || arg[0] == '<' || arg[0] == '[' || arg[0] == '{' {
				continue
			}
			if idx0 == 0 {
				idx0 = i
				continue
			}
			if idx1 == 0 {
				idx1 = i
				break
			}
		}
		if idx1 == 0 {
			panic("couldn't find first two args of commutative op " + s[0])
		}
		if cnt[s[idx0]] == 1 && cnt[s[idx1]] == 1 || s[idx0] == s[idx1] && cnt[s[idx0]] == 2 {
			// When we have (Add x y) with no ther uses of x and y in the matching rule,
			// then we can skip the commutative match (Add y x).
			commutative = false
		}
	}

	// Recursively commute arguments.
	a := make([][]string, len(s))
	for i, arg := range s {
		a[i] = commute1(arg, cnt, arch)
	}

	// Choose all possibilities from all args.
	r := crossProduct(a)

	// If commutative, do that again with its two args reversed.
	if commutative {
		a[idx0], a[idx1] = a[idx1], a[idx0]
		r = append(r, crossProduct(a)...)
	}

	// Construct result.
	for i, x := range r {
		r[i] = prefix + "(" + x + ")"
	}
	return r
}

// varCount returns a map which counts the number of occurrences of
// Value variables in m.
func varCount(m string) map[string]int {
	cnt := map[string]int{}
	varCount1(m, cnt)
	return cnt
}
func varCount1(m string, cnt map[string]int) {
	if m[0] == '<' || m[0] == '[' || m[0] == '{' {
		return
	}
	if isVariable(m) {
		cnt[m]++
		return
	}
	// Split up input.
	colon := strings.Index(m, ":")
	if colon >= 0 && isVariable(m[:colon]) {
		cnt[m[:colon]]++
		m = m[colon+1:]
	}
	if m[0] != '(' || m[len(m)-1] != ')' {
		panic("non-compound expr in commute1: " + m)
	}
	s := split(m[1 : len(m)-1])
	for _, arg := range s[1:] {
		varCount1(arg, cnt)
	}
}

// crossProduct returns all possible values
// x[0][i] + " " + x[1][j] + " " + ... + " " + x[len(x)-1][k]
// for all valid values of i, j, ..., k.
func crossProduct(x [][]string) []string {
	if len(x) == 1 {
		return x[0]
	}
	var r []string
	for _, tail := range crossProduct(x[1:]) {
		for _, first := range x[0] {
			r = append(r, first+" "+tail)
		}
	}
	return r
}

// normalizeWhitespace replaces 2+ whitespace sequences with a single space.
func normalizeWhitespace(x string) string {
	x = strings.Join(strings.Fields(x), " ")
	x = strings.Replace(x, "( ", "(", -1)
	x = strings.Replace(x, " )", ")", -1)
	return x
}
