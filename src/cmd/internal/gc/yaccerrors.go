// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program implements the core idea from
//
//	Clinton L. Jeffery, Generating LR syntax error messages from examples,
//	ACM TOPLAS 25(5) (September 2003).  http://doi.acm.org/10.1145/937563.937566
//
// It reads Bison's summary of a grammar followed by a file
// like go.errors, replacing lines beginning with % by the
// yystate and yychar that will be active when an error happens
// while parsing that line.
//
// Unlike the system described in the paper, the lines in go.errors
// give grammar symbol name lists, not actual program fragments.
// This is a little less programmer-friendly but doesn't require being
// able to run the text through lex.c.

package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

func xatoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		log.Fatal(err)
	}
	return n
}

func trimParen(s string) string {
	s = strings.TrimPrefix(s, "(")
	s = strings.TrimSuffix(s, ")")
	return s
}

type action struct {
	token string
	n     int
}

var shift = map[int][]action{}
var reduce = map[int][]action{}

type rule struct {
	lhs  string
	size int
}

var rules = map[int]rule{}

func readYaccOutput() {
	r, err := os.Open("y.output")
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	var state int

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		nf := len(f)

		if nf >= 4 && f[1] == "terminals," && f[3] == "nonterminals" {
			// We're done.
			break
		}

		if nf >= 2 && f[0] == "state" {
			state = xatoi(f[1])
			continue
		}
		if nf >= 3 && (f[1] == "shift" || f[1] == "goto") {
			shift[state] = append(shift[state], action{f[0], xatoi(f[2])})
			continue
		}
		if nf >= 3 && f[1] == "reduce" {
			reduce[state] = append(reduce[state], action{f[0], xatoi(f[2])})
			continue
		}
		if nf >= 3 && strings.HasSuffix(f[0], ":") && strings.HasPrefix(f[nf-1], "(") && strings.HasSuffix(f[nf-1], ")") {
			n := xatoi(trimParen(f[nf-1]))

			size := nf - 2
			if size == 1 && f[1] == "." {
				size = 0
			}

			rules[n] = rule{strings.TrimSuffix(f[0], ":"), size}
			continue
		}
	}
}

func runMachine(w io.Writer, s string) {
	f := strings.Fields(s)

	// Run it through the LR machine and print the induced "yystate, yychar,"
	// at the point where the error happens.

	var stack []int
	state := 0
	i := 1
	tok := ""

Loop:
	if tok == "" && i < len(f) {
		tok = f[i]
		i++
	}

	for _, a := range shift[state] {
		if a.token == tok {
			if false {
				fmt.Println("SHIFT ", tok, " ", state, " -> ", a)
			}
			stack = append(stack, state)
			state = a.n
			tok = ""
			goto Loop
		}
	}

	for _, a := range reduce[state] {
		if a.token == tok || a.token == "." {
			stack = append(stack, state)
			rule, ok := rules[a.n]
			if !ok {
				log.Fatal("missing rule")
			}
			stack = stack[:len(stack)-rule.size]
			state = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if tok != "" {
				i--
			}
			tok = rule.lhs
			if false {
				fmt.Println("REDUCE ", stack, " ", state, " ", tok, " rule ", rule)
			}
			goto Loop
		}
	}

	// No shift or reduce applied - found the error.
	fmt.Fprintf(w, "\t{%d, %s,\n", state, tok)
}

func processGoErrors() {
	r, err := os.Open("go.errors")
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	w, err := os.Create("yymsg.go")
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()

	fmt.Fprintf(w, "// DO NOT EDIT - generated with go generate\n\n")

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		s := scanner.Text()

		// Treat % as first field on line as introducing a pattern (token sequence).
		if strings.HasPrefix(strings.TrimSpace(s), "%") {
			runMachine(w, s)
			continue
		}

		fmt.Fprintln(w, s)
	}
}

func main() {
	readYaccOutput()
	processGoErrors()
}
