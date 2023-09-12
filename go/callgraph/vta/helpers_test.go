// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"os"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa/ssautil"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
)

// want extracts the contents of the first comment
// section starting with "WANT:\n". The returned
// content is split into lines without // prefix.
func want(f *ast.File) []string {
	for _, c := range f.Comments {
		text := strings.TrimSpace(c.Text())
		if t := strings.TrimPrefix(text, "WANT:\n"); t != text {
			return strings.Split(t, "\n")
		}
	}
	return nil
}

// testProg returns an ssa representation of a program at
// `path`, assumed to define package "testdata," and the
// test want result as list of strings.
func testProg(path string, mode ssa.BuilderMode) (*ssa.Program, []string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}

	conf := loader.Config{
		ParserMode: parser.ParseComments,
	}

	f, err := conf.ParseFile(path, content)
	if err != nil {
		return nil, nil, err
	}

	conf.CreateFromFiles("testdata", f)
	iprog, err := conf.Load()
	if err != nil {
		return nil, nil, err
	}

	prog := ssautil.CreateProgram(iprog, mode)
	// Set debug mode to exercise DebugRef instructions.
	prog.Package(iprog.Created[0].Pkg).SetDebugMode(true)
	prog.Build()
	return prog, want(f), nil
}

func firstRegInstr(f *ssa.Function) ssa.Value {
	for _, b := range f.Blocks {
		for _, i := range b.Instrs {
			if v, ok := i.(ssa.Value); ok {
				return v
			}
		}
	}
	return nil
}

// funcName returns a name of the function `f`
// prefixed with the name of the receiver type.
func funcName(f *ssa.Function) string {
	recv := f.Signature.Recv()
	if recv == nil {
		return f.Name()
	}
	tp := recv.Type().String()
	return tp[strings.LastIndex(tp, ".")+1:] + "." + f.Name()
}

// callGraphStr stringifes `g` into a list of strings where
// each entry is of the form
//
//	f: cs1 -> f1, f2, ...; ...; csw -> fx, fy, ...
//
// f is a function, cs1, ..., csw are call sites in f, and
// f1, f2, ..., fx, fy, ... are the resolved callees.
func callGraphStr(g *callgraph.Graph) []string {
	var gs []string
	for f, n := range g.Nodes {
		c := make(map[string][]string)
		for _, edge := range n.Out {
			cs := edge.Site.String()
			c[cs] = append(c[cs], funcName(edge.Callee.Func))
		}

		var cs []string
		for site, fs := range c {
			sort.Strings(fs)
			entry := fmt.Sprintf("%v -> %v", site, strings.Join(fs, ", "))
			cs = append(cs, entry)
		}

		sort.Strings(cs)
		entry := fmt.Sprintf("%v: %v", funcName(f), strings.Join(cs, "; "))
		gs = append(gs, entry)
	}
	return gs
}

// Logs the functions of prog to t.
func logFns(t testing.TB, prog *ssa.Program) {
	for fn := range ssautil.AllFunctions(prog) {
		var buf bytes.Buffer
		fn.WriteTo(&buf)
		t.Log(buf.String())
	}
}
