// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"go/ast"
	"go/parser"
	"io/ioutil"
	"strings"

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
func testProg(path string) (*ssa.Program, []string, error) {
	content, err := ioutil.ReadFile(path)
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

	prog := ssautil.CreateProgram(iprog, 0)
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
