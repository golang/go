// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

const src = `package main

import "log"

func f1() {
	live()
	return
	dead()
}

func f2() {
	for {
		live()
	}
	dead()
}

func f3() {
	if true { // even known values are ignored
		return
	}
	for true { // even known values are ignored
		live()
	}
	for {
		live()
	}
	dead()
}

func f4(x int) {
	switch x {
	case 1:
		live()
		fallthrough
	case 2:
		live()
		log.Fatal()
	default:
		panic("oops")
	}
	dead()
}

func f4(ch chan int) {
	select {
	case <-ch:
		live()
		return
	default:
		live()
		panic("oops")
	}
	dead()
}

func f5(unknown bool) {
	for {
		if unknown {
			break
		}
		continue
		dead()
	}
	live()
}

func f6(unknown bool) {
outer:
	for {
		for {
			break outer
			dead()
		}
		dead()
	}
	live()
}

func f7() {
	for {
		break nosuchlabel
		dead()
	}
	dead()
}

func f8() {
	select{}
	dead()
}

func f9(ch chan int) {
	select {
	case <-ch:
		return
	}
	dead()
}

func f10(ch chan int) {
	select {
	case <-ch:
		return
		dead()
	default:
	}
	live()
}

func f11() {
	goto; // mustn't crash
	dead()
}

`

func TestDeadCode(t *testing.T) {
	// We'll use dead code detection to verify the CFG.

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "dummy.go", src, parser.Mode(0))
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range f.Decls {
		if decl, ok := decl.(*ast.FuncDecl); ok {
			g := New(decl.Body, mayReturn)

			// Print statements in unreachable blocks
			// (in order determined by builder).
			var buf bytes.Buffer
			for _, b := range g.Blocks {
				if !b.Live {
					for _, n := range b.Nodes {
						fmt.Fprintf(&buf, "\t%s\n", formatNode(fset, n))
					}
				}
			}

			// Check that the result contains "dead" at least once but not "live".
			if !bytes.Contains(buf.Bytes(), []byte("dead")) ||
				bytes.Contains(buf.Bytes(), []byte("live")) {
				t.Errorf("unexpected dead statements in function %s:\n%s",
					decl.Name.Name,
					&buf)
				t.Logf("control flow graph:\n%s", g.Format(fset))
			}
		}
	}
}

// A trivial mayReturn predicate that looks only at syntax, not types.
func mayReturn(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name != "panic"
	case *ast.SelectorExpr:
		return fun.Sel.Name != "Fatal"
	}
	return true
}
