// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

const testfile = "format_test.go"

func diff(t *testing.T, dst, src []byte) {
	line := 1
	offs := 0 // line offset
	for i := 0; i < len(dst) && i < len(src); i++ {
		d := dst[i]
		s := src[i]
		if d != s {
			t.Errorf("dst:%d: %s\n", line, dst[offs:i+1])
			t.Errorf("src:%d: %s\n", line, src[offs:i+1])
			return
		}
		if s == '\n' {
			line++
			offs = i + 1
		}
	}
	if len(dst) != len(src) {
		t.Errorf("len(dst) = %d, len(src) = %d\nsrc = %q", len(dst), len(src), src)
	}
}

func TestNode(t *testing.T) {
	src, err := os.ReadFile(testfile)
	if err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, testfile, src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer

	if err = Node(&buf, fset, file); err != nil {
		t.Fatal("Node failed:", err)
	}

	diff(t, buf.Bytes(), src)
}

// Node is documented to not modify the AST.
// Test that it is so even when numbers are normalized.
func TestNodeNoModify(t *testing.T) {
	const (
		src    = "package p\n\nconst _ = 0000000123i\n"
		golden = "package p\n\nconst _ = 123i\n"
	)

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	// Capture original address and value of a BasicLit node
	// which will undergo formatting changes during printing.
	wantLit := file.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec).Values[0].(*ast.BasicLit)
	wantVal := wantLit.Value

	var buf bytes.Buffer
	if err = Node(&buf, fset, file); err != nil {
		t.Fatal("Node failed:", err)
	}
	diff(t, buf.Bytes(), []byte(golden))

	// Check if anything changed after Node returned.
	gotLit := file.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec).Values[0].(*ast.BasicLit)
	gotVal := gotLit.Value

	if gotLit != wantLit {
		t.Errorf("got *ast.BasicLit address %p, want %p", gotLit, wantLit)
	}
	if gotVal != wantVal {
		t.Errorf("got *ast.BasicLit value %q, want %q", gotVal, wantVal)
	}
}

func TestSource(t *testing.T) {
	src, err := os.ReadFile(testfile)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Source(src)
	if err != nil {
		t.Fatal("Source failed:", err)
	}

	diff(t, res, src)
}

// Test cases that are expected to fail are marked by the prefix "ERROR".
// The formatted result must look the same as the input for successful tests.
var tests = []string{
	// declaration lists
	`import "go/format"`,
	"var x int",
	"var x int\n\ntype T struct{}",

	// statement lists
	"x := 0",
	"f(a, b, c)\nvar x int = f(1, 2, 3)",

	// indentation, leading and trailing space
	"\tx := 0\n\tgo f()",
	"\tx := 0\n\tgo f()\n\n\n",
	"\n\t\t\n\n\tx := 0\n\tgo f()\n\n\n",
	"\n\t\t\n\n\t\t\tx := 0\n\t\t\tgo f()\n\n\n",
	"\n\t\t\n\n\t\t\tx := 0\n\t\t\tconst s = `\nfoo\n`\n\n\n",     // no indentation added inside raw strings
	"\n\t\t\n\n\t\t\tx := 0\n\t\t\tconst s = `\n\t\tfoo\n`\n\n\n", // no indentation removed inside raw strings

	// comments
	"/* Comment */",
	"\t/* Comment */ ",
	"\n/* Comment */ ",
	"i := 5 /* Comment */",         // issue #5551
	"\ta()\n//line :1",             // issue #11276
	"\t//xxx\n\ta()\n//line :2",    // issue #11276
	"\ta() //line :1\n\tb()\n",     // issue #11276
	"x := 0\n//line :1\n//line :2", // issue #11276

	// whitespace
	"",     // issue #11275
	" ",    // issue #11275
	"\t",   // issue #11275
	"\t\t", // issue #11275
	"\n",   // issue #11275
	"\n\n", // issue #11275
	"\t\n", // issue #11275

	// erroneous programs
	"ERROR1 + 2 +",
	"ERRORx :=  0",
}

func String(s string) (string, error) {
	res, err := Source([]byte(s))
	if err != nil {
		return "", err
	}
	return string(res), nil
}

func TestPartial(t *testing.T) {
	for _, src := range tests {
		if strings.HasPrefix(src, "ERROR") {
			// test expected to fail
			src = src[5:] // remove ERROR prefix
			res, err := String(src)
			if err == nil && res == src {
				t.Errorf("formatting succeeded but was expected to fail:\n%q", src)
			}
		} else {
			// test expected to succeed
			res, err := String(src)
			if err != nil {
				t.Errorf("formatting failed (%s):\n%q", err, src)
			} else if res != src {
				t.Errorf("formatting incorrect:\nsource: %q\nresult: %q", src, res)
			}
		}
	}
}
