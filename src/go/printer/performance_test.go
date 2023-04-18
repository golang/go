// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a simple printer performance benchmark:
// go test -bench=BenchmarkPrint

package printer

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"testing"
)

var (
	fileNode *ast.File
	fileSize int64

	declNode ast.Decl
	declSize int64
)

func testprint(out io.Writer, node ast.Node) {
	if err := (&Config{TabIndent | UseSpaces | normalizeNumbers, 8, 0}).Fprint(out, fset, node); err != nil {
		log.Fatalf("print error: %s", err)
	}
}

// cannot initialize in init because (printer) Fprint launches goroutines.
func initialize() {
	const filename = "testdata/parser.go"

	src, err := os.ReadFile(filename)
	if err != nil {
		log.Fatalf("%s", err)
	}

	file, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		log.Fatalf("%s", err)
	}

	var buf bytes.Buffer
	testprint(&buf, file)
	if !bytes.Equal(buf.Bytes(), src) {
		log.Fatalf("print error: %s not idempotent", filename)
	}

	fileNode = file
	fileSize = int64(len(src))

	for _, decl := range file.Decls {
		// The first global variable, which is pretty short:
		//
		//	var unresolved = new(ast.Object)
		if decl, ok := decl.(*ast.GenDecl); ok && decl.Tok == token.VAR {
			declNode = decl
			declSize = int64(fset.Position(decl.End()).Offset - fset.Position(decl.Pos()).Offset)
			break
		}

	}
}

func BenchmarkPrintFile(b *testing.B) {
	if fileNode == nil {
		initialize()
	}
	b.ReportAllocs()
	b.SetBytes(fileSize)
	for i := 0; i < b.N; i++ {
		testprint(io.Discard, fileNode)
	}
}

func BenchmarkPrintDecl(b *testing.B) {
	if declNode == nil {
		initialize()
	}
	b.ReportAllocs()
	b.SetBytes(declSize)
	for i := 0; i < b.N; i++ {
		testprint(io.Discard, declNode)
	}
}
