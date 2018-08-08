// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssautil_test

import (
	"bytes"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa/ssautil"
)

const hello = `package main

import "fmt"

func main() {
	fmt.Println("Hello, world")
}
`

func TestBuildPackage(t *testing.T) {
	// There is a more substantial test of BuildPackage and the
	// SSA program it builds in ../ssa/builder_test.go.

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "hello.go", hello, 0)
	if err != nil {
		t.Fatal(err)
	}

	pkg := types.NewPackage("hello", "")
	ssapkg, _, err := ssautil.BuildPackage(&types.Config{Importer: importer.Default()}, fset, pkg, []*ast.File{f}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if pkg.Name() != "main" {
		t.Errorf("pkg.Name() = %s, want main", pkg.Name())
	}
	if ssapkg.Func("main") == nil {
		ssapkg.WriteTo(os.Stderr)
		t.Errorf("ssapkg has no main function")
	}
}

func TestPackages(t *testing.T) {
	cfg := &packages.Config{Mode: packages.LoadSyntax}
	initial, err := packages.Load(cfg, "bytes")
	if err != nil {
		t.Fatal(err)
	}

	prog, pkgs := ssautil.Packages(initial, 0)
	bytesNewBuffer := pkgs[0].Func("NewBuffer")
	bytesNewBuffer.Pkg.Build()

	// We'll dump the SSA of bytes.NewBuffer because it is small and stable.
	out := new(bytes.Buffer)
	bytesNewBuffer.WriteTo(out)

	// For determinism, sanitize the location.
	location := prog.Fset.Position(bytesNewBuffer.Pos()).String()
	got := strings.Replace(out.String(), location, "$GOROOT/src/bytes/buffer.go:1", -1)

	want := `
# Name: bytes.NewBuffer
# Package: bytes
# Location: $GOROOT/src/bytes/buffer.go:1
func NewBuffer(buf []byte) *Buffer:
0:                                                                entry P:0 S:0
	t0 = new Buffer (complit)                                       *Buffer
	t1 = &t0.buf [#0]                                               *[]byte
	*t1 = buf
	return t0

`[1:]
	if got != want {
		t.Errorf("bytes.NewBuffer SSA = <<%s>>, want <<%s>>", got, want)
	}
}

func TestBuildPackage_MissingImport(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "bad.go", `package bad; import "missing"`, 0)
	if err != nil {
		t.Fatal(err)
	}

	pkg := types.NewPackage("bad", "")
	ssapkg, _, err := ssautil.BuildPackage(new(types.Config), fset, pkg, []*ast.File{f}, 0)
	if err == nil || ssapkg != nil {
		t.Fatal("BuildPackage succeeded unexpectedly")
	}
}
