// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"os/exec"
	"testing"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
)

const offendingSource = `package jsonmarshal

type draw struct{}

func (d *draw) str() string {
	switch 0 {
	case 0:
		return "a"
	}
	return "b"
}

func pick[T comparable](d *draw, x T, zero T) T {
	if true {
		return zero
	}
	return x
}

func TestMarshalMatchesEncodingJSONSweep(t *testing.T) {
	for range 10 {}
}
`

func TestLogosScanChild(t *testing.T) {
	if os.Getenv("GO_LOGOS_SCAN_HELPER") != "1" {
		return
	}
	fset := token.NewFileSet()
	file := scanLogosTestFile(fset, "json_marshal_test.lgo", []byte(offendingSource))
	if file == nil {
		fmt.Fprintf(os.Stderr, "scanLogosTestFile returned nil\n")
		os.Exit(2)
	}

	var names []string
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			names = append(names, fn.Name.Name)
			if fn.Body == nil || !fn.Body.Lbrace.IsValid() || !fn.Body.Rbrace.IsValid() {
				fmt.Fprintf(os.Stderr, "func %s has invalid body bounds: %+v\n", fn.Name.Name, fn.Body)
				os.Exit(3)
			}
		}
	}

	want := []string{"pick", "TestMarshalMatchesEncodingJSONSweep"}
	if len(names) != len(want) {
		fmt.Fprintf(os.Stderr, "got decls %v, want %v\n", names, want)
		os.Exit(4)
	}
	for i := range want {
		if names[i] != want[i] {
			fmt.Fprintf(os.Stderr, "decl[%d] = %q, want %q\n", i, names[i], want[i])
			os.Exit(5)
		}
	}

	for _, bad := range []string{
		"package p\nfunc foo() {\n",
		"package p\nfunc foo() { if true {\n",
		"package p\nfunc foo() {}\n",
		"package p\nfunc foo() { }\n}\nfunc bar() {}\n",
	} {
		if scanLogosTestFile(fset, "bad_test.lgo", []byte(bad)) == nil {
			fmt.Fprintf(os.Stderr, "scanLogosTestFile on malformed returned nil\n")
			os.Exit(6)
		}
	}
	os.Exit(0)
}

func TestLogosScanProgressBounded(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestLogosScanChild$")
	cmd.Env = append(os.Environ(), "GO_LOGOS_SCAN_HELPER=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			t.Fatalf("scanLogosTestFile livelocked: child process timed out and was killed: %v", err)
		}
		t.Fatalf("child helper failed: %v\n%s", err, out)
	}
}
func TestLogosLintMarkSkipsNonMainModules(t *testing.T) {
	old := cfg.Logos
	cfg.Logos = true
	t.Cleanup(func() { cfg.Logos = old })
	std := &Package{Standard: true}
	if std.Internal.LogosLint != nil {
		t.Errorf("standard package LogosLint = %q, want nil", std.Internal.LogosLint)
	}
	dep := &Package{Module: &modinfo.ModulePublic{Path: "example.com/dep", Main: false}}
	logosLintMark(nil, dep)
	if dep.Internal.LogosLint != nil {
		t.Errorf("third-party module LogosLint = %q, want nil", dep.Internal.LogosLint)
	}
	bare := &Package{}
	logosLintMark(nil, bare)
	if bare.Internal.LogosLint != nil {
		t.Errorf("module-less package LogosLint = %q, want nil", bare.Internal.LogosLint)
	}
	main := []byte("package main\n")
	testmain := &Package{Module: &modinfo.ModulePublic{Path: "example.com/m", Main: true}}
	testmain.Internal.TestmainGo = &main
	logosLintMark(nil, testmain)
	if testmain.Internal.LogosLint != nil {
		t.Errorf("testmain package LogosLint = %q, want nil", testmain.Internal.LogosLint)
	}
}
