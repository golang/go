// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// runGenTest runs a test-generator, then runs the generated test.
// Generated test can either fail in compilation or execution.
// The environment variable parameter(s) is passed to the run
// of the generated test.
func runGenTest(t *testing.T, filename, tmpname string, ev ...string) {
	testenv.MustHaveGoRun(t)
	gotool := testenv.GoToolPath(t)
	var stdout, stderr bytes.Buffer
	cmd := testenv.Command(t, gotool, "run", filepath.Join("testdata", filename))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	// Write stdout into a temporary file
	rungo := filepath.Join(t.TempDir(), "run.go")
	ok := os.WriteFile(rungo, stdout.Bytes(), 0600)
	if ok != nil {
		t.Fatalf("Failed to create temporary file %s", rungo)
	}

	stdout.Reset()
	stderr.Reset()
	cmd = testenv.Command(t, gotool, "run", "-gcflags=-d=ssa/check/on", rungo)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = append(cmd.Env, ev...)
	err := cmd.Run()
	if err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	if s := stderr.String(); s != "" {
		t.Errorf("Stderr = %s\nWant empty", s)
	}
	if s := stdout.String(); s != "" {
		t.Errorf("Stdout = %s\nWant empty", s)
	}
}

func TestGenFlowGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("not run in short mode.")
	}
	runGenTest(t, "flowgraph_generator1.go", "ssa_fg_tmp1")
}

// TestCode runs all the tests in the testdata directory as subtests.
// These tests are special because we want to run them with different
// compiler flags set (and thus they can't just be _test.go files in
// this directory).
func TestCode(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	gotool := testenv.GoToolPath(t)

	// Make a temporary directory to work in.
	tmpdir := t.TempDir()

	// Find all the test functions (and the files containing them).
	var srcs []string // files containing Test functions
	type test struct {
		name      string // TestFoo
		usesFloat bool   // might use float operations
	}
	var tests []test
	files, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("can't read testdata directory: %v", err)
	}
	for _, f := range files {
		if !strings.HasSuffix(f.Name(), "_test.go") {
			continue
		}
		text, err := os.ReadFile(filepath.Join("testdata", f.Name()))
		if err != nil {
			t.Fatalf("can't read testdata/%s: %v", f.Name(), err)
		}
		fset := token.NewFileSet()
		code, err := parser.ParseFile(fset, f.Name(), text, 0)
		if err != nil {
			t.Fatalf("can't parse testdata/%s: %v", f.Name(), err)
		}
		srcs = append(srcs, filepath.Join("testdata", f.Name()))
		foundTest := false
		for _, d := range code.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			if !strings.HasPrefix(fd.Name.Name, "Test") {
				continue
			}
			if fd.Recv != nil {
				continue
			}
			if fd.Type.Results != nil {
				continue
			}
			if len(fd.Type.Params.List) != 1 {
				continue
			}
			p := fd.Type.Params.List[0]
			if len(p.Names) != 1 {
				continue
			}
			s, ok := p.Type.(*ast.StarExpr)
			if !ok {
				continue
			}
			sel, ok := s.X.(*ast.SelectorExpr)
			if !ok {
				continue
			}
			base, ok := sel.X.(*ast.Ident)
			if !ok {
				continue
			}
			if base.Name != "testing" {
				continue
			}
			if sel.Sel.Name != "T" {
				continue
			}
			// Found a testing function.
			tests = append(tests, test{name: fd.Name.Name, usesFloat: bytes.Contains(text, []byte("float"))})
			foundTest = true
		}
		if !foundTest {
			t.Fatalf("test file testdata/%s has no tests in it", f.Name())
		}
	}

	flags := []string{""}
	if runtime.GOARCH == "arm" || runtime.GOARCH == "mips" || runtime.GOARCH == "mips64" || runtime.GOARCH == "386" {
		flags = append(flags, ",softfloat")
	}
	for _, flag := range flags {
		args := []string{"test", "-c", "-gcflags=-d=ssa/check/on" + flag, "-o", filepath.Join(tmpdir, "code.test")}
		args = append(args, srcs...)
		out, err := testenv.Command(t, gotool, args...).CombinedOutput()
		if err != nil || len(out) != 0 {
			t.Fatalf("Build failed: %v\n%s\n", err, out)
		}

		// Now we have a test binary. Run it with all the tests as subtests of this one.
		for _, test := range tests {

			if flag == ",softfloat" && !test.usesFloat {
				// No point in running the soft float version if the test doesn't use floats.
				continue
			}
			t.Run(fmt.Sprintf("%s%s", test.name[4:], flag), func(t *testing.T) {
				out, err := testenv.Command(t, filepath.Join(tmpdir, "code.test"), "-test.run=^"+test.name+"$").CombinedOutput()
				if err != nil || string(out) != "PASS\n" {
					t.Errorf("Failed:\n%s\n", out)
				}
			})
		}
	}
}
