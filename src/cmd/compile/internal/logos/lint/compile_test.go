// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// compileResult is the outcome of one run of the compiler tool.
type compileResult struct {
	output   string
	exitCode int
	objectOK bool
}

// compileFile runs go tool compile on one source file named name with the
// given extra flags and the default lint policy, and reports whether the
// object file exists afterwards.
func compileFile(t *testing.T, name, src string, flags ...string) compileResult {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	srcPath := filepath.Join(dir, name)
	policy := filepath.Join(dir, "policy.json")
	object := filepath.Join(dir, "out.a")
	for path, data := range map[string]string{srcPath: src, policy: `{"version":1}`} {
		if err := os.WriteFile(path, []byte(data), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	args := []string{"tool", "compile", "-p", "p", "-logos.cfg=" + policy, "-o", object}
	args = append(append(args, flags...), srcPath)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	out, err := cmd.CombinedOutput()
	res := compileResult{output: string(out)}
	if ee, ok := err.(*exec.ExitError); ok {
		res.exitCode = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	_, statErr := os.Stat(object)
	res.objectOK = statErr == nil
	return res
}

const compileSrc = `package p

func f() error { return nil }

func g() {
	f()
}
`

// TestLintOnlyStopsBeforeBackend checks the -logos.lintonly contract: a Go
// package lints, the compile writes no object file, and a finding that is an
// error fails the compile.
func TestLintOnlyStopsBeforeBackend(t *testing.T) {
	res := compileFile(t, "x.go", compileSrc, "-logos.lintonly")
	if res.exitCode != 2 || !strings.Contains(res.output, "[slop::discarded_error]") {
		t.Errorf("lintonly Go compile: exit %d, output %q", res.exitCode, res.output)
	}
	if res.objectOK {
		t.Error("lintonly compile wrote an object file")
	}
	const clean = "package p\n\nfunc f() int { return 1 }\n"
	res = compileFile(t, "x.go", clean, "-logos.lintonly")
	if res.exitCode != 0 || res.output != "" || res.objectOK {
		t.Errorf("lintonly clean compile: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
	}
	res = compileFile(t, "x.go", clean)
	if res.exitCode != 0 || !res.objectOK {
		t.Errorf("ordinary compile: exit %d, object %v, output %q", res.exitCode, res.objectOK, res.output)
	}
}

// TestGoPackageNeedsLintOnly checks that an ordinary compile of a Go package
// runs no lint even when a policy is given: only -logos.lintonly lints Go.
func TestGoPackageNeedsLintOnly(t *testing.T) {
	res := compileFile(t, "x.go", compileSrc)
	if res.exitCode != 0 || res.output != "" || !res.objectOK {
		t.Errorf("ordinary Go compile linted: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
	}
}

// TestLgoPackageLintsEveryCompile checks that a loGo/s package lints on an
// ordinary compile, and writes no object on failure.
func TestLgoPackageLintsEveryCompile(t *testing.T) {
	res := compileFile(t, "x.lgo", compileSrc)
	if res.exitCode != 2 || !strings.Contains(res.output, "x.lgo:6:2: [slop::discarded_error]") || res.objectOK {
		t.Errorf("lgo compile: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
	}
}

// TestCgoOutputIsNotLinted checks that compiler-generated cgo files carry no
// finding.
func TestCgoOutputIsNotLinted(t *testing.T) {
	res := compileFile(t, "x.cgo1.go", compileSrc, "-logos.lintonly")
	if res.exitCode != 0 || res.output != "" {
		t.Errorf("cgo output linted: exit %d, output %q", res.exitCode, res.output)
	}
}

// TestBadPolicyFailsTheCompile checks that an unreadable -logos.cfg is a
// build error and not a silent no-lint compile.
func TestBadPolicyFailsTheCompile(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "x.lgo")
	policy := filepath.Join(dir, "policy.json")
	for path, data := range map[string]string{src: "package p\n", policy: `{"version":7}`} {
		if err := os.WriteFile(path, []byte(data), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	out, err := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-p", "p", "-logos.cfg="+policy, "-o", filepath.Join(dir, "o.a"), src).CombinedOutput()
	if err == nil || !strings.Contains(string(out), "unsupported version 7") {
		t.Errorf("bad policy: err %v, output %q", err, out)
	}
}

// TestJSONDiagnostics checks -logos.json on a Go type error: one JSON line
// with an empty rule_id.
func TestJSONDiagnostics(t *testing.T) {
	res := compileFile(t, "x.go", "package p\n\nfunc f() { undefined() }\n", "-logos.json")
	lines := strings.Split(strings.TrimSpace(res.output), "\n")
	if res.exitCode != 2 || len(lines) != 1 || !strings.HasPrefix(lines[0], `{"schema_version":"1.0","severity":"deny","rule_id":"","file":"`) {
		t.Errorf("JSON compile: exit %d, output %q", res.exitCode, res.output)
	}
}
