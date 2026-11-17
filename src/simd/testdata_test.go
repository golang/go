// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package simd_test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// testdataGOEXPERIMENT returns the GOEXPERIMENT setting for the go commands
// that build testdata. If GOEXPERIMENT already enables simd or simdstable, it
// returns GOEXPERIMENT unchanged, so that testdata is built with the same
// experiments as this test, and otherwise it adds simdstable.
func testdataGOEXPERIMENT() string {
	goexp := os.Getenv("GOEXPERIMENT")
	for _, exp := range strings.Split(goexp, ",") {
		if exp == "simd" || exp == "simdstable" {
			return goexp
		}
	}
	if goexp != "" {
		goexp += ","
	}
	return goexp + "simdstable"
}

func common(t *testing.T, dir, what, failWith string, moreEnv ...string) {
	t.Helper()
	t.Logf("subprocess test in testdata")
	testenv.MustHaveGoRun(t)
	args := []string{"test", "-C", dir}
	if testing.Verbose() {
		args = append(args, "-v")
	}
	args = append(args, what)
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)

	cmd.Env = append(cmd.Environ(), "GOEXPERIMENT="+testdataGOEXPERIMENT())
	cmd.Env = append(cmd.Env, moreEnv...)

	if failWith == "" {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Error(err)
		}
	} else {
		combined, err := cmd.CombinedOutput()
		combinedString := string(combined)
		sawFailure := strings.Contains(combinedString, failWith)
		if err == nil && !sawFailure {
			t.Errorf("Saw no error and did not see expected failure string '%s' in '%s'", failWith, combinedString)
		} else if err == nil && sawFailure {
			t.Errorf("Saw no error but did see expected failure string '%s'", failWith)
		} else if err != nil && !sawFailure {
			t.Errorf("Saw error %v but did see expected failure string '%s' in '%s'", err, failWith, combinedString)
		} else /* err != nil && sawFailure */ {
			t.Logf("Saw error %v and expected failure string '%s'", err, failWith)
		}
	}
}

func run(t *testing.T, dir, what, failWith string, moreEnv ...string) {
	t.Helper()
	t.Logf("subprocess test in testdata")
	testenv.MustHaveGoRun(t)
	args := []string{"run", "-C", dir}

	args = append(args, what)
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)

	cmd.Env = append(cmd.Environ(), "GOEXPERIMENT="+testdataGOEXPERIMENT())
	cmd.Env = append(cmd.Env, moreEnv...)

	if failWith == "" {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Error(err)
		}
	} else {
		combined, err := cmd.CombinedOutput()
		combinedString := string(combined)
		sawFailure := strings.Contains(combinedString, failWith)
		if err == nil && !sawFailure {
			t.Errorf("Saw no error and did not see expected failure string '%s' in '%s'", failWith, combinedString)
		} else if err == nil && sawFailure {
			t.Errorf("Saw no error but did see expected failure string '%s'", failWith)
		} else if err != nil && !sawFailure {
			t.Errorf("Saw error %v but did see expected failure string '%s' in '%s'", err, failWith, combinedString)
		} else /* err != nil && sawFailure */ {
			t.Logf("Saw error %v and expected failure string '%s'", err, failWith)
		}
	}
}
func TestIFace(t *testing.T) {
	common(t, "testdata", "iface_test.go", "")
}

func TestImportShadowing(t *testing.T) {
	for _, f := range []string{"dotimport_test.go", "alias_shadow_test.go"} {
		common(t, "testdata", f, "")
	}
}

func TestSizeof(t *testing.T) {
	common(t, "testdata", "sizeof_test.go", "")
}

func TestCompileOk(t *testing.T) {
	common(t, "testdata", "compiles_test.go", "")
}

func TestCompileError(t *testing.T) {
	common(t, "testdata", "errors_test.go",
		"array length unsafe.Sizeof(v_from_simd) (value of type uintptr) must be constant")
}

func TestToString(t *testing.T) {
	common(t, "testdata", "tostring_test.go", "")
}

func TestChainedImport(t *testing.T) {
	run(t, "testdata/timport", "./main1", "")
}

func TestRewrittenInit(t *testing.T) {
	run(t, "testdata/timport", "./main2", "")
}

func TestB81614(t *testing.T) {
	run(t, "testdata/b81614a", "./main", "")
	run(t, "testdata/b81614b", "./main", "")
	run(t, "testdata/b81614c", "./main", "")
	run(t, "testdata/b81614d", "./main", "")
}

func TestLoadPart(t *testing.T) {
	common(t, "testdata", "loadpart_test.go", "")
}

// TestOriginalsNotCompiled checks that the compiler generates nothing for
// the declarations that the midway rewrite keeps only for type checking and
// export data: the portable vector types and their methods in this package,
// and the pre-rewrite versions of declarations that depend on them in other
// packages. Rewritten code never refers to them, but if they are in an
// object file, they still end up in a shared library, where the vector
// types' method tables refer to methods that have no implementation.
func TestOriginalsNotCompiled(t *testing.T) {
	switch runtime.GOARCH {
	case "amd64", "arm64", "wasm":
	default:
		t.Skipf("the compiler doesn't rewrite simd code on %s", runtime.GOARCH)
	}
	testenv.MustHaveGoBuild(t)

	// symbols builds pkg in dir and returns its symbol table.
	symbols := func(dir, pkg string) string {
		t.Helper()
		archive := filepath.Join(t.TempDir(), "pkg.a")
		for _, args := range [][]string{
			{"build", "-C", dir, "-o", archive, pkg},
			{"tool", "nm", archive},
		} {
			cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
			cmd.Env = append(cmd.Environ(), "GOEXPERIMENT="+testdataGOEXPERIMENT())
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go %s: %v\n%s", strings.Join(args, " "), err, out)
			}
			if args[0] == "tool" {
				return string(out)
			}
		}
		panic("unreachable")
	}
	check := func(syms, want, dontWant string) {
		t.Helper()
		if !regexp.MustCompile(want).MatchString(syms) {
			t.Errorf("no symbol matches %s", want)
		}
		for _, sym := range regexp.MustCompile(dontWant).FindAllString(syms, -1) {
			t.Errorf("unexpected symbol %s", strings.TrimSpace(sym))
		}
	}

	// The type descriptor and methods of a portable vector type.
	check(symbols(".", "simd"),
		`(?m) T simd\.VectorBits$`,
		`(?m) (type:\*?simd\.Float32s|simd\.Float32s\.\w+|simd\.\(\*Float32s\)\.\w+)$`)

	// A function and a variable with a portable vector type.
	check(symbols("testdata", "./pkg"),
		`(?m) T simd/testdata/pkg\.F@simd128$`,
		`(?m) simd/testdata/pkg\.[FV]$`)
}
