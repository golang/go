// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

type devirtualization struct {
	pos    string
	callee string
}

const profFileName = "devirt.pprof"
const preProfFileName = "devirt.pprof.node_map"

// testPGODevirtualize tests that specific PGO devirtualize rewrites are performed.
func testPGODevirtualize(t *testing.T, dir string, want []devirtualization, pgoProfileName string) {
	testenv.MustHaveGoRun(t)
	t.Parallel()

	const pkg = "example.com/pgo/devirtualize"

	// Add a go.mod so we have a consistent symbol names in this temp dir.
	goMod := fmt.Sprintf(`module %s
go 1.21
`, pkg)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatalf("error writing go.mod: %v", err)
	}

	// Run the test without PGO to ensure that the test assertions are
	// correct even in the non-optimized version.
	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "test", "."))
	cmd.Dir = dir
	b, err := cmd.CombinedOutput()
	t.Logf("Test without PGO:\n%s", b)
	if err != nil {
		t.Fatalf("Test failed without PGO: %v", err)
	}

	// Build the test with the profile.
	pprof := filepath.Join(dir, pgoProfileName)
	gcflag := fmt.Sprintf("-gcflags=-m=2 -pgoprofile=%s -d=pgodebug=3", pprof)
	out := filepath.Join(dir, "test.exe")
	cmd = testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "test", "-o", out, gcflag, "."))
	cmd.Dir = dir

	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatalf("error creating pipe: %v", err)
	}
	defer pr.Close()
	cmd.Stdout = pw
	cmd.Stderr = pw

	err = cmd.Start()
	pw.Close()
	if err != nil {
		t.Fatalf("error starting go test: %v", err)
	}

	got := make(map[devirtualization]struct{})

	devirtualizedLine := regexp.MustCompile(`(.*): PGO devirtualizing \w+ call .* to (.*)`)

	scanner := bufio.NewScanner(pr)
	for scanner.Scan() {
		line := scanner.Text()
		t.Logf("child: %s", line)

		m := devirtualizedLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}

		d := devirtualization{
			pos:    m[1],
			callee: m[2],
		}
		got[d] = struct{}{}
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("error running go test: %v", err)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading go test output: %v", err)
	}

	if len(got) != len(want) {
		t.Errorf("mismatched devirtualization count; got %v want %v", got, want)
	}
	for _, w := range want {
		if _, ok := got[w]; ok {
			continue
		}
		t.Errorf("devirtualization %v missing; got %v", w, got)
	}

	// Run test with PGO to ensure the assertions are still true.
	cmd = testenv.CleanCmdEnv(testenv.Command(t, out))
	cmd.Dir = dir
	b, err = cmd.CombinedOutput()
	t.Logf("Test with PGO:\n%s", b)
	if err != nil {
		t.Fatalf("Test failed without PGO: %v", err)
	}
}

// TestPGODevirtualize tests that specific functions are devirtualized when PGO
// is applied to the exact source that was profiled.
func TestPGODevirtualize(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata", "pgo", "devirtualize")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "mult.pkg"), 0755); err != nil {
		t.Fatalf("error creating dir: %v", err)
	}
	for _, file := range []string{"devirt.go", "devirt_test.go", profFileName, filepath.Join("mult.pkg", "mult.go")} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	want := []devirtualization{
		// ExerciseIface
		{
			pos:    "./devirt.go:101:20",
			callee: "mult.Mult.Multiply",
		},
		{
			pos:    "./devirt.go:101:39",
			callee: "Add.Add",
		},
		// ExerciseFuncConcrete
		{
			pos:    "./devirt.go:173:36",
			callee: "AddFn",
		},
		{
			pos:    "./devirt.go:173:15",
			callee: "mult.MultFn",
		},
		// ExerciseFuncField
		{
			pos:    "./devirt.go:207:35",
			callee: "AddFn",
		},
		{
			pos:    "./devirt.go:207:19",
			callee: "mult.MultFn",
		},
		// ExerciseFuncClosure
		// TODO(prattmic): Closure callees not implemented.
		//{
		//	pos:    "./devirt.go:249:27",
		//	callee: "AddClosure.func1",
		//},
		//{
		//	pos:    "./devirt.go:249:15",
		//	callee: "mult.MultClosure.func1",
		//},
	}

	testPGODevirtualize(t, dir, want, profFileName)
}

// TestPGOPreprocessDevirtualize tests that specific functions are devirtualized when PGO
// is applied to the exact source that was profiled. The input profile is PGO preprocessed file.
func TestPGOPreprocessDevirtualize(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata", "pgo", "devirtualize")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "mult.pkg"), 0755); err != nil {
		t.Fatalf("error creating dir: %v", err)
	}
	for _, file := range []string{"devirt.go", "devirt_test.go", preProfFileName, filepath.Join("mult.pkg", "mult.go")} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	want := []devirtualization{
		// ExerciseIface
		{
			pos:    "./devirt.go:101:20",
			callee: "mult.Mult.Multiply",
		},
		{
			pos:    "./devirt.go:101:39",
			callee: "Add.Add",
		},
		// ExerciseFuncConcrete
		{
			pos:    "./devirt.go:173:36",
			callee: "AddFn",
		},
		{
			pos:    "./devirt.go:173:15",
			callee: "mult.MultFn",
		},
		// ExerciseFuncField
		{
			pos:    "./devirt.go:207:35",
			callee: "AddFn",
		},
		{
			pos:    "./devirt.go:207:19",
			callee: "mult.MultFn",
		},
		// ExerciseFuncClosure
		// TODO(prattmic): Closure callees not implemented.
		//{
		//	pos:    "./devirt.go:249:27",
		//	callee: "AddClosure.func1",
		//},
		//{
		//	pos:    "./devirt.go:249:15",
		//	callee: "mult.MultClosure.func1",
		//},
	}

	testPGODevirtualize(t, dir, want, preProfFileName)
}

// Regression test for https://go.dev/issue/65615. If a target function changes
// from non-generic to generic we can't devirtualize it (don't know the type
// parameters), but the compiler should not crash.
func TestLookupFuncGeneric(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata", "pgo", "devirtualize")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "mult.pkg"), 0755); err != nil {
		t.Fatalf("error creating dir: %v", err)
	}
	for _, file := range []string{"devirt.go", "devirt_test.go", profFileName, filepath.Join("mult.pkg", "mult.go")} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	// Change MultFn from a concrete function to a parameterized function.
	if err := convertMultToGeneric(filepath.Join(dir, "mult.pkg", "mult.go")); err != nil {
		t.Fatalf("error editing mult.go: %v", err)
	}

	// Same as TestPGODevirtualize except for MultFn, which we cannot
	// devirtualize to because it has become generic.
	//
	// Note that the important part of this test is that the build is
	// successful, not the specific devirtualizations.
	want := []devirtualization{
		// ExerciseIface
		{
			pos:    "./devirt.go:101:20",
			callee: "mult.Mult.Multiply",
		},
		{
			pos:    "./devirt.go:101:39",
			callee: "Add.Add",
		},
		// ExerciseFuncConcrete
		{
			pos:    "./devirt.go:173:36",
			callee: "AddFn",
		},
		// ExerciseFuncField
		{
			pos:    "./devirt.go:207:35",
			callee: "AddFn",
		},
		// ExerciseFuncClosure
		// TODO(prattmic): Closure callees not implemented.
		//{
		//	pos:    "./devirt.go:249:27",
		//	callee: "AddClosure.func1",
		//},
		//{
		//	pos:    "./devirt.go:249:15",
		//	callee: "mult.MultClosure.func1",
		//},
	}

	testPGODevirtualize(t, dir, want, profFileName)
}

var multFnRe = regexp.MustCompile(`func MultFn\(a, b int64\) int64`)

func convertMultToGeneric(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error opening: %w", err)
	}

	if !multFnRe.Match(content) {
		return fmt.Errorf("MultFn not found; update regexp?")
	}

	// Users of MultFn shouldn't need adjustment, type inference should
	// work OK.
	content = multFnRe.ReplaceAll(content, []byte(`func MultFn[T int32|int64](a, b T) T`))

	return os.WriteFile(path, content, 0644)
}
