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

// testPGODevirtualize tests that specific PGO devirtualize rewrites are performed.
func testPGODevirtualize(t *testing.T, dir string) {
	testenv.MustHaveGoRun(t)
	t.Parallel()

	const pkg = "example.com/pgo/devirtualize"

	// Add a go.mod so we have a consistent symbol names in this temp dir.
	goMod := fmt.Sprintf(`module %s
go 1.19
`, pkg)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatalf("error writing go.mod: %v", err)
	}

	// Build the test with the profile.
	pprof := filepath.Join(dir, "devirt.pprof")
	gcflag := fmt.Sprintf("-gcflags=-m=2 -pgoprofile=%s -d=pgodebug=3", pprof)
	out := filepath.Join(dir, "test.exe")
	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "build", "-o", out, gcflag, "."))
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

	type devirtualization struct {
		pos    string
		callee string
	}

	want := []devirtualization{
		{
			pos:    "./devirt.go:66:21",
			callee: "mult.Mult.Multiply",
		},
		{
			pos:    "./devirt.go:66:31",
			callee: "Add.Add",
		},
	}

	got := make(map[devirtualization]struct{})

	devirtualizedLine := regexp.MustCompile(`(.*): PGO devirtualizing .* to (.*)`)

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
	for _, file := range []string{"devirt.go", "devirt_test.go", "devirt.pprof", filepath.Join("mult.pkg", "mult.go")} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	testPGODevirtualize(t, dir)
}
