// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package checker_test

import (
	"flag"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"runtime"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/internal/checker"
	"golang.org/x/tools/internal/testenv"
)

func main() {
	checker.Fix = true
	patterns := flag.Args()

	code := checker.Run(patterns, []*analysis.Analyzer{analyzer})
	os.Exit(code)
}

// TestFixes ensures that checker.Run applies fixes correctly.
// This test fork/execs the main function above.
func TestFixes(t *testing.T) {
	oses := map[string]bool{"darwin": true, "linux": true}
	if !oses[runtime.GOOS] {
		t.Skipf("skipping fork/exec test on this platform")
	}

	if os.Getenv("TESTFIXES_CHILD") == "1" {
		// child process

		// replace [progname -test.run=TestFixes -- ...]
		//      by [progname ...]
		os.Args = os.Args[2:]
		os.Args[0] = "vet"
		main()
		panic("unreachable")
	}

	testenv.NeedsTool(t, "go")

	files := map[string]string{
		"rename/foo.go": `package rename

func Foo() {
	bar := 12
	_ = bar
}

// the end
`,
		"rename/intestfile_test.go": `package rename

func InTestFile() {
	bar := 13
	_ = bar
}

// the end
`,
		"rename/foo_test.go": `package rename_test

func Foo() {
	bar := 14
	_ = bar
}

// the end
`,
	}
	fixed := map[string]string{
		"rename/foo.go": `package rename

func Foo() {
	baz := 12
	_ = baz
}

// the end
`,
		"rename/intestfile_test.go": `package rename

func InTestFile() {
	baz := 13
	_ = baz
}

// the end
`,
		"rename/foo_test.go": `package rename_test

func Foo() {
	baz := 14
	_ = baz
}

// the end
`,
	}
	dir, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatalf("Creating test files failed with %s", err)
	}
	defer cleanup()

	args := []string{"-test.run=TestFixes", "--", "rename"}
	cmd := exec.Command(os.Args[0], args...)
	cmd.Env = append(os.Environ(), "TESTFIXES_CHILD=1", "GOPATH="+dir, "GO111MODULE=off", "GOPROXY=off")

	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s: out=<<%s>>", args, out)
	}
	var exitcode int
	if err, ok := err.(*exec.ExitError); ok {
		exitcode = err.ExitCode() // requires go1.12
	}

	const diagnosticsExitCode = 3
	if exitcode != diagnosticsExitCode {
		t.Errorf("%s: exited %d, want %d", args, exitcode, diagnosticsExitCode)
	}

	for name, want := range fixed {
		path := path.Join(dir, "src", name)
		contents, err := ioutil.ReadFile(path)
		if err != nil {
			t.Errorf("error reading %s: %v", path, err)
		}
		if got := string(contents); got != want {
			t.Errorf("contents of %s file did not match expectations. got=%s, want=%s", path, got, want)
		}
	}
}
