// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package multichecker_test

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/multichecker"
	"golang.org/x/tools/go/analysis/passes/findcall"
	"golang.org/x/tools/internal/testenv"
)

func main() {
	fail := &analysis.Analyzer{
		Name: "fail",
		Doc:  "always fail on a package 'sort'",
		Run: func(pass *analysis.Pass) (interface{}, error) {
			if pass.Pkg.Path() == "sort" {
				return nil, fmt.Errorf("failed")
			}
			return nil, nil
		},
	}
	multichecker.Main(findcall.Analyzer, fail)
}

// TestExitCode ensures that analysis failures are reported correctly.
// This test fork/execs the main function above.
func TestExitCode(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("skipping fork/exec test on this platform")
	}

	if os.Getenv("MULTICHECKER_CHILD") == "1" {
		// child process

		// replace [progname -test.run=TestExitCode -- ...]
		//      by [progname ...]
		os.Args = os.Args[2:]
		os.Args[0] = "vet"
		main()
		panic("unreachable")
	}

	testenv.NeedsTool(t, "go")

	for _, test := range []struct {
		args []string
		want int
	}{
		{[]string{"nosuchdir/..."}, 1},                      // matched no packages
		{[]string{"nosuchpkg"}, 1},                          // matched no packages
		{[]string{"-unknownflag"}, 2},                       // flag error
		{[]string{"-findcall.name=panic", "io"}, 3},         // finds diagnostics
		{[]string{"-findcall=0", "io"}, 0},                  // no checkers
		{[]string{"-findcall.name=nosuchfunc", "io"}, 0},    // no diagnostics
		{[]string{"-findcall.name=panic", "sort", "io"}, 1}, // 'fail' failed on 'sort'

		// -json: exits zero even in face of diagnostics or package errors.
		{[]string{"-findcall.name=panic", "-json", "io"}, 0},
		{[]string{"-findcall.name=panic", "-json", "io"}, 0},
		{[]string{"-findcall.name=panic", "-json", "sort", "io"}, 0},
	} {
		args := []string{"-test.run=TestExitCode", "--"}
		args = append(args, test.args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), "MULTICHECKER_CHILD=1")
		out, err := cmd.CombinedOutput()
		if len(out) > 0 {
			t.Logf("%s: out=<<%s>>", test.args, out)
		}
		var exitcode int
		if err, ok := err.(*exec.ExitError); ok {
			exitcode = err.ExitCode() // requires go1.12
		}
		if exitcode != test.want {
			t.Errorf("%s: exited %d, want %d", test.args, exitcode, test.want)
		}
	}
}
