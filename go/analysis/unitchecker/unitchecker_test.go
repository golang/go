// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package unitchecker_test

// This test depends on features such as
// go vet's support for vetx files (1.11) and
// the (*os.ProcessState).ExitCode method (1.12).

import (
	"flag"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/findcall"
	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/unitchecker"
)

func TestMain(m *testing.M) {
	if os.Getenv("UNITCHECKER_CHILD") == "1" {
		// child process
		main()
		panic("unreachable")
	}

	// test
	flag.Parse()
	os.Exit(m.Run())
}

func main() {
	unitchecker.Main(
		findcall.Analyzer,
		printf.Analyzer,
	)
}

// This is a very basic integration test of modular
// analysis with facts using unitchecker under "go vet".
// It fork/execs the main function above.
func TestIntegration(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("skipping fork/exec test on this platform")
	}

	testdata := analysistest.TestData()

	const wantA = `# a
testdata/src/a/a.go:4:11: call of MyFunc123(...)
`
	const wantB = `# b
testdata/src/b/b.go:6:13: call of MyFunc123(...)
testdata/src/b/b.go:7:11: call of MyFunc123(...)
`

	for _, test := range []struct {
		args string
		want string
	}{
		{args: "a", want: wantA},
		{args: "b", want: wantB},
		{args: "a b", want: wantA + wantB},
	} {
		cmd := exec.Command("go", "vet", "-vettool="+os.Args[0], "-findcall.name=MyFunc123")
		cmd.Args = append(cmd.Args, strings.Fields(test.args)...)
		cmd.Env = append(os.Environ(),
			"UNITCHECKER_CHILD=1",
			"GOPATH="+testdata,
		)

		out, err := cmd.CombinedOutput()
		exitcode := -1
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitcode = exitErr.ExitCode()
		}
		if exitcode != 2 {
			t.Errorf("%s: got exit code %d, want 2", test.args, exitcode)
		}

		if got := string(out); got != test.want {
			t.Errorf("%s: got <<%s>>, want <<%s>>", test.args, got, test.want)
		}
	}
}
