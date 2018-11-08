// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package unitchecker_test

// This test depends on go1.12 features such as go vet's support for
// GOVETTOOL, and the (*os/exec.ExitError).ExitCode method.

import (
	"flag"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
	"golang.org/x/tools/go/analysis/internal/unitchecker"
	"golang.org/x/tools/go/analysis/passes/findcall"
	"golang.org/x/tools/go/analysis/passes/printf"
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
	findcall.Analyzer.Flags.Set("name", "MyFunc123")

	var analyzers = []*analysis.Analyzer{
		findcall.Analyzer,
		printf.Analyzer,
	}

	if err := analysis.Validate(analyzers); err != nil {
		log.Fatal(err)
	}
	analyzers = analysisflags.Parse(analyzers, true)

	args := flag.Args()
	if len(args) != 1 || !strings.HasSuffix(args[0], ".cfg") {
		log.Fatalf("invalid command: want .cfg file")
	}

	unitchecker.Main(args[0], analyzers)
}

// This is a very basic integration test of modular
// analysis with facts using unitchecker under "go vet".
// It fork/execs the main function above.
func TestIntegration(t *testing.T) {
	t.Skip("skipping broken test; golang.org/issue/28676")

	if runtime.GOOS != "linux" {
		t.Skipf("skipping fork/exec test on this platform")
	}

	testdata := analysistest.TestData()

	cmd := exec.Command("go", "vet", "b")
	cmd.Env = append(os.Environ(),
		"GOVETTOOL="+os.Args[0],
		"UNITCHECKER_CHILD=1",
		"GOPATH="+testdata,
	)

	out, err := cmd.CombinedOutput()
	exitcode := -1
	if exitErr, ok := err.(*exec.ExitError); ok {
		exitcode = exitErr.ExitCode()
	}
	if exitcode != 2 {
		t.Errorf("got exit code %d, want 2", exitcode)
	}

	want := `
# a
testdata/src/a/a.go:4:11: call of MyFunc123(...)
# b
testdata/src/b/b.go:6:13: call of MyFunc123(...)
testdata/src/b/b.go:7:11: call of MyFunc123(...)
`[1:]
	if got := string(out); got != want {
		t.Errorf("got <<%s>>, want <<%s>>", got, want)
	}

	if t.Failed() {
		t.Logf("err=%v stderr=<<%s>", err, cmd.Stderr)
	}
}
