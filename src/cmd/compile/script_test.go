// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"os"
	"runtime"
	"testing"
)

//go:generate go test cmd/compile -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

var testCompiler string

// TestMain allows this test binary to run as the compiler
// itself, which is helpful for running script tests.
// If COMPILE_TEST_EXEC_COMPILE is set, we treat the run
// as a 'go tool compile' invocation, otherwise behave
// as a normal test binary.
func TestMain(m *testing.M) {
	// Are we being asked to run as the compiler?
	// If so then kick off main.
	if os.Getenv("COMPILE_TEST_EXEC_COMPILE") != "" {
		main()
		os.Exit(0)
	}

	if testExe, err := os.Executable(); err == nil {
		// on wasm, some phones, we expect an error from os.Executable()
		testCompiler = testExe
	}

	// Regular run, just execute tests.
	os.Exit(m.Run())
}

func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	doReplacement := true
	switch runtime.GOOS {
	case "wasip1", "js":
		// wasm doesn't support os.Executable, so we'll skip replacing
		// the installed linker with our test binary.
		doReplacement = false
	}
	repls := []scripttest.ToolReplacement{}
	if doReplacement {
		if testCompiler == "" {
			t.Fatalf("testCompiler not set, can't replace")
		}
		repls = []scripttest.ToolReplacement{
			scripttest.ToolReplacement{
				ToolName:        "compile",
				ReplacementPath: testCompiler,
				EnvVar:          "COMPILE_TEST_EXEC_COMPILE=1",
			},
		}
	}
	scripttest.RunToolScriptTest(t, repls, "testdata/script", *fixReadme)
}
