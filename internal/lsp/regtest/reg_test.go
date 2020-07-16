// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/tool"
)

var (
	runSubprocessTests       = flag.Bool("enable_gopls_subprocess_tests", false, "run regtests against a gopls subprocess")
	goplsBinaryPath          = flag.String("gopls_test_binary", "", "path to the gopls binary for use as a remote, for use with the -enable_gopls_subprocess_tests flag")
	regtestTimeout           = flag.Duration("regtest_timeout", 20*time.Second, "default timeout for each regtest")
	skipCleanup              = flag.Bool("regtest_skip_cleanup", false, "whether to skip cleaning up temp directories")
	printGoroutinesOnFailure = flag.Bool("regtest_print_goroutines", false, "whether to print goroutines info on failure")
)

var runner *Runner

func run(t *testing.T, files string, f TestFunc) {
	runner.Run(t, files, f)
}

func withOptions(opts ...RunOption) configuredRunner {
	return configuredRunner{opts: opts}
}

type configuredRunner struct {
	opts []RunOption
}

func (r configuredRunner) run(t *testing.T, files string, f TestFunc) {
	runner.Run(t, files, f, r.opts...)
}

func TestMain(m *testing.M) {
	flag.Parse()
	if os.Getenv("_GOPLS_TEST_BINARY_RUN_AS_GOPLS") == "true" {
		tool.Main(context.Background(), cmd.New("gopls", "", nil, nil), os.Args[1:])
		os.Exit(0)
	}

	runner = &Runner{
		DefaultModes:             NormalModes,
		Timeout:                  *regtestTimeout,
		PrintGoroutinesOnFailure: *printGoroutinesOnFailure,
		SkipCleanup:              *skipCleanup,
	}
	if *runSubprocessTests {
		goplsPath := *goplsBinaryPath
		if goplsPath == "" {
			var err error
			goplsPath, err = os.Executable()
			if err != nil {
				panic(fmt.Sprintf("finding test binary path: %v", err))
			}
		}
		runner.DefaultModes = NormalModes | SeparateProcess
		runner.GoplsPath = goplsPath
	}
	dir, err := ioutil.TempDir("", "gopls-regtest-")
	if err != nil {
		panic(fmt.Errorf("creating regtest temp directory: %v", err))
	}
	runner.TempDir = dir

	code := m.Run()
	if err := runner.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "closing test runner: %v\n", err)
		os.Exit(1)
	}
	os.Exit(code)
}
