// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/cmd"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/tool"
)

var (
	runSubprocessTests       = flag.Bool("enable_gopls_subprocess_tests", false, "run regtests against a gopls subprocess")
	goplsBinaryPath          = flag.String("gopls_test_binary", "", "path to the gopls binary for use as a remote, for use with the -enable_gopls_subprocess_tests flag")
	regtestTimeout           = flag.Duration("regtest_timeout", defaultRegtestTimeout(), "if nonzero, default timeout for each regtest; defaults to GOPLS_REGTEST_TIMEOUT")
	skipCleanup              = flag.Bool("regtest_skip_cleanup", false, "whether to skip cleaning up temp directories")
	printGoroutinesOnFailure = flag.Bool("regtest_print_goroutines", false, "whether to print goroutines info on failure")
	printLogs                = flag.Bool("regtest_print_logs", false, "whether to print LSP logs")
)

func defaultRegtestTimeout() time.Duration {
	s := os.Getenv("GOPLS_REGTEST_TIMEOUT")
	if s == "" {
		return 0
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid GOPLS_REGTEST_TIMEOUT %q: %v\n", s, err)
		os.Exit(2)
	}
	return d
}

var runner *Runner

type regtestRunner interface {
	Run(t *testing.T, files string, f TestFunc)
}

func Run(t *testing.T, files string, f TestFunc) {
	runner.Run(t, files, f)
}

func WithOptions(opts ...RunOption) configuredRunner {
	return configuredRunner{opts: opts}
}

type configuredRunner struct {
	opts []RunOption
}

func (r configuredRunner) Run(t *testing.T, files string, f TestFunc) {
	runner.Run(t, files, f, r.opts...)
}

type RunMultiple []struct {
	Name   string
	Runner regtestRunner
}

func (r RunMultiple) Run(t *testing.T, files string, f TestFunc) {
	for _, runner := range r {
		t.Run(runner.Name, func(t *testing.T) {
			runner.Runner.Run(t, files, f)
		})
	}
}

// DefaultModes returns the default modes to run for each regression test (they
// may be reconfigured by the tests themselves).
func DefaultModes() Mode {
	modes := Default
	if !testing.Short() {
		modes |= Experimental | Forwarded
	}
	if *runSubprocessTests {
		modes |= SeparateProcess
	}
	return modes
}

// Main sets up and tears down the shared regtest state.
func Main(m *testing.M, hook func(*source.Options)) {
	// golang/go#54461: enable additional debugging around hanging Go commands.
	gocommand.DebugHangingGoCommands = true

	// If this magic environment variable is set, run gopls instead of the test
	// suite. See the documentation for runTestAsGoplsEnvvar for more details.
	if os.Getenv(runTestAsGoplsEnvvar) == "true" {
		tool.Main(context.Background(), cmd.New("gopls", "", nil, hook), os.Args[1:])
		os.Exit(0)
	}

	if !testenv.HasExec() {
		fmt.Printf("skipping all tests: exec not supported on %s\n", runtime.GOOS)
		os.Exit(0)
	}
	testenv.ExitIfSmallMachine()

	// Disable GOPACKAGESDRIVER, as it can cause spurious test failures.
	os.Setenv("GOPACKAGESDRIVER", "off")

	flag.Parse()

	runner = &Runner{
		DefaultModes:             DefaultModes(),
		Timeout:                  *regtestTimeout,
		PrintGoroutinesOnFailure: *printGoroutinesOnFailure,
		SkipCleanup:              *skipCleanup,
		OptionsHook:              hook,
		store:                    memoize.NewStore(memoize.NeverEvict),
	}

	runner.goplsPath = *goplsBinaryPath
	if runner.goplsPath == "" {
		var err error
		runner.goplsPath, err = os.Executable()
		if err != nil {
			panic(fmt.Sprintf("finding test binary path: %v", err))
		}
	}

	dir, err := os.MkdirTemp("", "gopls-regtest-")
	if err != nil {
		panic(fmt.Errorf("creating regtest temp directory: %v", err))
	}
	runner.tempDir = dir

	var code int
	defer func() {
		if err := runner.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "closing test runner: %v\n", err)
			// Regtest cleanup is broken in go1.12 and earlier, and sometimes flakes on
			// Windows due to file locking, but this is OK for our CI.
			//
			// Fail on go1.13+, except for windows and android which have shutdown problems.
			if testenv.Go1Point() >= 13 && runtime.GOOS != "windows" && runtime.GOOS != "android" {
				os.Exit(1)
			}
		}
		os.Exit(code)
	}()
	code = m.Run()
}
