// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"flag"
	"fmt"
	"os"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/tool"
)

var (
	runSubprocessTests       = flag.Bool("enable_gopls_subprocess_tests", false, "run regtests against a gopls subprocess")
	goplsBinaryPath          = flag.String("gopls_test_binary", "", "path to the gopls binary for use as a remote, for use with the -gopls_subprocess_testmode flag")
	alwaysPrintLogs          = flag.Bool("regtest_print_rpc_logs", false, "whether to always print RPC logs")
	regtestTimeout           = flag.Duration("regtest_timeout", 60*time.Second, "default timeout for each regtest")
	printGoroutinesOnFailure = flag.Bool("regtest_print_goroutines", false, "whether to print goroutine info on failure")
)

var runner *Runner

func TestMain(m *testing.M) {
	flag.Parse()
	if os.Getenv("_GOPLS_TEST_BINARY_RUN_AS_GOPLS") == "true" {
		tool.Main(context.Background(), cmd.New("gopls", "", nil, nil), os.Args[1:])
		os.Exit(0)
	}
	resetExitFuncs := lsprpc.OverrideExitFuncsForTest()
	defer resetExitFuncs()

	runner = &Runner{
		DefaultModes:             NormalModes,
		Timeout:                  *regtestTimeout,
		AlwaysPrintLogs:          *alwaysPrintLogs,
		PrintGoroutinesOnFailure: *printGoroutinesOnFailure,
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

	code := m.Run()
	runner.Close()
	os.Exit(code)
}
