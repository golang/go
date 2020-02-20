// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/tool"
)

var (
	runSubprocessTests = flag.Bool("enable_gopls_subprocess_tests", false, "run regtests against a gopls subprocess")
	goplsBinaryPath    = flag.String("gopls_test_binary", "", "path to the gopls binary for use as a remote, for use with the -gopls_subprocess_testmode flag")
)

var runner *Runner

func TestMain(m *testing.M) {
	flag.Parse()
	if os.Getenv("_GOPLS_TEST_BINARY_RUN_AS_GOPLS") == "true" {
		tool.Main(context.Background(), cmd.New("gopls", "", nil, nil), os.Args[1:])
		os.Exit(0)
	}
	// Override functions that would shut down the test process
	defer func(lspExit, forwarderExit func(code int)) {
		lsp.ServerExitFunc = lspExit
		lsprpc.ForwarderExitFunc = forwarderExit
	}(lsp.ServerExitFunc, lsprpc.ForwarderExitFunc)
	// None of these regtests should be able to shut down a server process.
	lsp.ServerExitFunc = func(code int) {
		panic(fmt.Sprintf("LSP server exited with code %d", code))
	}
	// We don't want our forwarders to exit, but it's OK if they would have.
	lsprpc.ForwarderExitFunc = func(code int) {}

	const testTimeout = 60 * time.Second
	if *runSubprocessTests {
		goplsPath := *goplsBinaryPath
		if goplsPath == "" {
			var err error
			goplsPath, err = testBinaryPath()
			if err != nil {
				panic(fmt.Sprintf("finding test binary path: %v", err))
			}
		}
		runner = NewTestRunner(NormalModes|SeparateProcess, testTimeout, goplsPath)
	} else {
		runner = NewTestRunner(NormalModes, testTimeout, "")
	}
	code := m.Run()
	runner.Close()
	os.Exit(code)
}

func testBinaryPath() (string, error) {
	pth := os.Args[0]
	if !filepath.IsAbs(pth) {
		cwd, err := os.Getwd()
		if err == nil {
			return "", fmt.Errorf("os.Getwd: %v", err)
		}
		pth = filepath.Join(cwd, pth)
	}
	if _, err := os.Stat(pth); err != nil {
		return "", fmt.Errorf("os.Stat: %v", err)
	}
	return pth, nil
}
