// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"os"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/lsprpc"
)

var runner *Runner

func TestMain(m *testing.M) {
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
	runner = NewTestRunner(AllModes, 30*time.Second)
	defer runner.Close()
	os.Exit(m.Run())
}
