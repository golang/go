// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopls_test

import (
	"os"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/cache"
	cmdtest "golang.org/x/tools/internal/lsp/cmd/test"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestCommandLine(t *testing.T) {
	packagestest.TestAll(t, testCommandLine)
}

func commandLineOptions(options *source.Options) {
	options.StaticCheck = true
	options.GoDiff = false
	hooks.Options(options)
}

func testCommandLine(t *testing.T, exporter packagestest.Exporter) {
	const testdata = "../../internal/lsp/testdata"
	if stat, err := os.Stat(testdata); err != nil || !stat.IsDir() {
		t.Skip("testdata directory not present")
	}
	data := tests.Load(t, exporter, testdata)
	ctx := tests.Context(t)
	ctx = debug.WithInstance(ctx, "", "")
	cache := cache.New(ctx, commandLineOptions)
	ss := lsprpc.NewStreamServer(cache)
	ts := servertest.NewTCPServer(ctx, ss)
	for _, data := range data {
		defer data.Exported.Cleanup()
		t.Run(data.Folder, func(t *testing.T) {
			t.Helper()
			tests.Run(t, cmdtest.NewRunner(exporter, data, tests.Context(t), ts.Addr, commandLineOptions), data)
		})
	}
}
