// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopls_test

import (
	"os"
	"testing"

	"golang.org/x/tools/gopls/internal/hooks"
	cmdtest "golang.org/x/tools/gopls/internal/lsp/cmd/test"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	testenv.ExitIfSmallMachine()

	// Set the global exporter to nil so that we don't log to stderr. This avoids
	// a lot of misleading noise in test output.
	//
	// See also ../internal/lsp/lsp_test.go.
	event.SetExporter(nil)

	os.Exit(m.Run())
}

func TestCommandLine(t *testing.T) {
	cmdtest.TestCommandLine(t, "../internal/lsp/testdata", commandLineOptions)
}

func commandLineOptions(options *source.Options) {
	options.Staticcheck = true
	options.GoDiff = false
	tests.DefaultOptions(options)
	hooks.Options(options)
}
