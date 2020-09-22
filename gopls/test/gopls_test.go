// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopls_test

import (
	"os"
	"testing"

	"golang.org/x/tools/gopls/internal/hooks"
	cmdtest "golang.org/x/tools/internal/lsp/cmd/test"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestCommandLine(t *testing.T) {
	cmdtest.TestCommandLine(t, "../../internal/lsp/testdata", commandLineOptions)
}

func commandLineOptions(options *source.Options) {
	options.Staticcheck = true
	options.GoDiff = false
	tests.DefaultOptions(options)
	hooks.Options(options)
}
