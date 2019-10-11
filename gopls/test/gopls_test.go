// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopls_test

import (
	"os"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/gopls/internal/hooks"
	cmdtest "golang.org/x/tools/internal/lsp/cmd/test"
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

func testCommandLine(t *testing.T, exporter packagestest.Exporter) {
	const testdata = "../../internal/lsp/testdata"
	if stat, err := os.Stat(testdata); err != nil || !stat.IsDir() {
		t.Skip("testdata directory not present")
	}
	data := tests.Load(t, exporter, testdata)
	defer data.Exported.Cleanup()
	tests.Run(t, cmdtest.NewRunner(exporter, data, tests.Context(t), hooks.Options), data)
}
