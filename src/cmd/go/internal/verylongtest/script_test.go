// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verylongtest

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"testing"
)

//go:generate go test cmd/go/internal/verylongtest -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

func TestScript(t *testing.T) {
	if testing.Short() {
		// Don't bother setting up the script engine. None of these are short tests.
		t.Skip()
	}
	testenv.MustHaveGoBuild(t)
	testenv.SkipIfShortAndSlow(t)
	scripttest.RunToolScriptTest(t, nil, "testdata/script", *fixReadme)
}
