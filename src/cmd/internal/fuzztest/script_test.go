// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzztest

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"testing"
)

//go:generate go test cmd/internal/fuzztest -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.SkipIfShortAndSlow(t)
	scripttest.RunToolScriptTest(t, nil, "testdata/script", *fixReadme)
}
