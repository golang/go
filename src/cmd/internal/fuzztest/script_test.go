// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzztest

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"runtime"
	"testing"
)

//go:generate go test cmd/internal/fuzztest -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.SkipIfShortAndSlow(t)
	if runtime.GOARCH == "386" && runtime.GOMAXPROCS(0) > 2 {
		// # 65434: limit GOMAXPROCS for 386 to avoid running out of address space.
		t.Setenv("GOMAXPROCS", "2")
	}
	scripttest.RunToolScriptTest(t, nil, "testdata/script", *fixReadme)
}
