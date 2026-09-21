// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"os"
	"testing"
)

//go:generate go test cmd/nm -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	testExe, testExeErr := os.Executable()

	repls := []scripttest.ToolReplacement{}
	if testExeErr == nil {
		repls = []scripttest.ToolReplacement{
			scripttest.ToolReplacement{
				ToolName:        "nm",
				ReplacementPath: testExe,
				EnvVar:          "GO_NMTEST_IS_NM=1",
			},
		}
	}
	scripttest.RunToolScriptTest(t, repls, "testdata/script", *fixReadme)
}
