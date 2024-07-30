// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/script/scripttest"
	"flag"
	"internal/testenv"
	"runtime"
	"testing"
)

//go:generate go test cmd/link -v -run=TestScript/README --fixreadme

var fixReadme = flag.Bool("fixreadme", false, "if true, update README for script tests")

func TestScript(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	doReplacement := true
	switch runtime.GOOS {
	case "wasip1", "js":
		// wasm doesn't support os.Executable, so we'll skip replacing
		// the installed linker with our test binary.
		doReplacement = false
	}
	repls := []scripttest.ToolReplacement{}
	if doReplacement {
		if testLinker == "" {
			t.Fatalf("testLinker not set, can't replace")
		}
		repls = []scripttest.ToolReplacement{
			scripttest.ToolReplacement{
				ToolName:        "link",
				ReplacementPath: testLinker,
				EnvVar:          "LINK_TEST_EXEC_LINKER=1",
			},
		}
	}
	scripttest.RunToolScriptTest(t, repls, "testdata/script", *fixReadme)
}
