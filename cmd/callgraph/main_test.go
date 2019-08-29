// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android
// +build go1.11

package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func init() {
	// This test currently requires GOPATH mode.
	// Explicitly disabling module mode should suffix, but
	// we'll also turn off GOPROXY just for good measure.
	if err := os.Setenv("GO111MODULE", "off"); err != nil {
		log.Fatal(err)
	}
	if err := os.Setenv("GOPROXY", "off"); err != nil {
		log.Fatal(err)
	}
}

func TestCallgraph(t *testing.T) {
	testenv.NeedsTool(t, "go")

	gopath, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		algo  string
		tests bool
		want  []string
	}{
		{"rta", false, []string{
			// rta imprecisely shows cross product of {main,main2} x {C,D}
			`pkg.main --> (pkg.C).f`,
			`pkg.main --> (pkg.D).f`,
			`pkg.main --> pkg.main2`,
			`pkg.main2 --> (pkg.C).f`,
			`pkg.main2 --> (pkg.D).f`,
		}},
		{"pta", false, []string{
			// pta distinguishes main->C, main2->D.  Also has a root node.
			`<root> --> pkg.init`,
			`<root> --> pkg.main`,
			`pkg.main --> (pkg.C).f`,
			`pkg.main --> pkg.main2`,
			`pkg.main2 --> (pkg.D).f`,
		}},
		// tests: both the package's main and the test's main are called.
		// The callgraph includes all the guts of the "testing" package.
		{"rta", true, []string{
			`pkg.test.main --> testing.MainStart`,
			`testing.runExample --> pkg.Example`,
			`pkg.Example --> (pkg.C).f`,
			`pkg.main --> (pkg.C).f`,
		}},
		{"pta", true, []string{
			`<root> --> pkg.test.main`,
			`<root> --> pkg.main`,
			`pkg.test.main --> testing.MainStart`,
			`testing.runExample --> pkg.Example`,
			`pkg.Example --> (pkg.C).f`,
			`pkg.main --> (pkg.C).f`,
		}},
	} {
		const format = "{{.Caller}} --> {{.Callee}}"
		stdout = new(bytes.Buffer)
		if err := doCallgraph("testdata/src", gopath, test.algo, format, test.tests, []string{"pkg"}); err != nil {
			t.Error(err)
			continue
		}

		edges := make(map[string]bool)
		for _, line := range strings.Split(fmt.Sprint(stdout), "\n") {
			edges[line] = true
		}
		for _, edge := range test.want {
			if !edges[edge] {
				t.Errorf("callgraph(%q, %t): missing edge: %s",
					test.algo, test.tests, edge)
			}
		}
		if t.Failed() {
			t.Log("got:\n", stdout)
		}
	}
}
