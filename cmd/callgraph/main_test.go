// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android

package main

import (
	"bytes"
	"fmt"
	"go/build"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestCallgraph(t *testing.T) {
	ctxt := build.Default // copy
	ctxt.GOPATH = "testdata"

	const format = "{{.Caller}} --> {{.Callee}}"

	for _, test := range []struct {
		algo, format string
		tests        bool
		want         []string
	}{
		{"rta", format, false, []string{
			// rta imprecisely shows cross product of {main,main2} x {C,D}
			`pkg.main --> (pkg.C).f`,
			`pkg.main --> (pkg.D).f`,
			`pkg.main --> pkg.main2`,
			`pkg.main2 --> (pkg.C).f`,
			`pkg.main2 --> (pkg.D).f`,
		}},
		{"pta", format, false, []string{
			// pta distinguishes main->C, main2->D.  Also has a root node.
			`<root> --> pkg.init`,
			`<root> --> pkg.main`,
			`pkg.main --> (pkg.C).f`,
			`pkg.main --> pkg.main2`,
			`pkg.main2 --> (pkg.D).f`,
		}},
		// tests: main is not called.
		{"rta", format, true, []string{
			`pkg.Example --> (pkg.C).f`,
			`test$main.init --> pkg.init`,
		}},
		{"pta", format, true, []string{
			`<root> --> pkg.Example`,
			`<root> --> test$main.init`,
			`pkg.Example --> (pkg.C).f`,
			`test$main.init --> pkg.init`,
		}},
	} {
		stdout = new(bytes.Buffer)
		if err := doCallgraph(&ctxt, test.algo, test.format, test.tests, []string{"pkg"}); err != nil {
			t.Error(err)
			continue
		}

		got := sortedLines(fmt.Sprint(stdout))
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("callgraph(%q, %q, %t):\ngot:\n%s\nwant:\n%s",
				test.algo, test.format, test.tests,
				strings.Join(got, "\n"),
				strings.Join(test.want, "\n"))
		}
	}
}

func sortedLines(s string) []string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")
	sort.Strings(lines)
	return lines
}
