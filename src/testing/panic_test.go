// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
)

var testPanicTest = flag.String("test_panic_test", "", "TestPanic: indicates which test should panic")

func TestPanic(t *testing.T) {
	testenv.MustHaveExec(t)

	testCases := []struct {
		desc  string
		flags []string
		want  string
	}{{
		desc:  "root test panics",
		flags: []string{"-test_panic_test=TestPanicHelper"},
		want: `
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
`,
	}, {
		desc:  "subtest panics",
		flags: []string{"-test_panic_test=TestPanicHelper/1"},
		want: `
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
`,
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=TestPanicHelper")
			cmd.Args = append(cmd.Args, tc.flags...)
			cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
			b, _ := cmd.CombinedOutput()
			got := string(b)
			want := strings.TrimSpace(tc.want)
			re := makeRegexp(want)
			if ok, err := regexp.MatchString(re, got); !ok || err != nil {
				t.Errorf("output:\ngot:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

func makeRegexp(s string) string {
	s = regexp.QuoteMeta(s)
	s = strings.ReplaceAll(s, ":NNN:", `:\d+:`)
	s = strings.ReplaceAll(s, "N\\.NNs", `\d*\.\d*s`)
	return s
}

func TestPanicHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	t.Log(t.Name())
	if t.Name() == *testPanicTest {
		panic("panic")
	}
	for i := 0; i < 3; i++ {
		t.Run(fmt.Sprintf("%v", i), func(t *testing.T) {
			t.Log(t.Name())
			if t.Name() == *testPanicTest {
				panic("panic")
			}
		})
	}
}
