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
	"runtime"
	"strings"
	"testing"
)

var testPanicTest = flag.String("test_panic_test", "", "TestPanic: indicates which test should panic")
var testPanicParallel = flag.Bool("test_panic_parallel", false, "TestPanic: run subtests in parallel")
var testPanicCleanup = flag.Bool("test_panic_cleanup", false, "TestPanic: indicates whether test should call Cleanup")
var testPanicCleanupPanic = flag.String("test_panic_cleanup_panic", "", "TestPanic: indicate whether test should call Cleanup function that panics")

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
    TestPanicHelper
`,
	}, {
		desc:  "subtest panics",
		flags: []string{"-test_panic_test=TestPanicHelper/1"},
		want: `
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "subtest panics with cleanup",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "subtest panics with outer cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=outer"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
`,
	}, {
		desc:  "subtest panics with middle cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=middle"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "subtest panics with inner cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=inner"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "parallel subtest panics with cleanup",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_parallel"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "parallel subtest panics with outer cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=outer", "-test_panic_parallel"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
`,
	}, {
		desc:  "parallel subtest panics with middle cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=middle", "-test_panic_parallel"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}, {
		desc:  "parallel subtest panics with inner cleanup panic",
		flags: []string{"-test_panic_test=TestPanicHelper/1", "-test_panic_cleanup", "-test_panic_cleanup_panic=inner", "-test_panic_parallel"},
		want: `
ran inner cleanup 1
ran middle cleanup 1
ran outer cleanup
--- FAIL: TestPanicHelper (N.NNs)
    panic_test.go:NNN: TestPanicHelper
    TestPanicHelper
    --- FAIL: TestPanicHelper/1 (N.NNs)
        panic_test.go:NNN: TestPanicHelper/1
        TestPanicHelper/1
`,
	}}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			cmd := exec.Command(testenv.Executable(t), "-test.run=^TestPanicHelper$")
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
	t.Output().Write([]byte(t.Name()))
	if t.Name() == *testPanicTest {
		panic("panic")
	}
	switch *testPanicCleanupPanic {
	case "", "outer", "middle", "inner":
	default:
		t.Fatalf("bad -test_panic_cleanup_panic: %s", *testPanicCleanupPanic)
	}
	t.Cleanup(func() {
		fmt.Println("ran outer cleanup")
		if *testPanicCleanupPanic == "outer" {
			panic("outer cleanup")
		}
	})
	for i := 0; i < 3; i++ {
		t.Run(fmt.Sprintf("%v", i), func(t *testing.T) {
			chosen := t.Name() == *testPanicTest
			if chosen && *testPanicCleanup {
				t.Cleanup(func() {
					fmt.Printf("ran middle cleanup %d\n", i)
					if *testPanicCleanupPanic == "middle" {
						panic("middle cleanup")
					}
				})
			}
			if chosen && *testPanicParallel {
				t.Parallel()
			}
			t.Log(t.Name())
			t.Output().Write([]byte(t.Name()))
			if chosen {
				if *testPanicCleanup {
					t.Cleanup(func() {
						fmt.Printf("ran inner cleanup %d\n", i)
						if *testPanicCleanupPanic == "inner" {
							panic("inner cleanup")
						}
					})
				}
				panic("panic")
			}
		})
	}
}

func TestMorePanic(t *testing.T) {
	testenv.MustHaveExec(t)

	testCases := []struct {
		desc  string
		flags []string
		want  string
	}{
		{
			desc:  "Issue 48502: call runtime.Goexit in t.Cleanup after panic",
			flags: []string{"-test.run=^TestGoexitInCleanupAfterPanicHelper$"},
			want: `panic: die
	panic: test executed panic(nil) or runtime.Goexit`,
		},
		{
			desc:  "Issue 48515: call t.Run in t.Cleanup should trigger panic",
			flags: []string{"-test.run=^TestCallRunInCleanupHelper$"},
			want:  `panic: testing: t.Run called during t.Cleanup`,
		},
	}

	for _, tc := range testCases {
		cmd := exec.Command(testenv.Executable(t), tc.flags...)
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
		b, _ := cmd.CombinedOutput()
		got := string(b)
		want := tc.want
		re := makeRegexp(want)
		if ok, err := regexp.MatchString(re, got); !ok || err != nil {
			t.Errorf("output:\ngot:\n%s\nwant:\n%s", got, want)
		}
	}
}

func TestCallRunInCleanupHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	t.Cleanup(func() {
		t.Run("in-cleanup", func(t *testing.T) {
			t.Log("must not be executed")
		})
	})
}

func TestGoexitInCleanupAfterPanicHelper(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	t.Cleanup(func() { runtime.Goexit() })
	t.Parallel()
	panic("die")
}
