// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
	"runtime"
	"strings"
	"testing"
)

func TestCoroLockOSThread(t *testing.T) {
	for _, test := range []string{
		"CoroLockOSThreadIterLock",
		"CoroLockOSThreadIterLockYield",
		"CoroLockOSThreadLock",
		"CoroLockOSThreadLockIterNested",
		"CoroLockOSThreadLockIterLock",
		"CoroLockOSThreadLockIterLockYield",
		"CoroLockOSThreadLockIterYieldNewG",
		"CoroLockOSThreadLockAfterPull",
		"CoroLockOSThreadStopLocked",
		"CoroLockOSThreadStopLockedIterNested",
	} {
		t.Run(test, func { t -> checkCoroTestProgOutput(t, runTestProg(t, "testprog", test)) })
	}
}

func TestCoroCgoCallback(t *testing.T) {
	testenv.MustHaveCGO(t)
	if runtime.GOOS == "windows" {
		t.Skip("coro cgo callback tests not supported on Windows")
	}
	for _, test := range []string{
		"CoroCgoIterCallback",
		"CoroCgoIterCallbackYield",
		"CoroCgoCallback",
		"CoroCgoCallbackIterNested",
		"CoroCgoCallbackIterCallback",
		"CoroCgoCallbackIterCallbackYield",
		"CoroCgoCallbackAfterPull",
		"CoroCgoStopCallback",
		"CoroCgoStopCallbackIterNested",
	} {
		t.Run(test, func { t -> checkCoroTestProgOutput(t, runTestProg(t, "testprogcgo", test)) })
	}
}

func checkCoroTestProgOutput(t *testing.T, output string) {
	t.Helper()

	c := strings.SplitN(output, "\n", 2)
	if len(c) == 1 {
		t.Fatalf("expected at least one complete line in the output, got:\n%s", output)
	}
	expect, ok := strings.CutPrefix(c[0], "expect: ")
	if !ok {
		t.Fatalf("expected first line of output to start with \"expect: \", got: %q", c[0])
	}
	rest := c[1]
	if expect == "OK" && rest != "OK\n" {
		t.Fatalf("expected just 'OK' in the output, got:\n%s", rest)
	}
	if !strings.Contains(rest, expect) {
		t.Fatalf("expected %q in the output, got:\n%s", expect, rest)
	}
}
