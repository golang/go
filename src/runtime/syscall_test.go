// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
	"runtime"
	"testing"
)

func TestSyscallArgs(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skipf("skipping test: GOARCH=%s", runtime.GOARCH)
	}
	testenv.MustHaveCGO(t)

	exe, err := buildTestProg(t, "testsyscall")
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, exe)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("test program failed: %v\n%s", err, out)
	}
}
