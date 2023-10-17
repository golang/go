// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import "testing"

func TestFreeBSDNumCPU(t *testing.T) {
	got := runTestProg(t, "testprog", "FreeBSDNumCPU")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got:\n%s", want, got)
	}
}
