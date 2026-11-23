// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"strings"
	"testing"
)

func TestGOMAXPROCSUpdate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test: long sleeps")
	}
	t.Parallel()

	got := runTestProg(t, "testprog", "WindowsUpdateGOMAXPROCS", "GOMAXPROCS=")
	if strings.Contains(got, "SKIP") {
		t.Skip(got)
	}
	if !strings.Contains(got, "OK") {
		t.Fatalf("output got %q want OK", got)
	}
}

func TestCgroupGOMAXPROCSDontUpdate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test: long sleeps")
	}
	t.Parallel()

	// Two ways to disable updates: explicit GOMAXPROCS or GODEBUG for
	// update feature.
	for _, v := range []string{"GOMAXPROCS=4", "GODEBUG=updatemaxprocs=0"} {
		t.Run(v, func(t *testing.T) {
			got := runTestProg(t, "testprog", "WindowsDontUpdateGOMAXPROCS", "GOMAXPROCS=", v)
			if strings.Contains(got, "SKIP") {
				t.Skip(got)
			}
			if !strings.Contains(got, "OK") {
				t.Fatalf("output got %q want OK", got)
			}
		})
	}
}
