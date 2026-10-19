// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

import (
	"strings"
	"testing"
)

// TestLogosAsmcheckTwinRouted pins Important A for .go twins: an
// asmcheck case under test/logos must reach the logos harness instead of
// falling through to the upstream single-file path, which stages nothing
// for it and dies with ENOENT. It asserts routing only; the run-side
// asmcheck case in runLogosGo is still missing, so the report names that
// seam as well.
func TestLogosAsmcheckTwinRouted(t *testing.T) {
	if !routeLogosHarness("logos", "codegen/asm_twin.go", "asmcheck", nil) {
		t.Fatal("routeLogosHarness did not route test/logos asmcheck twin to the logos harness")
	}
}

// TestLogosUnsupportedRundirMessage pins Important A for .lgo files: a
// .lgo case whose action is outside the harness set must reach the logos
// harness instead of the upstream ENOENT path. It asserts routing only;
// no function returns an error string today, so the exact text can only
// be named here, not compared. The report names the seam the fix must add
// before the text itself is assertable.
func TestLogosUnsupportedRundirMessage(t *testing.T) {
	want := "logos harness: unsupported action rundir for .lgo"
	if !routeLogosHarness("logos", "codegen/unsupported_action.lgo", "rundir", nil) {
		t.Fatalf("routeLogosHarness did not route .lgo rundir; want error %q", want)
	}
}

// TestLogosGCFlagsScopedToTestPackage pins Important C: test-only
// compiler flags must reach only the test package while dependencies
// keep the shared base. The fixed shape is two flags,
// -gcflags=all=<base> plus -gcflags=<base> -e <flags>, so no all=
// pattern may carry -m, -e, or -d.
func TestLogosGCFlagsScopedToTestPackage(t *testing.T) {
	t.Setenv("GO_GCFLAGS", "-l")
	got := logosGCFlags([]string{"-m"})
	val, _ := strings.CutPrefix(got, "-gcflags=")
	head, _, _ := strings.Cut(val, " -gcflags=")
	if strings.Contains(head, "all=") && (strings.Contains(head, "-m") || strings.Contains(head, "-e") || strings.Contains(head, "-d=")) {
		t.Fatalf("logosGCFlags leaks test flags to dependencies: %q", got)
	}
	if !strings.Contains(got, "-m") {
		t.Fatalf("logosGCFlags dropped the test flag: %q", got)
	}
	if !strings.Contains(got, "all=") || !strings.Contains(got, "-l") {
		t.Fatalf("logosGCFlags dropped the shared base from dependencies: %q", got)
	}
}
