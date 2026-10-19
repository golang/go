// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

import (
	"strings"
	"testing"
)

// TestLogosUnsupportedActionText drives one .lgo case through runLogos
// with an action outside the harness set and pins the exact error the
// case reports: `logos harness: unsupported action <action> for .lgo`.
// The plain-Go upstream path dies with ENOENT instead, so this test
// fails when the dispatch drops the logos harness or the error text
// drifts from the harness contract.
func TestLogosUnsupportedActionText(t *testing.T) {
	ts := test{goFile: "codegen/unsupported_action.lgo", T: t}
	err := ts.runLogos(logosCase{action: "rundir", src: []byte("package main\n")})
	if err == nil {
		t.Fatal("runLogos on unsupported .lgo action returned nil")
	}
	want := "logos harness: unsupported action rundir for .lgo"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("runLogos error = %q; want it to contain %q", err.Error(), want)
	}
}

// TestRouteLogosHarnessSkipBypass pins the one non-trivial dispatch
// contract: // skip never routes into the logos harness, and every
// other action under test/logos does. A regression that routes skip
// into runLogos would try to stage and build a deliberately skipped
// file.
func TestRouteLogosHarnessSkipBypass(t *testing.T) {
	if routeLogosHarness("logos", "codegen/unsupported_action.lgo", "skip", nil) {
		t.Fatal("routeLogosHarness routed a skip case into the logos harness")
	}
	if routeLogosHarness("fixedbugs", "x.go", "rundir", nil) {
		t.Fatal("routeLogosHarness routed a non-logos directory case")
	}
}

// TestLogosGCFlagsScopedToTestPackage pins test-only compiler flags to
// the test package: the staged argv keeps an all=<base> element free of
// -m, -e, and -d so dependencies compile with the shared base only. A
// regression that folds test flags into the all= element rebuilds the
// standard library with debug flags.
func TestLogosGCFlagsScopedToTestPackage(t *testing.T) {
	t.Setenv("GO_GCFLAGS", "-l")
	args := logosGCFlagsArgs([]string{"-m"})
	if len(args) != 2 {
		t.Fatalf("logosGCFlagsArgs returned %d elements, want 2: %q", len(args), args)
	}
	all, scoped := args[0], args[1]
	if !strings.Contains(all, "all=-l") {
		t.Fatalf("dependency element lost the shared base: %q", all)
	}
	if strings.Contains(all, "-m") || strings.Contains(all, "-e") || strings.Contains(all, "-d=") {
		t.Fatalf("dependency element carries test flags: %q", all)
	}
	if !strings.Contains(scoped, "-m") || !strings.Contains(scoped, "-l") || !strings.Contains(scoped, "-e") {
		t.Fatalf("test-package element dropped flags or -e: %q", scoped)
	}
}
