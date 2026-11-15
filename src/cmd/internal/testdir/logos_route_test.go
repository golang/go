// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdir_test

import (
	"strings"
	"testing"
)

func TestLogosUnsupportedActionText(t *testing.T) {
	ts := test{goFile: "run/x.lgo", T: t}
	err := ts.runLogos(logosCase{action: "rundir", src: []byte("package main\n")})
	if err == nil {
		t.Fatal("runLogos on unsupported .lgo action returned nil")
	}
	want := "logos harness: unsupported action rundir for .lgo"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("runLogos error = %q; want it to contain %q", err.Error(), want)
	}
}

func TestRouteLogosHarnessSkipBypass(t *testing.T) {
	if routeLogosHarness("logos", "run/x.lgo", "skip", nil) {
		t.Fatal("routeLogosHarness routed a skip case into the logos harness")
	}
	if routeLogosHarness("fixedbugs", "x.go", "rundir", nil) {
		t.Fatal("routeLogosHarness routed a non-logos directory case")
	}
}

func TestLogosCompilerFlagsScopedToTestPackage(t *testing.T) {
	t.Setenv("GO_GCFLAGS", "-l")
	args := logosTestFlagsArgs([]string{"-m"})
	if len(args) != 2 {
		t.Fatalf("logosTestFlagsArgs returned %d elements, want 2: %q", len(args), args)
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

func TestLogosTestCmd(t *testing.T) {
	got := strings.Join(logosTestCmd("logos", "-gcflags=all=", []string{"-race"}, ""), " ")
	if want := "logos test -count=1 -gcflags=all= -race ."; got != want {
		t.Fatalf("seed corpus argv = %q, want %q", got, want)
	}
	got = strings.Join(logosTestCmd("logos", "-gcflags=all=", nil, "60s"), " ")
	if want := "logos test -count=1 -gcflags=all= -fuzz=^Fuzz -fuzztime=60s ."; got != want {
		t.Fatalf("fuzz argv = %q, want %q", got, want)
	}
}
