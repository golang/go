// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
)

func TestFixedGOROOT(t *testing.T) {
	// Restore both the real GOROOT environment variable, and runtime's copies:
	if orig, ok := syscall.Getenv("GOROOT"); ok {
		defer syscall.Setenv("GOROOT", orig)
	} else {
		defer syscall.Unsetenv("GOROOT")
	}
	envs := runtime.Envs()
	oldenvs := append([]string{}, envs...)
	defer runtime.SetEnvs(oldenvs)

	// attempt to reuse existing envs backing array.
	want := runtime.GOROOT()
	runtime.SetEnvs(append(envs[:0], "GOROOT="+want))

	if got := runtime.GOROOT(); got != want {
		t.Errorf(`initial runtime.GOROOT()=%q, want %q`, got, want)
	}
	if err := syscall.Setenv("GOROOT", "/os"); err != nil {
		t.Fatal(err)
	}
	if got := runtime.GOROOT(); got != want {
		t.Errorf(`after setenv runtime.GOROOT()=%q, want %q`, got, want)
	}
	if err := syscall.Unsetenv("GOROOT"); err != nil {
		t.Fatal(err)
	}
	if got := runtime.GOROOT(); got != want {
		t.Errorf(`after unsetenv runtime.GOROOT()=%q, want %q`, got, want)
	}
}
