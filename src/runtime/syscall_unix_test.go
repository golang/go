// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"runtime"
	"syscall"
	"testing"
)

func TestSyscallFlagAlignment(t *testing.T) {
	// TODO(mknyszek): Check other flags.
	check := func(name string, got, want int) {
		if got != want {
			t.Errorf("flag %s does not line up: got %d, want %d", name, got, want)
		}
	}
	check("O_WRONLY", runtime.O_WRONLY, syscall.O_WRONLY)
	check("O_CREAT", runtime.O_CREAT, syscall.O_CREAT)
	check("O_TRUNC", runtime.O_TRUNC, syscall.O_TRUNC)
}
