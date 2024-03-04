// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"internal/runtime/syscall"
	"testing"
)

func TestEpollctlErrorSign(t *testing.T) {
	v := syscall.EpollCtl(-1, 1, -1, &syscall.EpollEvent{})

	const EBADF = 0x09
	if v != EBADF {
		t.Errorf("epollctl = %v, want %v", v, EBADF)
	}
}
