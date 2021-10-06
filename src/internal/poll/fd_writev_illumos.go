// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos
// +build illumos

package poll

import (
	"internal/syscall/unix"
	"syscall"
)

func writev(fd int, iovecs []syscall.Iovec) (uintptr, error) {
	return unix.Writev(fd, iovecs)
}
