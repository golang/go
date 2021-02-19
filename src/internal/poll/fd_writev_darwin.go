// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin
// +build darwin

package poll

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

// Implemented in syscall/syscall_darwin.go.
//go:linkname writev syscall.writev
func writev(fd int, iovecs []syscall.Iovec) (uintptr, error)
