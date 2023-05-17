// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package unix

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

// Implemented in the runtime package.
//
//go:linkname fcntl runtime.fcntl
func fcntl(fd int32, cmd int32, arg int32) int32

func Fcntl(fd int, cmd int, arg int) (int, error) {
	val := fcntl(int32(fd), int32(cmd), int32(arg))
	if val < 0 {
		return 0, syscall.Errno(-val)
	}
	return int(val), nil
}
