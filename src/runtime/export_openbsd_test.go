// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

package runtime

func Fcntl(fd, cmd, arg uintptr) (uintptr, uintptr) {
	r := fcntl(int32(fd), int32(cmd), int32(arg))
	if r < 0 {
		return ^uintptr(0), uintptr(-r)
	}
	return uintptr(r), 0
}
