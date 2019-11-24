// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func Fcntl(fd, cmd, arg uintptr) (uintptr, uintptr) {
	return sysvicall3Err(&libc_fcntl, fd, cmd, arg)
}
