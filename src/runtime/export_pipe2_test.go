// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd linux netbsd openbsd solaris

package runtime

func Pipe() (r, w int32, errno int32) {
	r, w, errno = pipe2(0)
	if errno == _ENOSYS {
		return pipe()
	}
	return r, w, errno
}
