// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Ioctl(fd, req, arg uintptr) Errno {
	err := ioctl(int(fd), int(req), int(arg))
	if err != nil {
		return err.(Errno)
	}
	return 0
}
