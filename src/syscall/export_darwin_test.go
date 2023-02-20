// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

func IoctlPtr(fd, req uintptr, arg unsafe.Pointer) Errno {
	err := ioctlPtr(int(fd), uint(req), arg)
	if err != nil {
		return err.(Errno)
	}
	return 0
}
