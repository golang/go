// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package unix

import "syscall"

func IsNonblock(fd int) (nonblocking bool, err error) {
	flag, e1 := Fcntl(fd, syscall.F_GETFL, 0)
	if e1 != nil {
		return false, e1
	}
	return flag&syscall.O_NONBLOCK != 0, nil
}

func HasNonblockFlag(flag int) bool {
	return flag&syscall.O_NONBLOCK != 0
}
