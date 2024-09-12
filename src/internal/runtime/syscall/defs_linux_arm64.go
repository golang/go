// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const (
	SYS_EPOLL_CREATE1 = 20
	SYS_EPOLL_CTL     = 21
	SYS_EPOLL_PWAIT   = 22
	SYS_FCNTL         = 25
	SYS_MPROTECT      = 226
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 19
	SYS_GETRANDOM     = 278

	EFD_NONBLOCK = 0x800
)

type EpollEvent struct {
	Events uint32
	_pad   uint32
	Data   [8]byte // to match amd64
}
