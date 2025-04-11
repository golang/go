// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const (
	SYS_FCNTL         = 55
	SYS_MPROTECT      = 125
	SYS_PRCTL         = 172
	SYS_EPOLL_CTL     = 251
	SYS_EPOLL_PWAIT   = 346
	SYS_EPOLL_WAIT    = -1
	SYS_EPOLL_CREATE1 = 357
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 356

	EFD_NONBLOCK = 0x800
	NONE         = -1
)

type EpollEvent struct {
	Events uint32
	_pad   uint32
	Data   [8]byte // to match amd64
}
