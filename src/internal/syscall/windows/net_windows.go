// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
	_ "unsafe"
)

//go:linkname WSASendtoInet4 syscall.wsaSendtoInet4
//go:noescape
func WSASendtoInet4(s syscall.Handle, bufs *syscall.WSABuf, bufcnt uint32, sent *uint32, flags uint32, to *syscall.SockaddrInet4, overlapped *syscall.Overlapped, croutine *byte) (err error)

//go:linkname WSASendtoInet6 syscall.wsaSendtoInet6
//go:noescape
func WSASendtoInet6(s syscall.Handle, bufs *syscall.WSABuf, bufcnt uint32, sent *uint32, flags uint32, to *syscall.SockaddrInet6, overlapped *syscall.Overlapped, croutine *byte) (err error)
