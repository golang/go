// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

// Illumos system calls not present on Solaris.

package syscall

import "unsafe"

//go:cgo_import_dynamic libc_accept4 accept4 "libsocket.so"
//go:cgo_import_dynamic libc_flock flock "libc.so"

//go:linkname procAccept4 libc_accept4
//go:linkname procFlock libc_flock

var (
	procAccept4,
	procFlock libcFunc
)

func Accept4(fd int, flags int) (int, Sockaddr, error) {
	var rsa RawSockaddrAny
	var addrlen _Socklen = SizeofSockaddrAny
	nfd, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procAccept4)), 4, uintptr(fd), uintptr(unsafe.Pointer(&rsa)), uintptr(unsafe.Pointer(&addrlen)), uintptr(flags), 0, 0)
	if errno != 0 {
		return 0, nil, errno
	}
	if addrlen > SizeofSockaddrAny {
		panic("RawSockaddrAny too small")
	}
	sa, err := anyToSockaddr(&rsa)
	if err != nil {
		Close(int(nfd))
		return 0, nil, err
	}
	return int(nfd), sa, nil
}

func Flock(fd int, how int) error {
	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procFlock)), 2, uintptr(fd), uintptr(how), 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}
