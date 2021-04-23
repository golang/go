// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// illumos system calls not present on Solaris.

//go:build amd64 && illumos
// +build amd64,illumos

package unix

import (
	"unsafe"
)

func bytes2iovec(bs [][]byte) []Iovec {
	iovecs := make([]Iovec, len(bs))
	for i, b := range bs {
		iovecs[i].SetLen(len(b))
		if len(b) > 0 {
			// somehow Iovec.Base on illumos is (*int8), not (*byte)
			iovecs[i].Base = (*int8)(unsafe.Pointer(&b[0]))
		} else {
			iovecs[i].Base = (*int8)(unsafe.Pointer(&_zero))
		}
	}
	return iovecs
}

//sys	readv(fd int, iovs []Iovec) (n int, err error)

func Readv(fd int, iovs [][]byte) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	n, err = readv(fd, iovecs)
	return n, err
}

//sys	preadv(fd int, iovs []Iovec, off int64) (n int, err error)

func Preadv(fd int, iovs [][]byte, off int64) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	n, err = preadv(fd, iovecs, off)
	return n, err
}

//sys	writev(fd int, iovs []Iovec) (n int, err error)

func Writev(fd int, iovs [][]byte) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	n, err = writev(fd, iovecs)
	return n, err
}

//sys	pwritev(fd int, iovs []Iovec, off int64) (n int, err error)

func Pwritev(fd int, iovs [][]byte, off int64) (n int, err error) {
	iovecs := bytes2iovec(iovs)
	n, err = pwritev(fd, iovecs, off)
	return n, err
}

//sys	accept4(s int, rsa *RawSockaddrAny, addrlen *_Socklen, flags int) (fd int, err error) = libsocket.accept4

func Accept4(fd int, flags int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	nfd, err = accept4(fd, &rsa, &len, flags)
	if err != nil {
		return
	}
	if len > SizeofSockaddrAny {
		panic("RawSockaddrAny too small")
	}
	sa, err = anyToSockaddr(fd, &rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

//sys	putmsg(fd int, clptr *strbuf, dataptr *strbuf, flags int) (err error)

func Putmsg(fd int, cl []byte, data []byte, flags int) (err error) {
	var clp, datap *strbuf
	if len(cl) > 0 {
		clp = &strbuf{
			Len: int32(len(cl)),
			Buf: (*int8)(unsafe.Pointer(&cl[0])),
		}
	}
	if len(data) > 0 {
		datap = &strbuf{
			Len: int32(len(data)),
			Buf: (*int8)(unsafe.Pointer(&data[0])),
		}
	}
	return putmsg(fd, clp, datap, flags)
}

//sys	getmsg(fd int, clptr *strbuf, dataptr *strbuf, flags *int) (err error)

func Getmsg(fd int, cl []byte, data []byte) (retCl []byte, retData []byte, flags int, err error) {
	var clp, datap *strbuf
	if len(cl) > 0 {
		clp = &strbuf{
			Maxlen: int32(len(cl)),
			Buf:    (*int8)(unsafe.Pointer(&cl[0])),
		}
	}
	if len(data) > 0 {
		datap = &strbuf{
			Maxlen: int32(len(data)),
			Buf:    (*int8)(unsafe.Pointer(&data[0])),
		}
	}

	if err = getmsg(fd, clp, datap, &flags); err != nil {
		return nil, nil, 0, err
	}

	if len(cl) > 0 {
		retCl = cl[:clp.Len]
	}
	if len(data) > 0 {
		retData = data[:datap.Len]
	}
	return retCl, retData, flags, nil
}
