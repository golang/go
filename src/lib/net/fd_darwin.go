// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network file descriptors.

package net

import (
	"os";
	"syscall";
	"net"
)

/* BUG 6g has trouble with this.

export type FD os.FD;

export func NewFD(fd int64) (nfd *FD, err *os.Error) {
	ofd := os.NewFD(fd)
	return ofd, nil
}

func (fd *FD) Close() *os.Error {
	var ofd *os.FD = fd
	return ofd.Close()
}

func (fd *FD) Read(p *[]byte) (n int, err *os.Error) {
	var ofd *os.FD = fd;
	n, err = ofd.Read(p)
	return n, err
}

func (fd *FD) Write(p *[]byte) (n int, err *os.Error) {
	var ofd *os.FD = fd;
	n, err = ofd.Write(p)
	return n, err
}

*/

// TODO: Replace with kqueue/kevent.

export type FD struct {
	fd int64;
	osfd *os.FD;
}

export func NewFD(fd int64) (nfd *FD, err *os.Error) {
	nfd = new(FD);
	nfd.osfd = os.NewFD(fd);
	nfd.fd = fd
	return nfd, nil
}

func (fd *FD) Close() *os.Error {
	return fd.osfd.Close()
}

func (fd *FD) Read(p *[]byte) (n int, err *os.Error) {
	n, err = fd.osfd.Read(p)
	return n, err
}

func (fd *FD) Write(p *[]byte) (n int, err *os.Error) {
	n, err = fd.osfd.Write(p)
	return n, err
}

func (fd *FD) Accept(sa *syscall.Sockaddr) (nfd *FD, err *os.Error) {
	s, e := syscall.accept(fd.fd, sa);
	if e != 0 {
		return nil, os.ErrnoToError(e)
	}
	nfd, err = NewFD(s)
	return nfd, err
}

