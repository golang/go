// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"io"
	"os"
	"syscall"
)

func (fd *netFD) status(ln int) (string, error) {
	if !fd.ok() {
		return "", syscall.EINVAL
	}

	status, err := os.Open(fd.dir + "/status")
	if err != nil {
		return "", err
	}
	defer status.Close()
	buf := make([]byte, ln)
	n, err := io.ReadFull(status, buf[:])
	if err != nil {
		return "", err
	}
	return string(buf[:n]), nil
}

func newFileFD(f *os.File) (net *netFD, err error) {
	var ctl *os.File
	close := func(fd int) {
		if err != nil {
			syscall.Close(fd)
		}
	}

	path, err := syscall.Fd2path(int(f.Fd()))
	if err != nil {
		return nil, os.NewSyscallError("fd2path", err)
	}
	comp := splitAtBytes(path, "/")
	n := len(comp)
	if n < 3 || comp[0][0:3] != "net" {
		return nil, syscall.EPLAN9
	}

	name := comp[2]
	switch file := comp[n-1]; file {
	case "ctl", "clone":
		fd, err := syscall.Dup(int(f.Fd()), -1)
		if err != nil {
			return nil, os.NewSyscallError("dup", err)
		}
		defer close(fd)

		dir := netdir + "/" + comp[n-2]
		ctl = os.NewFile(uintptr(fd), dir+"/"+file)
		ctl.Seek(0, io.SeekStart)
		var buf [16]byte
		n, err := ctl.Read(buf[:])
		if err != nil {
			return nil, err
		}
		name = string(buf[:n])
	default:
		if len(comp) < 4 {
			return nil, errors.New("could not find control file for connection")
		}
		dir := netdir + "/" + comp[1] + "/" + name
		ctl, err = os.OpenFile(dir+"/ctl", os.O_RDWR, 0)
		if err != nil {
			return nil, err
		}
		defer close(int(ctl.Fd()))
	}
	dir := netdir + "/" + comp[1] + "/" + name
	laddr, err := readPlan9Addr(comp[1], dir+"/local")
	if err != nil {
		return nil, err
	}
	return newFD(comp[1], name, nil, ctl, nil, laddr, nil)
}

func fileConn(f *os.File) (Conn, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	if !fd.ok() {
		return nil, syscall.EINVAL
	}

	fd.data, err = os.OpenFile(fd.dir+"/data", os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}

	switch fd.laddr.(type) {
	case *TCPAddr:
		return newTCPConn(fd), nil
	case *UDPAddr:
		return newUDPConn(fd), nil
	}
	return nil, syscall.EPLAN9
}

func fileListener(f *os.File) (Listener, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	switch fd.laddr.(type) {
	case *TCPAddr:
	default:
		return nil, syscall.EPLAN9
	}

	// check that file corresponds to a listener
	s, err := fd.status(len("Listen"))
	if err != nil {
		return nil, err
	}
	if s != "Listen" {
		return nil, errors.New("file does not represent a listener")
	}

	return &TCPListener{fd}, nil
}

func filePacketConn(f *os.File) (PacketConn, error) {
	return nil, syscall.EPLAN9
}
