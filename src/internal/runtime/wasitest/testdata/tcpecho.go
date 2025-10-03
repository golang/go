// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"net"
	"os"
	"syscall"
)

func main() {
	if err := run(); err != nil {
		println(err)
		os.Exit(1)
	}
}

func run() error {
	l, err := findListener()
	if err != nil {
		return err
	}
	if l == nil {
		return errors.New("no pre-opened sockets available")
	}
	defer l.Close()

	c, err := l.Accept()
	if err != nil {
		return err
	}
	return handleConn(c)
}

func handleConn(c net.Conn) error {
	defer c.Close()

	var buf [128]byte
	n, err := c.Read(buf[:])
	if err != nil {
		return err
	}
	if _, err := c.Write(buf[:n]); err != nil {
		return err
	}
	if err := c.(*net.TCPConn).CloseWrite(); err != nil {
		return err
	}
	return c.Close()
}

func findListener() (net.Listener, error) {
	// We start looking for pre-opened sockets at fd=3 because 0, 1, and 2
	// are reserved for stdio. Pre-opened directors also start at fd=3, so
	// we skip fds that aren't sockets. Once we reach EBADF we know there
	// are no more pre-opens.
	for preopenFd := uintptr(3); ; preopenFd++ {
		f := os.NewFile(preopenFd, "")
		l, err := net.FileListener(f)
		f.Close()

		switch se, _ := errors.AsType[syscall.Errno](err); se {
		case syscall.ENOTSOCK:
			continue
		case syscall.EBADF:
			err = nil
		}
		return l, err
	}
}
