// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"time"
)

func (c *TCPConn) readFrom(r io.Reader) (int64, error) {
	return genericReadFrom(c, r)
}

func dialTCP(net string, laddr, raddr *TCPAddr, deadline time.Time, cancel <-chan struct{}) (*TCPConn, error) {
	if !deadline.IsZero() {
		panic("net.dialTCP: deadline not implemented on Plan 9")
	}
	// TODO(bradfitz,0intro): also use the cancel channel.
	switch net {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, errMissingAddress
	}
	fd, err := dialPlan9(net, laddr, raddr)
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

func (ln *TCPListener) ok() bool { return ln != nil && ln.fd != nil && ln.fd.ctl != nil }

func (ln *TCPListener) accept() (*TCPConn, error) {
	fd, err := ln.fd.acceptPlan9()
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

func (ln *TCPListener) close() error {
	if _, err := ln.fd.ctl.WriteString("hangup"); err != nil {
		ln.fd.ctl.Close()
		return err
	}
	if err := ln.fd.ctl.Close(); err != nil {
		return err
	}
	return nil
}

func (ln *TCPListener) file() (*os.File, error) {
	f, err := ln.dup()
	if err != nil {
		return nil, err
	}
	return f, nil
}

func listenTCP(network string, laddr *TCPAddr) (*TCPListener, error) {
	fd, err := listenPlan9(network, laddr)
	if err != nil {
		return nil, err
	}
	return &TCPListener{fd}, nil
}
