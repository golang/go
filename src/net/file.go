// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "os"

type fileAddr string

func (fileAddr) Network() string  { return "file+net" }
func (f fileAddr) String() string { return string(f) }

// FileConn returns a copy of the network connection corresponding to
// the open file f.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func FileConn(f *os.File) (c Conn, err error) {
	c, err = fileConn(f)
	if err != nil {
		err = &OpError{Op: "file", Net: "file+net", Source: nil, Addr: fileAddr(f.Name()), Err: err}
	}
	return
}

// FileListener returns a copy of the network listener corresponding
// to the open file f.
// It is the caller's responsibility to close ln when finished.
// Closing ln does not affect f, and closing f does not affect ln.
func FileListener(f *os.File) (ln Listener, err error) {
	ln, err = fileListener(f)
	if err != nil {
		err = &OpError{Op: "file", Net: "file+net", Source: nil, Addr: fileAddr(f.Name()), Err: err}
	}
	return
}

// FilePacketConn returns a copy of the packet network connection
// corresponding to the open file f.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func FilePacketConn(f *os.File) (c PacketConn, err error) {
	c, err = filePacketConn(f)
	if err != nil {
		err = &OpError{Op: "file", Net: "file+net", Source: nil, Addr: fileAddr(f.Name()), Err: err}
	}
	return
}

// A SocketAddr is used with SocketConn or SocketPacketConn to
// implement a user-configured socket address.
// The net package does not provide any implementations of SocketAddr;
// the caller of SocketConn or SocketPacketConn is expected to provide
// one.
type SocketAddr interface {
	// Addr takes a platform-specific socket address and returns
	// a net.Addr. The result may be nil when the syscall package,
	// system call or underlying protocol does not support the
	// socket address.
	Addr([]byte) Addr

	// Raw takes a net.Addr and returns a platform-specific socket
	// address. The result may be nil when the syscall package,
	// system call or underlying protocol does not support the
	// socket address.
	Raw(Addr) []byte
}

// SocketConn returns a copy of the network connection corresponding
// to the open file f and user-defined socket address sa.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func SocketConn(f *os.File, sa SocketAddr) (c Conn, err error) {
	c, err = socketConn(f, sa)
	if err != nil {
		err = &OpError{Op: "file", Net: "file+net", Source: nil, Addr: fileAddr(f.Name()), Err: err}
	}
	return
}

// SocketPacketConn returns a copy of the packet network connection
// corresponding to the open file f and user-defined socket address
// sa.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func SocketPacketConn(f *os.File, sa SocketAddr) (c PacketConn, err error) {
	c, err = socketPacketConn(f, sa)
	if err != nil {
		err = &OpError{Op: "file", Net: "file+net", Source: nil, Addr: fileAddr(f.Name()), Err: err}
	}
	return
}
