// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package unix

// Getsockname copies the binary encoding of the current address for s
// into addr.
func Getsockname(s int, addr []byte) error {
	return getsockname(s, addr)
}

// Getpeername copies the binary encoding of the peer address for s
// into addr.
func Getpeername(s int, addr []byte) error {
	return getpeername(s, addr)
}

var emptyPayload uintptr

// Recvfrom receives a message from s, copying the message into b.
// The socket address addr must be large enough for storing the source
// address of the message.
// Flags must be operation control flags or 0.
// It retunrs the number of bytes copied into b.
func Recvfrom(s int, b []byte, flags int, addr []byte) (int, error) {
	return recvfrom(s, b, flags, addr)
}

// Sendto sends a message to the socket address addr, copying the
// message from b.
// The socket address addr must be suitable for s.
// Flags must be operation control flags or 0.
// It retunrs the number of bytes copied from b.
func Sendto(s int, b []byte, flags int, addr []byte) (int, error) {
	return sendto(s, b, flags, addr)
}
