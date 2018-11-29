// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// A RawConn is a raw network connection.
type RawConn interface {
	// Control invokes f on the underlying connection's file
	// descriptor or handle.
	// The file descriptor fd is guaranteed to remain valid while
	// f executes but not after f returns.
	Control(f func(fd uintptr)) error

	// Read invokes f on the underlying connection's file
	// descriptor or handle; f is expected to try to read from the
	// file descriptor.
	// If f returns true, Read returns. Otherwise Read blocks
	// waiting for the connection to be ready for reading and
	// tries again repeatedly.
	// The file descriptor is guaranteed to remain valid while f
	// executes but not after f returns.
	Read(f func(fd uintptr) (done bool)) error

	// Write is like Read but for writing.
	Write(f func(fd uintptr) (done bool)) error
}

// Conn is implemented by some types in the net package to provide
// access to the underlying file descriptor or handle.
type Conn interface {
	// SyscallConn returns a raw network connection.
	SyscallConn() (RawConn, error)
}
