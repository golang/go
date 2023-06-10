// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package net

import (
	"syscall"
	"testing"
)

// The tests in this file intend to validate the ability for net.FileConn and
// net.FileListener to handle both TCP and UDP sockets. Ideally we would test
// the public interface by constructing an *os.File from a file descriptor
// opened on a socket, but the WASI preview 1 specification is too limited to
// support this approach for UDP sockets. Instead, we test the internals that
// make it possible for WASI host runtimes and guest programs to integrate
// socket extensions with the net package using net.FileConn/net.FileListener.
//
// Note that the creation of net.Conn and net.Listener values for TCP sockets
// has an end-to-end test in src/runtime/internal/wasitest, here we are only
// verifying the code paths specific to UDP, and error handling for invalid use
// of the functions.

func TestWasip1FileConnNet(t *testing.T) {
	tests := []struct {
		filetype syscall.Filetype
		network  string
		error    error
	}{
		{syscall.FILETYPE_SOCKET_STREAM, "tcp", nil},
		{syscall.FILETYPE_SOCKET_DGRAM, "udp", nil},
		{syscall.FILETYPE_BLOCK_DEVICE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_CHARACTER_DEVICE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_DIRECTORY, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_REGULAR_FILE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_SYMBOLIC_LINK, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_UNKNOWN, "", syscall.ENOTSOCK},
	}
	for _, test := range tests {
		net, err := fileConnNet(test.filetype)
		if net != test.network {
			t.Errorf("fileConnNet: network mismatch: want=%q got=%q", test.network, net)
		}
		if err != test.error {
			t.Errorf("fileConnNet: error mismatch: want=%v got=%v", test.error, err)
		}
	}
}

func TestWasip1FileListenNet(t *testing.T) {
	tests := []struct {
		filetype syscall.Filetype
		network  string
		error    error
	}{
		{syscall.FILETYPE_SOCKET_STREAM, "tcp", nil},
		{syscall.FILETYPE_SOCKET_DGRAM, "", syscall.EOPNOTSUPP},
		{syscall.FILETYPE_BLOCK_DEVICE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_CHARACTER_DEVICE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_DIRECTORY, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_REGULAR_FILE, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_SYMBOLIC_LINK, "", syscall.ENOTSOCK},
		{syscall.FILETYPE_UNKNOWN, "", syscall.ENOTSOCK},
	}
	for _, test := range tests {
		net, err := fileListenNet(test.filetype)
		if net != test.network {
			t.Errorf("fileListenNet: network mismatch: want=%q got=%q", test.network, net)
		}
		if err != test.error {
			t.Errorf("fileListenNet: error mismatch: want=%v got=%v", test.error, err)
		}
	}
}

func TestWasip1NewFileListener(t *testing.T) {
	if l, ok := newFileListener(newFD("tcp", -1)).(*TCPListener); !ok {
		t.Errorf("newFileListener: tcp listener type mismatch: %T", l)
	}
}

func TestWasip1NewFileConn(t *testing.T) {
	if c, ok := newFileConn(newFD("tcp", -1)).(*TCPConn); !ok {
		t.Errorf("newFileConn: tcp conn type mismatch: %T", c)
	}
	if c, ok := newFileConn(newFD("udp", -1)).(*UDPConn); !ok {
		t.Errorf("newFileConn: udp conn type mismatch: %T", c)
	}
}
