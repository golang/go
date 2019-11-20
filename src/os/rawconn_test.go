// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test use of raw connections.
// +build !plan9,!js

package os_test

import (
	"os"
	"syscall"
	"testing"
)

func TestRawConnReadWrite(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	rconn, err := r.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	wconn, err := w.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}

	var operr error
	err = wconn.Write(func(s uintptr) bool {
		_, operr = syscall.Write(syscallDescriptor(s), []byte{'b'})
		return operr != syscall.EAGAIN
	})
	if err != nil {
		t.Fatal(err)
	}
	if operr != nil {
		t.Fatal(err)
	}

	var n int
	buf := make([]byte, 1)
	err = rconn.Read(func(s uintptr) bool {
		n, operr = syscall.Read(syscallDescriptor(s), buf)
		return operr != syscall.EAGAIN
	})
	if err != nil {
		t.Fatal(err)
	}
	if operr != nil {
		t.Fatal(operr)
	}
	if n != 1 {
		t.Errorf("read %d bytes, expected 1", n)
	}
	if buf[0] != 'b' {
		t.Errorf("read %q, expected %q", buf, "b")
	}
}
