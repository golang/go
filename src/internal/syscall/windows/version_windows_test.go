// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows_test

import (
	"errors"
	"internal/syscall/windows"
	"syscall"
	"testing"
)

func TestSupportUnixSocket(t *testing.T) {
	var d syscall.WSAData
	if err := syscall.WSAStartup(uint32(0x202), &d); err != nil {
		t.Fatal(err)
	}
	defer syscall.WSACleanup()

	// Test that SupportUnixSocket returns true if WSASocket succeeds with AF_UNIX.
	got := windows.SupportUnixSocket()
	s, err := windows.WSASocket(syscall.AF_UNIX, syscall.SOCK_STREAM, 0, nil, 0, windows.WSA_FLAG_NO_HANDLE_INHERIT)
	if err == nil {
		syscall.Closesocket(s)
	}
	want := !errors.Is(err, windows.WSAEAFNOSUPPORT) && !errors.Is(err, windows.WSAEINVAL)
	if want != got {
		t.Errorf("SupportUnixSocket = %v; want %v", got, want)
	}
}
