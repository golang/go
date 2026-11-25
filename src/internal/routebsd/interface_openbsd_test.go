// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import (
	"encoding/binary"
	"syscall"
	"testing"
)

func TestInterfaceAddrMessageFlags(t *testing.T) {
	for _, tt := range []struct {
		name string
		typ  uint8
	}{
		{"RTM_NEWADDR", syscall.RTM_NEWADDR},
		{"RTM_DELADDR", syscall.RTM_DELADDR},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// Use the layout generated from the OpenBSD headers. The
			// address bitmap and flags must differ to catch a mix-up.
			msg := struct {
				Header syscall.IfaMsghdr
				Addr   syscall.RawSockaddrInet4
			}{
				Header: syscall.IfaMsghdr{
					Msglen:  syscall.SizeofIfaMsghdr + syscall.SizeofSockaddrInet4,
					Version: syscall.RTM_VERSION,
					Type:    tt.typ,
					Hdrlen:  syscall.SizeofIfaMsghdr,
					Index:   1,
					Addrs:   syscall.RTA_IFA,
					Flags:   1, // IFA_ROUTE
				},
				Addr: syscall.RawSockaddrInet4{
					Len:    syscall.SizeofSockaddrInet4,
					Family: syscall.AF_INET,
					Addr:   [4]byte{192, 0, 2, 1},
				},
			}
			b, err := binary.Append(nil, binary.NativeEndian, msg)
			if err != nil {
				t.Fatal(err)
			}
			ms, err := parseRIB(b)
			if err != nil {
				t.Fatalf("parseRIB: %v", err)
			}
			if len(ms) != 1 {
				t.Fatalf("parseRIB returned %d messages, want 1", len(ms))
			}
			m, ok := ms[0].(*InterfaceAddrMessage)
			if !ok {
				t.Fatalf("parseRIB returned %T, want *InterfaceAddrMessage", ms[0])
			}
			if want := int(msg.Header.Flags); m.Flags != want {
				t.Errorf("Flags = %#x, want %#x", m.Flags, want)
			}
		})
	}
}
