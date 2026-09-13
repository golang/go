// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import (
	"syscall"
	"testing"
)

// TestParseInterfaceAddrMessageDragonFly checks that the DragonFly parser
// reads ifam_index from the offset DragonFly actually uses. DragonFly orders
// the header fields index, flags, addrs; FreeBSD and darwin order them addrs,
// flags, index. The "classic" parser used the FreeBSD offsets for DragonFly
// as well, so ifam_index was read from the ifam_addrs field (usually 0).
func TestParseInterfaceAddrMessageDragonFly(t *testing.T) {
	w := &wireFormat{
		extOff:  syscall.SizeofIfaMsghdr,
		bodyOff: syscall.SizeofIfaMsghdr,
	}
	b := make([]byte, syscall.SizeofIfaMsghdr)
	// ifam_msglen (uint16), little-endian.
	b[0] = byte(len(b))
	b[1] = byte(len(b) >> 8)
	b[3] = syscall.RTM_NEWADDR
	// ifam_index is at offset 4 on DragonFly (offset 12 on FreeBSD/darwin).
	const index = 42
	b[4] = index

	m, err := w.parseInterfaceAddrMessage(b)
	if err != nil {
		t.Fatal(err)
	}
	am, ok := m.(*InterfaceAddrMessage)
	if !ok {
		t.Fatalf("got %T, want *InterfaceAddrMessage", m)
	}
	if am.Index != index {
		t.Errorf("Index = %d, want %d", am.Index, index)
	}
}
