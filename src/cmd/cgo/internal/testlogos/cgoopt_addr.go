// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
static unsigned char cgo_opt_addr_value;

static void cgo_opt_addr_consume(void *p) {
	if (p != 0) {
		cgo_opt_addr_value = *(unsigned char *)p;
	}
}

static unsigned char cgo_opt_addr_observed(void) {
	return cgo_opt_addr_value;
}
*/
import "C"

import "unsafe"

type cgoOptAddrPacket struct {
	header [16]byte
	next   *int
}

func cgoOptConsumeAddr(p *cgoOptAddrPacket) {
	C.cgo_opt_addr_consume(unsafe.Pointer(&p.header))
}

func cgoOptAddrObserved() byte {
	return byte(C.cgo_opt_addr_observed())
}
