// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var b [8]byte
	one := uint8(1)
	f16(&one, b[:2])
	if b[1] != 1 {
		println("2-byte value lost")
	}
	f32(&one, b[:4])
	if b[3] != 1 {
		println("4-byte value lost")
	}
	f64(&one, b[:8])
	if b[7] != 1 {
		println("8-byte value lost")
	}
}

//go:noinline
func f16(p *uint8, b []byte) {
	_ = b[1]            // bounds check
	x := *p             // load a byte
	y := uint16(x)      // zero extend to 16 bits
	b[0] = byte(y >> 8) // compute ROLW
	b[1] = byte(y)
	nop()               // spill/restore ROLW
	b[0] = byte(y >> 8) // use ROLW
	b[1] = byte(y)
}

//go:noinline
func f32(p *uint8, b []byte) {
	_ = b[3]             // bounds check
	x := *p              // load a byte
	y := uint32(x)       // zero extend to 32 bits
	b[0] = byte(y >> 24) // compute ROLL
	b[1] = byte(y >> 16)
	b[2] = byte(y >> 8)
	b[3] = byte(y)
	nop()                // spill/restore ROLL
	b[0] = byte(y >> 24) // use ROLL
	b[1] = byte(y >> 16)
	b[2] = byte(y >> 8)
	b[3] = byte(y)
}

//go:noinline
func f64(p *uint8, b []byte) {
	_ = b[7]             // bounds check
	x := *p              // load a byte
	y := uint64(x)       // zero extend to 64 bits
	b[0] = byte(y >> 56) // compute ROLQ
	b[1] = byte(y >> 48)
	b[2] = byte(y >> 40)
	b[3] = byte(y >> 32)
	b[4] = byte(y >> 24)
	b[5] = byte(y >> 16)
	b[6] = byte(y >> 8)
	b[7] = byte(y)
	nop()                // spill/restore ROLQ
	b[0] = byte(y >> 56) // use ROLQ
	b[1] = byte(y >> 48)
	b[2] = byte(y >> 40)
	b[3] = byte(y >> 32)
	b[4] = byte(y >> 24)
	b[5] = byte(y >> 16)
	b[6] = byte(y >> 8)
	b[7] = byte(y)
}

//go:noinline
func nop() {
}
