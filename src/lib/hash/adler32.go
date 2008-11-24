// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Adler-32 checksum. 
// Defined in RFC 1950:
//	Adler-32 is composed of two sums accumulated per byte: s1 is
//	the sum of all bytes, s2 is the sum of all s1 values. Both sums
//	are done modulo 65521. s1 is initialized to 1, s2 to zero.  The
//	Adler-32 checksum is stored as s2*65536 + s1 in most-
//	significant-byte first (network) order.

package adler32

import "os"

export type Digest struct {
	a, b uint32;
	n int;
}

const Mod = 65521;
const Maxiter = 5552;	// max mod-free iterations before would overflow uint32

export func NewDigest() *Digest {
	return &Digest{1, 0, 0};
}

func (d *Digest) Write(p *[]byte) (nn int, err *os.Error) {
	a, b, n := d.a, d.b, d.n;
	for i := 0; i < len(p); i++ {
		a += uint32(p[i]);
		b += a;
		n++;
		if n == Maxiter {
			a %= Mod;
			b %= Mod;
			n = 0;
		}
	}
	d.a, d.b, d.n = a, b, n;
	return len(p), nil
}

func (d *Digest) Sum32() uint32 {
	a, b := d.a, d.b;
	if a >= Mod || b >= Mod {
		a %= Mod;
		b %= Mod;
	}
	return b<<16 | a;
}

func (d *Digest) Sum() *[]byte {
	p := new([]byte, 4);
	s := d.Sum32();
	p[0] = byte(s>>24);
	p[1] = byte(s>>16);
	p[2] = byte(s>>8);
	p[3] = byte(s);
	return p;
}
