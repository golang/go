// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MD5 hash algorithm.  See RFC 1321.

package md5

import "os"

package const (
	Chunk = 64
)

const (
	A = 0x67452301;
	B = 0xEFCDAB89;
	C = 0x98BADCFE;
	D = 0x10325476;
)

export type Digest struct {
	s [4]uint32;
	x [Chunk]byte;
	nx int;
	len uint64;
}

export func NewDigest() *Digest {
	d := new(Digest);
	d.s[0] = A;
	d.s[1] = B;
	d.s[2] = C;
	d.s[3] = D;
	return d;
}

package func Block(dig *Digest, p []byte) int

func (d *Digest) Write(p []byte) (nn int, err *os.Error) {
	nn = len(p);
	d.len += uint64(nn);
	if d.nx > 0 {
		n := len(p);
		if n > Chunk-d.nx {
			n = Chunk-d.nx;
		}
		for i := 0; i < n; i++ {
			d.x[d.nx+i] = p[i];
		}
		d.nx += n;
		if d.nx == Chunk {
			Block(d, d.x);
			d.nx = 0;
		}
		p = p[n:len(p)];
	}
	n := Block(d, p);
	p = p[n:len(p)];
	if len(p) > 0 {
		for i := 0; i < len(p); i++ {
			d.x[i] = p[i];
		}
		d.nx = len(p);
	}
	return;
}

func (d *Digest) Sum() []byte {
	// Padding.  Add a 1 bit and 0 bits until 56 bytes mod 64.
	len := d.len;
	var tmp [64]byte;
	tmp[0] = 0x80;
	if len%64 < 56 {
		d.Write(tmp[0:56-len%64]);
	} else {
		d.Write(tmp[0:64+56-len%64]);
	}

	// Length in bits.
	len <<= 3;
	for i := uint(0); i < 8; i++ {
		tmp[i] = byte(len>>(8*i));
	}
	d.Write(tmp[0:8]);

	if d.nx != 0 {
		panicln("oops");
	}

	p := new([]byte, 16);
	j := 0;
	for i := 0; i < 4; i++ {
		s := d.s[i];
		p[j] = byte(s); j++;
		p[j] = byte(s>>8); j++;
		p[j] = byte(s>>16); j++;
		p[j] = byte(s>>24); j++;
	}
	return p;
}

