// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CRC-32 checksum.
// http://en.wikipedia.org/wiki/Cyclic_redundancy_check for links

package crc32

import "os"

export const (
	// Far and away the most common CRC-32 polynomial.
	// Used by ethernet (IEEE 802.3), v.42, fddi, gzip, zip, png, mpeg-2, ...
	IEEE = 0xedb88320;

	// Castagnoli's polynomial, used in iSCSI.
	// Has better error detection characteristics than IEEE.
	// http://dx.doi.org/10.1109/26.231911
	Castagnoli = 0x82f63b78;

	// Koopman's polynomial.
	// Also has better error detection characteristics than IEEE.
	// http://dx.doi.org/10.1109/DSN.2002.1028931
	Koopman = 0xeb31d82e;
)

// TODO(rsc): Change to [256]uint32 once 6g can handle it.
export type Table []uint32

export func MakeTable(poly uint32) Table {
	t := make(Table, 256);
	for i := 0; i < 256; i++ {
		crc := uint32(i);
		for j := 0; j < 8; j++ {
			if crc&1 == 1 {
				crc = (crc>>1) ^ poly;
			} else {
				crc >>= 1;
			}
		}
		t[i] = crc;
	}
	return t;
}

export var ieee = MakeTable(IEEE);

export type Digest struct {
	crc uint32;
	tab Table;
}

export func NewDigest(tab Table) *Digest {
	return &Digest{0, tab};
}

export func NewIEEEDigest() *Digest {
	return NewDigest(ieee);
}

func (d *Digest) Write(p []byte) (n int, err *os.Error) {
	crc := d.crc ^ 0xFFFFFFFF;
	tab := d.tab;
	for i := 0; i < len(p); i++ {
		crc = tab[byte(crc) ^ p[i]] ^ (crc >> 8);
	}
	d.crc = crc ^ 0xFFFFFFFF;
	return len(p), nil;
}

func (d *Digest) Sum32() uint32 {
	return d.crc
}

func (d *Digest) Sum() []byte {
	p := make([]byte, 4);
	s := d.Sum32();
	p[0] = byte(s>>24);
	p[1] = byte(s>>16);
	p[2] = byte(s>>8);
	p[3] = byte(s);
	return p;
}


