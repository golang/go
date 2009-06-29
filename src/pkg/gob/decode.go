// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"io";
	"os";
)

// DecodeUint reads an encoded unsigned integer from r.
func DecodeUint(r io.Reader) (x uint64, err os.Error) {
	var buf [1]byte;
	for shift := uint(0);; shift += 7 {
		n, err := r.Read(&buf);
		if n != 1 {
			return 0, err
		}
		b := uint64(buf[0]);
		x |= b << shift;
		if b&0x80 != 0 {
			x &^= 0x80 << shift;
			break
		}
	}
	return x, nil;
}

// DecodeInt reads an encoded signed integer from r.
func DecodeInt(r io.Reader) (i int64, err os.Error) {
	x, err := DecodeUint(r);
	if err != nil {
		return
	}
	if x & 1 != 0 {
		return ^int64(x>>1), nil
	}
	return int64(x >> 1), nil
}
