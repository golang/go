// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"io";
	"os";
)

// Integers encode as a variant of Google's protocol buffer varint (varvarint?).
// The variant is that the continuation bytes have a zero top bit instead of a one.
// That way there's only one bit to clear and the value is a little easier to see if
// you're the unfortunate sort of person who must read the hex to debug.

// EncodeUint writes an encoded unsigned integer to w.
func EncodeUint(w io.Writer, x uint64) os.Error {
	var buf [16]byte;
	var n int;
	for n = 0; x > 127; n++ {
		buf[n] = uint8(x & 0x7F);
		x >>= 7;
	}
	buf[n] = 0x80 | uint8(x);
	nn, err := w.Write(buf[0:n+1]);
	return err;
}

// EncodeInt writes an encoded signed integer to w.
// The low bit of the encoding says whether to bit complement the (other bits of the) uint to recover the int.
func EncodeInt(w io.Writer, i int64) os.Error {
	var x uint64;
	if i < 0 {
		x = uint64(^i << 1) | 1
	} else {
		x = uint64(i << 1)
	}
	return EncodeUint(w, uint64(x))
}
