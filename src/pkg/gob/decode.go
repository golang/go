// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"io";
	"os";
	"unsafe";
)

// The global execution state of an instance of the decoder.
type DecState struct {
	r	io.Reader;
	err	os.Error;
	base	uintptr;
	buf [1]byte;	// buffer used by the decoder; here to avoid allocation.
}

// DecodeUint reads an encoded unsigned integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func DecodeUint(state *DecState) (x uint64) {
	if state.err != nil {
		return
	}
	for shift := uint(0);; shift += 7 {
		var n int;
		n, state.err = state.r.Read(&state.buf);
		if n != 1 {
			return 0
		}
		b := uint64(state.buf[0]);
		x |= b << shift;
		if b&0x80 != 0 {
			x &^= 0x80 << shift;
			break
		}
	}
	return x;
}

// DecodeInt reads an encoded signed integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func DecodeInt(state *DecState) int64 {
	x := DecodeUint(state);
	if state.err != nil {
		return 0
	}
	if x & 1 != 0 {
		return ^int64(x>>1)
	}
	return int64(x >> 1)
}
