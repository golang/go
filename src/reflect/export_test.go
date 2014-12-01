// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

// MakeRO returns a copy of v with the read-only flag set.
func MakeRO(v Value) Value {
	v.flag |= flagRO
	return v
}

// IsRO reports whether v's read-only flag is set.
func IsRO(v Value) bool {
	return v.flag&flagRO != 0
}

var ArrayOf = arrayOf
var CallGC = &callGC

const PtrSize = ptrSize
const BitsPointer = bitsPointer
const BitsScalar = bitsScalar

func FuncLayout(t Type, rcvr Type) (frametype Type, argSize, retOffset uintptr, stack []byte) {
	var ft *rtype
	var s *bitVector
	if rcvr != nil {
		ft, argSize, retOffset, s = funcLayout(t.(*rtype), rcvr.(*rtype))
	} else {
		ft, argSize, retOffset, s = funcLayout(t.(*rtype), nil)
	}
	frametype = ft
	for i := uint32(0); i < s.n; i += 2 {
		stack = append(stack, s.data[i/8]>>(i%8)&3)
	}
	return
}
