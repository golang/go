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

func FuncLayout(t Type, rcvr Type) (frametype Type, argSize, retOffset uintptr, stack []byte, gc []byte, ptrs bool) {
	var ft *rtype
	var s *bitVector
	if rcvr != nil {
		ft, argSize, retOffset, s, _ = funcLayout(t.(*rtype), rcvr.(*rtype))
	} else {
		ft, argSize, retOffset, s, _ = funcLayout(t.(*rtype), nil)
	}
	frametype = ft
	for i := uint32(0); i < s.n; i += 2 {
		stack = append(stack, s.data[i/8]>>(i%8)&3)
	}
	if ft.kind&kindGCProg != 0 {
		panic("can't handle gc programs")
	}
	gcdata := (*[1000]byte)(ft.gc[0])
	for i := uintptr(0); i < ft.size/ptrSize; i++ {
		gc = append(gc, gcdata[i/2]>>(i%2*4+2)&3)
	}
	ptrs = ft.kind&kindNoPointers == 0
	return
}

func TypeLinks() []string {
	var r []string
	for _, m := range typelinks() {
		for _, t := range m {
			r = append(r, *t.string)
		}
	}
	return r
}
