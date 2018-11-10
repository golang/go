// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import "unsafe"

// MakeRO returns a copy of v with the read-only flag set.
func MakeRO(v Value) Value {
	v.flag |= flagStickyRO
	return v
}

// IsRO reports whether v's read-only flag is set.
func IsRO(v Value) bool {
	return v.flag&flagStickyRO != 0
}

var CallGC = &callGC

const PtrSize = ptrSize

func FuncLayout(t Type, rcvr Type) (frametype Type, argSize, retOffset uintptr, stack []byte, gc []byte, ptrs bool) {
	var ft *rtype
	var s *bitVector
	if rcvr != nil {
		ft, argSize, retOffset, s, _ = funcLayout(t.(*rtype), rcvr.(*rtype))
	} else {
		ft, argSize, retOffset, s, _ = funcLayout(t.(*rtype), nil)
	}
	frametype = ft
	for i := uint32(0); i < s.n; i++ {
		stack = append(stack, s.data[i/8]>>(i%8)&1)
	}
	if ft.kind&kindGCProg != 0 {
		panic("can't handle gc programs")
	}
	gcdata := (*[1000]byte)(unsafe.Pointer(ft.gcdata))
	for i := uintptr(0); i < ft.ptrdata/ptrSize; i++ {
		gc = append(gc, gcdata[i/8]>>(i%8)&1)
	}
	ptrs = ft.kind&kindNoPointers == 0
	return
}

func TypeLinks() []string {
	var r []string
	sections, offset := typelinks()
	for i, offs := range offset {
		rodata := sections[i]
		for _, off := range offs {
			typ := (*rtype)(resolveTypeOff(unsafe.Pointer(rodata), off))
			r = append(r, typ.String())
		}
	}
	return r
}

var GCBits = gcbits

func gcbits(interface{}) []byte // provided by runtime

func MapBucketOf(x, y Type) Type {
	return bucketOf(x.(*rtype), y.(*rtype))
}

func CachedBucketOf(m Type) Type {
	t := m.(*rtype)
	if Kind(t.kind&kindMask) != Map {
		panic("not map")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return tt.bucket
}

type EmbedWithUnexpMeth struct{}

func (EmbedWithUnexpMeth) f() {}

type pinUnexpMeth interface {
	f()
}

var pinUnexpMethI = pinUnexpMeth(EmbedWithUnexpMeth{})

func FirstMethodNameBytes(t Type) *byte {
	_ = pinUnexpMethI

	ut := t.uncommon()
	if ut == nil {
		panic("type has no methods")
	}
	m := ut.methods()[0]
	mname := t.(*rtype).nameOff(m.name)
	if *mname.data(0)&(1<<2) == 0 {
		panic("method name does not have pkgPath *string")
	}
	return mname.bytes
}

type OtherPkgFields struct {
	OtherExported   int
	otherUnexported int
}

func IsExported(t Type) bool {
	typ := t.(*rtype)
	n := typ.nameOff(typ.str)
	return n.isExported()
}

func ResolveReflectName(s string) {
	resolveReflectName(newName(s, "", "", false))
}

type Buffer struct {
	buf []byte
}
