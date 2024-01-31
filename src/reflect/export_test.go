// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"internal/abi"
	"internal/goarch"
	"internal/reflectlite"
	"sync"
	"unsafe"
)

// MakeRO returns a copy of v with the read-only flag set.
func MakeRO(v Value) Value {
	v.flag |= reflectlite.FlagStickyRO
	return v
}

// IsRO reports whether v's read-only flag is set.
func IsRO(v Value) bool {
	return v.flag&reflectlite.FlagStickyRO != 0
}

var CallGC = &callGC

// FuncLayout calls funcLayout and returns a subset of the results for testing.
//
// Bitmaps like stack, gc, inReg, and outReg are expanded such that each bit
// takes up one byte, so that writing out test cases is a little clearer.
// If ptrs is false, gc will be nil.
func FuncLayout(t Type, rcvr Type) (frametype Type, argSize, retOffset uintptr, stack, gc, inReg, outReg []byte, ptrs bool) {
	var ft *abi.Type
	var abid abiDesc
	if rcvr != nil {
		ft, _, abid = funcLayout((*funcType)(unsafe.Pointer(t.common())), rcvr.common())
	} else {
		ft, _, abid = funcLayout((*funcType)(unsafe.Pointer(t.(*rtype))), nil)
	}
	// Extract size information.
	argSize = abid.stackCallArgsSize
	retOffset = abid.retOffset
	frametype = toType(ft)

	// Expand stack pointer bitmap into byte-map.
	for i := uint32(0); i < abid.stackPtrs.n; i++ {
		stack = append(stack, abid.stackPtrs.data[i/8]>>(i%8)&1)
	}

	// Expand register pointer bitmaps into byte-maps.
	bool2byte := func(b bool) byte {
		if b {
			return 1
		}
		return 0
	}
	for i := 0; i < intArgRegs; i++ {
		inReg = append(inReg, bool2byte(abid.inRegPtrs.Get(i)))
		outReg = append(outReg, bool2byte(abid.outRegPtrs.Get(i)))
	}
	if ft.Kind_&kindGCProg != 0 {
		panic("can't handle gc programs")
	}

	// Expand frame type's GC bitmap into byte-map.
	ptrs = ft.PtrBytes != 0
	if ptrs {
		nptrs := ft.PtrBytes / goarch.PtrSize
		gcdata := ft.GcSlice(0, (nptrs+7)/8)
		for i := uintptr(0); i < nptrs; i++ {
			gc = append(gc, gcdata[i/8]>>(i%8)&1)
		}
	}
	return
}

func TypeLinks() []string {
	var r []string
	sections, offset := typelinks()
	for i, offs := range offset {
		rodata := sections[i]
		for _, off := range offs {
			typ := (*rtype)(resolveTypeOff(rodata, off))
			r = append(r, typ.String())
		}
	}
	return r
}

var GCBits = gcbits

func gcbits(any) []byte // provided by runtime

func MapBucketOf(x, y Type) Type {
	return toType(bucketOf(x.common(), y.common()))
}

func CachedBucketOf(m Type) Type {
	t := m.(*rtype)
	if Kind(t.t.Kind_&kindMask) != Map {
		panic("not map")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.Bucket)
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
	m := ut.Methods()[0]
	mname := t.(*rtype).nameOff(m.Name)
	if *mname.DataChecked(0, "name flag field")&(1<<2) == 0 {
		panic("method name does not have pkgPath *string")
	}
	return mname.Bytes
}

type OtherPkgFields struct {
	OtherExported   int
	otherUnexported int
}

func IsExported(t Type) bool {
	typ := t.(*rtype)
	n := typ.nameOff(typ.t.Str)
	return n.IsExported()
}

func ResolveReflectName(s string) {
	resolveReflectName(newName(s, "", false, false))
}

type Buffer struct {
	buf []byte
}

func clearLayoutCache() {
	layoutCache = sync.Map{}
}

func SetArgRegs(ints, floats int, floatSize uintptr) (oldInts, oldFloats int, oldFloatSize uintptr) {
	oldInts = intArgRegs
	oldFloats = floatArgRegs
	oldFloatSize = floatRegSize
	intArgRegs = ints
	floatArgRegs = floats
	floatRegSize = floatSize
	clearLayoutCache()
	return
}

var MethodValueCallCodePtr = methodValueCallCodePtr

var InternalIsZero = isZero

var IsRegularMemory = isRegularMemory
