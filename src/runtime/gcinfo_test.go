// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"runtime"
	"testing"
)

const (
	typeScalar  = 0
	typePointer = 1
)

// TestGCInfo tests that various objects in heap, data and bss receive correct GC pointer type info.
func TestGCInfo(t *testing.T) {
	verifyGCInfo(t, "bss Ptr", &bssPtr, infoPtr)
	verifyGCInfo(t, "bss ScalarPtr", &bssScalarPtr, infoScalarPtr)
	verifyGCInfo(t, "bss PtrScalar", &bssPtrScalar, infoPtrScalar)
	verifyGCInfo(t, "bss BigStruct", &bssBigStruct, infoBigStruct())
	verifyGCInfo(t, "bss string", &bssString, infoString)
	verifyGCInfo(t, "bss slice", &bssSlice, infoSlice)
	verifyGCInfo(t, "bss eface", &bssEface, infoEface)
	verifyGCInfo(t, "bss iface", &bssIface, infoIface)

	verifyGCInfo(t, "data Ptr", &dataPtr, infoPtr)
	verifyGCInfo(t, "data ScalarPtr", &dataScalarPtr, infoScalarPtr)
	verifyGCInfo(t, "data PtrScalar", &dataPtrScalar, infoPtrScalar)
	verifyGCInfo(t, "data BigStruct", &dataBigStruct, infoBigStruct())
	verifyGCInfo(t, "data string", &dataString, infoString)
	verifyGCInfo(t, "data slice", &dataSlice, infoSlice)
	verifyGCInfo(t, "data eface", &dataEface, infoEface)
	verifyGCInfo(t, "data iface", &dataIface, infoIface)

	{
		var x Ptr
		verifyGCInfo(t, "stack Ptr", &x, infoPtr)
		runtime.KeepAlive(x)
	}
	{
		var x ScalarPtr
		verifyGCInfo(t, "stack ScalarPtr", &x, infoScalarPtr)
		runtime.KeepAlive(x)
	}
	{
		var x PtrScalar
		verifyGCInfo(t, "stack PtrScalar", &x, infoPtrScalar)
		runtime.KeepAlive(x)
	}
	{
		var x BigStruct
		verifyGCInfo(t, "stack BigStruct", &x, infoBigStruct())
		runtime.KeepAlive(x)
	}
	{
		var x string
		verifyGCInfo(t, "stack string", &x, infoString)
		runtime.KeepAlive(x)
	}
	{
		var x []string
		verifyGCInfo(t, "stack slice", &x, infoSlice)
		runtime.KeepAlive(x)
	}
	{
		var x any
		verifyGCInfo(t, "stack eface", &x, infoEface)
		runtime.KeepAlive(x)
	}
	{
		var x Iface
		verifyGCInfo(t, "stack iface", &x, infoIface)
		runtime.KeepAlive(x)
	}

	for i := 0; i < 10; i++ {
		verifyGCInfo(t, "heap Ptr", runtime.Escape(new(Ptr)), trimDead(infoPtr))
		verifyGCInfo(t, "heap PtrSlice", runtime.Escape(&make([]*byte, 10)[0]), trimDead(infoPtr10))
		verifyGCInfo(t, "heap ScalarPtr", runtime.Escape(new(ScalarPtr)), trimDead(infoScalarPtr))
		verifyGCInfo(t, "heap ScalarPtrSlice", runtime.Escape(&make([]ScalarPtr, 4)[0]), trimDead(infoScalarPtr4))
		verifyGCInfo(t, "heap PtrScalar", runtime.Escape(new(PtrScalar)), trimDead(infoPtrScalar))
		verifyGCInfo(t, "heap BigStruct", runtime.Escape(new(BigStruct)), trimDead(infoBigStruct()))
		verifyGCInfo(t, "heap string", runtime.Escape(new(string)), trimDead(infoString))
		verifyGCInfo(t, "heap eface", runtime.Escape(new(any)), trimDead(infoEface))
		verifyGCInfo(t, "heap iface", runtime.Escape(new(Iface)), trimDead(infoIface))
	}
}

func verifyGCInfo(t *testing.T, name string, p any, mask0 []byte) {
	mask := runtime.PointerMask(p)
	if bytes.HasPrefix(mask, mask0) {
		// Just the prefix matching is OK.
		//
		// The Go runtime's pointer/scalar iterator generates pointers beyond
		// the size of the type, up to the size of the size class. This space
		// is safe for the GC to scan since it's zero, and GCBits checks to
		// make sure that's true. But we need to handle the fact that the bitmap
		// may be larger than we expect.
		return
	}
	t.Errorf("bad GC program for %v:\nwant %+v\ngot  %+v", name, mask0, mask)
}

func trimDead(mask []byte) []byte {
	for len(mask) > 0 && mask[len(mask)-1] == typeScalar {
		mask = mask[:len(mask)-1]
	}
	return mask
}

var infoPtr = []byte{typePointer}

type Ptr struct {
	*byte
}

var infoPtr10 = []byte{typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer, typePointer}

type ScalarPtr struct {
	q int
	w *int
	e int
	r *int
	t int
	y *int
}

var infoScalarPtr = []byte{typeScalar, typePointer, typeScalar, typePointer, typeScalar, typePointer}

var infoScalarPtr4 = append(append(append(append([]byte(nil), infoScalarPtr...), infoScalarPtr...), infoScalarPtr...), infoScalarPtr...)

type PtrScalar struct {
	q *int
	w int
	e *int
	r int
	t *int
	y int
}

var infoPtrScalar = []byte{typePointer, typeScalar, typePointer, typeScalar, typePointer, typeScalar}

type BigStruct struct {
	q *int
	w byte
	e [17]byte
	r []byte
	t int
	y uint16
	u uint64
	i string
}

func infoBigStruct() []byte {
	switch runtime.GOARCH {
	case "386", "arm", "mips", "mipsle":
		return []byte{
			typePointer,                                                // q *int
			typeScalar, typeScalar, typeScalar, typeScalar, typeScalar, // w byte; e [17]byte
			typePointer, typeScalar, typeScalar, // r []byte
			typeScalar, typeScalar, typeScalar, typeScalar, // t int; y uint16; u uint64
			typePointer, typeScalar, // i string
		}
	case "arm64", "amd64", "loong64", "mips64", "mips64le", "ppc64", "ppc64le", "riscv64", "s390x", "wasm":
		return []byte{
			typePointer,                        // q *int
			typeScalar, typeScalar, typeScalar, // w byte; e [17]byte
			typePointer, typeScalar, typeScalar, // r []byte
			typeScalar, typeScalar, typeScalar, // t int; y uint16; u uint64
			typePointer, typeScalar, // i string
		}
	default:
		panic("unknown arch")
	}
}

type Iface interface {
	f()
}

type IfaceImpl int

func (IfaceImpl) f() {
}

var (
	// BSS
	bssPtr       Ptr
	bssScalarPtr ScalarPtr
	bssPtrScalar PtrScalar
	bssBigStruct BigStruct
	bssString    string
	bssSlice     []string
	bssEface     any
	bssIface     Iface

	// DATA
	dataPtr             = Ptr{new(byte)}
	dataScalarPtr       = ScalarPtr{q: 1}
	dataPtrScalar       = PtrScalar{w: 1}
	dataBigStruct       = BigStruct{w: 1}
	dataString          = "foo"
	dataSlice           = []string{"foo"}
	dataEface     any   = 42
	dataIface     Iface = IfaceImpl(42)

	infoString = []byte{typePointer, typeScalar}
	infoSlice  = []byte{typePointer, typeScalar, typeScalar}
	infoEface  = []byte{typeScalar, typePointer}
	infoIface  = []byte{typeScalar, typePointer}
)
