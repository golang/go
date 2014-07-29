// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"runtime"
	"testing"
)

// TestGCInfo tests that various objects in heap, data and bss receive correct GC pointer type info.
func TestGCInfo(t *testing.T) {
	verifyGCInfo(t, "bss ScalarPtr", &bssScalarPtr, infoScalarPtr)
	verifyGCInfo(t, "bss PtrScalar", &bssPtrScalar, infoPtrScalar)
	verifyGCInfo(t, "bss Complex", &bssComplex, infoComplex())
	verifyGCInfo(t, "bss string", &bssString, infoString)
	verifyGCInfo(t, "bss eface", &bssEface, infoEface)

	verifyGCInfo(t, "data ScalarPtr", &dataScalarPtr, infoScalarPtr)
	verifyGCInfo(t, "data PtrScalar", &dataPtrScalar, infoPtrScalar)
	verifyGCInfo(t, "data Complex", &dataComplex, infoComplex())
	verifyGCInfo(t, "data string", &dataString, infoString)
	verifyGCInfo(t, "data eface", &dataEface, infoEface)

	for i := 0; i < 3; i++ {
		verifyGCInfo(t, "heap ScalarPtr", escape(new(ScalarPtr)), infoScalarPtr)
		verifyGCInfo(t, "heap PtrScalar", escape(new(PtrScalar)), infoPtrScalar)
		verifyGCInfo(t, "heap Complex", escape(new(Complex)), infoComplex())
		verifyGCInfo(t, "heap string", escape(new(string)), infoString)
		verifyGCInfo(t, "heap eface", escape(new(interface{})), infoEface)
	}

}

func verifyGCInfo(t *testing.T, name string, p interface{}, mask0 []byte) {
	mask := runtime.GCMask(p)
	if len(mask) > len(mask0) {
		mask0 = append(mask0, BitsDead)
		mask = mask[:len(mask0)]
	}
	if bytes.Compare(mask, mask0) != 0 {
		t.Errorf("bad GC program for %v:\nwant %+v\ngot  %+v", name, mask0, mask)
		return
	}
}

var gcinfoSink interface{}

func escape(p interface{}) interface{} {
	gcinfoSink = p
	return p
}

const (
	BitsDead = iota
	BitsScalar
	BitsPointer
	BitsMultiWord
)

const (
	BitsString = iota
	BitsSlice
	BitsIface
	BitsEface
)

type ScalarPtr struct {
	q int
	w *int
	e int
	r *int
	t int
	y *int
}

var infoScalarPtr = []byte{BitsScalar, BitsPointer, BitsScalar, BitsPointer, BitsScalar, BitsPointer}

type PtrScalar struct {
	q *int
	w int
	e *int
	r int
	t *int
	y int
}

var infoPtrScalar = []byte{BitsPointer, BitsScalar, BitsPointer, BitsScalar, BitsPointer, BitsScalar}

type Complex struct {
	q *int
	w byte
	e [17]byte
	r []byte
	t int
	y uint16
	u uint64
	i string
}

func infoComplex() []byte {
	switch runtime.GOARCH {
	case "386", "arm":
		return []byte{
			BitsPointer, BitsScalar, BitsScalar, BitsScalar,
			BitsScalar, BitsScalar, BitsMultiWord, BitsSlice,
			BitsScalar, BitsScalar, BitsScalar, BitsScalar,
			BitsScalar, BitsMultiWord, BitsString,
		}
	case "amd64":
		return []byte{
			BitsPointer, BitsScalar, BitsScalar, BitsScalar,
			BitsMultiWord, BitsSlice, BitsScalar, BitsScalar,
			BitsScalar, BitsScalar, BitsMultiWord, BitsString,
		}
	case "amd64p32":
		return []byte{
			BitsPointer, BitsScalar, BitsScalar, BitsScalar,
			BitsScalar, BitsScalar, BitsMultiWord, BitsSlice,
			BitsScalar, BitsScalar, BitsScalar, BitsScalar,
			BitsScalar, BitsScalar, BitsMultiWord, BitsString,
		}
	default:
		panic("unknown arch")
	}
}

var (
	// BSS
	bssScalarPtr ScalarPtr
	bssPtrScalar PtrScalar
	bssComplex   Complex
	bssString    string
	bssEface     interface{}

	// DATA
	dataScalarPtr             = ScalarPtr{q: 1}
	dataPtrScalar             = PtrScalar{w: 1}
	dataComplex               = Complex{w: 1}
	dataString                = "foo"
	dataEface     interface{} = 42

	infoString = []byte{BitsMultiWord, BitsString}
	infoEface  = []byte{BitsMultiWord, BitsEface}
)
