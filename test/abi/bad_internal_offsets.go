// compile

//go:build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genChecker0

var FailCount int

//go:noinline
func NoteFailure(fidx int, pkg string, pref string, parmNo int, _ uint64) {
	FailCount += 1
	if FailCount > 10 {
		panic("bad")
	}
}

//go:noinline
func NoteFailureElem(fidx int, pkg string, pref string, parmNo int, elem int, _ uint64) {
	FailCount += 1
	if FailCount > 10 {
		panic("bad")
	}
}

type StructF0S0 struct {
	F0 int16
	F1 string
	F2 StructF0S1
}

type StructF0S1 struct {
	_ uint16
}

// 0 returns 3 params
//go:registerparams
//go:noinline
func Test0(p0 uint32, p1 StructF0S0, p2 int32) {
	// consume some stack space, so as to trigger morestack
	var pad [256]uint64
	pad[FailCount]++
	if p0 == 0 {
		return
	}
	p1f0c := int16(-3096)
	if p1.F0 != p1f0c {
		NoteFailureElem(0, "genChecker0", "parm", 1, 0, pad[0])
		return
	}
	p1f1c := "f6ꂅ8ˋ<"
	if p1.F1 != p1f1c {
		NoteFailureElem(0, "genChecker0", "parm", 1, 1, pad[0])
		return
	}
	p1f2c := StructF0S1{}
	if p1.F2 != p1f2c {
		NoteFailureElem(0, "genChecker0", "parm", 1, 2, pad[0])
		return
	}
	p2f0c := int32(496713155)
	if p2 != p2f0c {
		NoteFailureElem(0, "genChecker0", "parm", 2, 0, pad[0])
		return
	}
	// recursive call
	Test0(p0-1, p1, p2)
	return
	// 0 addr-taken params, 0 addr-taken returns
}
