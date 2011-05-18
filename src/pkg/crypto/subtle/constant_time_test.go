// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import (
	"testing"
	"testing/quick"
)

type TestConstantTimeCompareStruct struct {
	a, b []byte
	out  int
}

var testConstantTimeCompareData = []TestConstantTimeCompareStruct{
	{[]byte{}, []byte{}, 1},
	{[]byte{0x11}, []byte{0x11}, 1},
	{[]byte{0x12}, []byte{0x11}, 0},
}

func TestConstantTimeCompare(t *testing.T) {
	for i, test := range testConstantTimeCompareData {
		if r := ConstantTimeCompare(test.a, test.b); r != test.out {
			t.Errorf("#%d bad result (got %x, want %x)", i, r, test.out)
		}
	}
}

type TestConstantTimeByteEqStruct struct {
	a, b uint8
	out  int
}

var testConstandTimeByteEqData = []TestConstantTimeByteEqStruct{
	{0, 0, 1},
	{0, 1, 0},
	{1, 0, 0},
	{0xff, 0xff, 1},
	{0xff, 0xfe, 0},
}

func byteEq(a, b uint8) int {
	if a == b {
		return 1
	}
	return 0
}

func TestConstantTimeByteEq(t *testing.T) {
	for i, test := range testConstandTimeByteEqData {
		if r := ConstantTimeByteEq(test.a, test.b); r != test.out {
			t.Errorf("#%d bad result (got %x, want %x)", i, r, test.out)
		}
	}
	err := quick.CheckEqual(ConstantTimeByteEq, byteEq, nil)
	if err != nil {
		t.Error(err)
	}
}

func eq(a, b int32) int {
	if a == b {
		return 1
	}
	return 0
}

func TestConstantTimeEq(t *testing.T) {
	err := quick.CheckEqual(ConstantTimeEq, eq, nil)
	if err != nil {
		t.Error(err)
	}
}

func makeCopy(v int, x, y []byte) []byte {
	if len(x) > len(y) {
		x = x[0:len(y)]
	} else {
		y = y[0:len(x)]
	}
	if v == 1 {
		copy(x, y)
	}
	return x
}

func constantTimeCopyWrapper(v int, x, y []byte) []byte {
	if len(x) > len(y) {
		x = x[0:len(y)]
	} else {
		y = y[0:len(x)]
	}
	v &= 1
	ConstantTimeCopy(v, x, y)
	return x
}

func TestConstantTimeCopy(t *testing.T) {
	err := quick.CheckEqual(constantTimeCopyWrapper, makeCopy, nil)
	if err != nil {
		t.Error(err)
	}
}
