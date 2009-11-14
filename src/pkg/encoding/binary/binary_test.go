// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binary

import (
	"os";
	"bytes";
	"math";
	"reflect";
	"testing";
)

type Struct struct {
	Int8	int8;
	Int16	int16;
	Int32	int32;
	Int64	int64;
	Uint8	uint8;
	Uint16	uint16;
	Uint32	uint32;
	Uint64	uint64;
	Float64	float64;
	Array	[4]uint8;
}

var s = Struct{
	0x01,
	0x0203,
	0x04050607,
	0x08090a0b0c0d0e0f,
	0x10,
	0x1112,
	0x13141516,
	0x1718191a1b1c1d1e,
	math.Float64frombits(0x1f20212223242526),
	[4]uint8{0x27, 0x28, 0x29, 0x2a},
}

var big = []byte{
	1,
	2, 3,
	4, 5, 6, 7,
	8, 9, 10, 11, 12, 13, 14, 15,
	16,
	17, 18,
	19, 20, 21, 22,
	23, 24, 25, 26, 27, 28, 29, 30,
	31, 32, 33, 34, 35, 36, 37, 38,
	39, 40, 41, 42,
}

var little = []byte{
	1,
	3, 2,
	7, 6, 5, 4,
	15, 14, 13, 12, 11, 10, 9, 8,
	16,
	18, 17,
	22, 21, 20, 19,
	30, 29, 28, 27, 26, 25, 24, 23,
	38, 37, 36, 35, 34, 33, 32, 31,
	39, 40, 41, 42,
}

func checkResult(t *testing.T, dir string, order, err os.Error, have, want interface{}) {
	if err != nil {
		t.Errorf("%v %v: %v", dir, order, err);
		return;
	}
	if !reflect.DeepEqual(have, want) {
		t.Errorf("%v %v:\n\thave %+v\n\twant %+v", dir, order, have, want)
	}
}

func testRead(t *testing.T, order ByteOrder, b []byte, s1 interface{}) {
	var s2 Struct;
	err := Read(bytes.NewBuffer(b), order, &s2);
	checkResult(t, "Read", order, err, s2, s1);
}

func testWrite(t *testing.T, order ByteOrder, b []byte, s1 interface{}) {
	buf := new(bytes.Buffer);
	err := Write(buf, order, s1);
	checkResult(t, "Write", order, err, buf.Bytes(), b);
}

func TestBigEndianRead(t *testing.T)	{ testRead(t, BigEndian, big, s) }

func TestLittleEndianRead(t *testing.T)	{ testRead(t, LittleEndian, little, s) }

func TestBigEndianWrite(t *testing.T)	{ testWrite(t, BigEndian, big, s) }

func TestLittleEndianWrite(t *testing.T)	{ testWrite(t, LittleEndian, little, s) }

func TestBigEndianPtrWrite(t *testing.T)	{ testWrite(t, BigEndian, big, &s) }

func TestLittleEndianPtrWrite(t *testing.T)	{ testWrite(t, LittleEndian, little, &s) }
