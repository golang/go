// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binary

import (
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

func TestRead(t *testing.T) {
	var sl, sb Struct;

	err := Read(bytes.NewBuffer(big), BigEndian, &sb);
	if err != nil {
		t.Errorf("Read big-endian: %v", err);
		goto little;
	}
	if !reflect.DeepEqual(sb, s) {
		t.Errorf("Read big-endian:\n\thave %+v\n\twant %+v", sb, s)
	}

little:
	err = Read(bytes.NewBuffer(little), LittleEndian, &sl);
	if err != nil {
		t.Errorf("Read little-endian: %v", err)
	}
	if !reflect.DeepEqual(sl, s) {
		t.Errorf("Read big-endian:\n\thave %+v\n\twant %+v", sl, s)
	}
}
