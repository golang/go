// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asn1

import (
	"bytes"
	"encoding/hex"
	"testing"
	"time"
)

type intStruct struct {
	A int
}

type twoIntStruct struct {
	A int
	B int
}

type nestedStruct struct {
	A intStruct
}

type marshalTest struct {
	in  interface{}
	out string // hex encoded
}

type implicitTagTest struct {
	A int "implicit,tag:5"
}

type explicitTagTest struct {
	A int "explicit,tag:5"
}

type ia5StringTest struct {
	A string "ia5"
}

type printableStringTest struct {
	A string "printable"
}

func setPST(t *time.Time) *time.Time {
	t.ZoneOffset = -28800
	return t
}

var marshalTests = []marshalTest{
	marshalTest{10, "02010a"},
	marshalTest{intStruct{64}, "3003020140"},
	marshalTest{twoIntStruct{64, 65}, "3006020140020141"},
	marshalTest{nestedStruct{intStruct{127}}, "3005300302017f"},
	marshalTest{[]byte{1, 2, 3}, "0403010203"},
	marshalTest{implicitTagTest{64}, "3003850140"},
	marshalTest{explicitTagTest{64}, "3005a503020140"},
	marshalTest{time.SecondsToUTC(0), "170d3730303130313030303030305a"},
	marshalTest{time.SecondsToUTC(1258325776), "170d3039313131353232353631365a"},
	marshalTest{setPST(time.SecondsToUTC(1258325776)), "17113039313131353232353631362d30383030"},
	marshalTest{BitString{[]byte{0x80}, 1}, "03020780"},
	marshalTest{BitString{[]byte{0x81, 0xf0}, 12}, "03030481f0"},
	marshalTest{ObjectIdentifier([]int{1, 2, 3, 4}), "06032a0304"},
	marshalTest{"test", "130474657374"},
	marshalTest{ia5StringTest{"test"}, "3006160474657374"},
	marshalTest{printableStringTest{"test"}, "3006130474657374"},
}

func TestMarshal(t *testing.T) {
	for i, test := range marshalTests {
		buf := bytes.NewBuffer(nil)
		err := Marshal(buf, test.in)
		if err != nil {
			t.Errorf("#%d failed: %s", i, err)
		}
		out, _ := hex.DecodeString(test.out)
		if bytes.Compare(out, buf.Bytes()) != 0 {
			t.Errorf("#%d got: %x want %x", i, buf.Bytes(), out)
		}
	}
}
