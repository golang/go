// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asn1

import (
	"bytes"
	"encoding/hex"
	"math/big"
	"strings"
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

type bigIntStruct struct {
	A *big.Int
}

type nestedStruct struct {
	A intStruct
}

type rawContentsStruct struct {
	Raw RawContent
	A   int
}

type implicitTagTest struct {
	A int `asn1:"implicit,tag:5"`
}

type explicitTagTest struct {
	A int `asn1:"explicit,tag:5"`
}

type flagTest struct {
	A Flag `asn1:"tag:0,optional"`
}

type generalizedTimeTest struct {
	A time.Time `asn1:"generalized"`
}

type ia5StringTest struct {
	A string `asn1:"ia5"`
}

type printableStringTest struct {
	A string `asn1:"printable"`
}

type optionalRawValueTest struct {
	A RawValue `asn1:"optional"`
}

type omitEmptyTest struct {
	A []string `asn1:"omitempty"`
}

type defaultTest struct {
	A int `asn1:"optional,default:1"`
}

type testSET []int

var PST = time.FixedZone("PST", -8*60*60)

type marshalTest struct {
	in  interface{}
	out string // hex encoded
}

func farFuture() time.Time {
	t, err := time.Parse(time.RFC3339, "2100-04-05T12:01:01Z")
	if err != nil {
		panic(err)
	}
	return t
}

var marshalTests = []marshalTest{
	{10, "02010a"},
	{127, "02017f"},
	{128, "02020080"},
	{-128, "020180"},
	{-129, "0202ff7f"},
	{intStruct{64}, "3003020140"},
	{bigIntStruct{big.NewInt(0x123456)}, "30050203123456"},
	{twoIntStruct{64, 65}, "3006020140020141"},
	{nestedStruct{intStruct{127}}, "3005300302017f"},
	{[]byte{1, 2, 3}, "0403010203"},
	{implicitTagTest{64}, "3003850140"},
	{explicitTagTest{64}, "3005a503020140"},
	{flagTest{true}, "30028000"},
	{flagTest{false}, "3000"},
	{time.Unix(0, 0).UTC(), "170d3730303130313030303030305a"},
	{time.Unix(1258325776, 0).UTC(), "170d3039313131353232353631365a"},
	{time.Unix(1258325776, 0).In(PST), "17113039313131353134353631362d30383030"},
	{farFuture(), "180f32313030303430353132303130315a"},
	{generalizedTimeTest{time.Unix(1258325776, 0).UTC()}, "3011180f32303039313131353232353631365a"},
	{BitString{[]byte{0x80}, 1}, "03020780"},
	{BitString{[]byte{0x81, 0xf0}, 12}, "03030481f0"},
	{ObjectIdentifier([]int{1, 2, 3, 4}), "06032a0304"},
	{ObjectIdentifier([]int{1, 2, 840, 133549, 1, 1, 5}), "06092a864888932d010105"},
	{ObjectIdentifier([]int{2, 100, 3}), "0603813403"},
	{"test", "130474657374"},
	{
		"" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // This is 127 times 'x'
		"137f" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"78787878787878787878787878787878787878787878787878787878787878",
	},
	{
		"" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" +
			"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // This is 128 times 'x'
		"138180" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"7878787878787878787878787878787878787878787878787878787878787878" +
			"7878787878787878787878787878787878787878787878787878787878787878",
	},
	{ia5StringTest{"test"}, "3006160474657374"},
	{optionalRawValueTest{}, "3000"},
	{printableStringTest{"test"}, "3006130474657374"},
	{printableStringTest{"test*"}, "30071305746573742a"},
	{rawContentsStruct{nil, 64}, "3003020140"},
	{rawContentsStruct{[]byte{0x30, 3, 1, 2, 3}, 64}, "3003010203"},
	{RawValue{Tag: 1, Class: 2, IsCompound: false, Bytes: []byte{1, 2, 3}}, "8103010203"},
	{testSET([]int{10}), "310302010a"},
	{omitEmptyTest{[]string{}}, "3000"},
	{omitEmptyTest{[]string{"1"}}, "30053003130131"},
	{"Σ", "0c02cea3"},
	{defaultTest{0}, "3003020100"},
	{defaultTest{1}, "3000"},
	{defaultTest{2}, "3003020102"},
}

func TestMarshal(t *testing.T) {
	for i, test := range marshalTests {
		data, err := Marshal(test.in)
		if err != nil {
			t.Errorf("#%d failed: %s", i, err)
		}
		out, _ := hex.DecodeString(test.out)
		if !bytes.Equal(out, data) {
			t.Errorf("#%d got: %x want %x\n\t%q\n\t%q", i, data, out, data, out)

		}
	}
}

type marshalErrTest struct {
	in  interface{}
	err string
}

var marshalErrTests = []marshalErrTest{
	{bigIntStruct{nil}, "empty integer"},
}

func TestMarshalError(t *testing.T) {
	for i, test := range marshalErrTests {
		_, err := Marshal(test.in)
		if err == nil {
			t.Errorf("#%d should fail, but success", i)
			continue
		}

		if !strings.Contains(err.Error(), test.err) {
			t.Errorf("#%d got: %v want %v", i, err, test.err)
		}
	}
}

func TestInvalidUTF8(t *testing.T) {
	_, err := Marshal(string([]byte{0xff, 0xff}))
	if err == nil {
		t.Errorf("invalid UTF8 string was accepted")
	}
}

func BenchmarkMarshal(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		for _, test := range marshalTests {
			Marshal(test.in)
		}
	}
}
