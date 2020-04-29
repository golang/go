// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asn1

import (
	"bytes"
	"encoding/hex"
	"math/big"
	"reflect"
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

type genericStringTest struct {
	A string
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

type applicationTest struct {
	A int `asn1:"application,tag:0"`
	B int `asn1:"application,tag:1,explicit"`
}

type privateTest struct {
	A int `asn1:"private,tag:0"`
	B int `asn1:"private,tag:1,explicit"`
	C int `asn1:"private,tag:31"`  // tag size should be 2 octet
	D int `asn1:"private,tag:128"` // tag size should be 3 octet
}

type numericStringTest struct {
	A string `asn1:"numeric"`
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
	{genericStringTest{"test"}, "3006130474657374"},
	{genericStringTest{"test*"}, "30070c05746573742a"},
	{genericStringTest{"test&"}, "30070c057465737426"},
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
	{applicationTest{1, 2}, "30084001016103020102"},
	{privateTest{1, 2, 3, 4}, "3011c00101e103020102df1f0103df81000104"},
	{numericStringTest{"1 9"}, "30051203312039"},
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

type marshalWithParamsTest struct {
	in     interface{}
	params string
	out    string // hex encoded
}

var marshalWithParamsTests = []marshalWithParamsTest{
	{intStruct{10}, "set", "310302010a"},
	{intStruct{10}, "application", "600302010a"},
	{intStruct{10}, "private", "e00302010a"},
}

func TestMarshalWithParams(t *testing.T) {
	for i, test := range marshalWithParamsTests {
		data, err := MarshalWithParams(test.in, test.params)
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
	{numericStringTest{"a"}, "invalid character"},
	{ia5StringTest{"\xb0"}, "invalid character"},
	{printableStringTest{"!"}, "invalid character"},
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

func TestMarshalOID(t *testing.T) {
	var marshalTestsOID = []marshalTest{
		{[]byte("\x06\x01\x30"), "0403060130"}, // bytes format returns a byte sequence \x04
		// {ObjectIdentifier([]int{0}), "060100"}, // returns an error as OID 0.0 has the same encoding
		{[]byte("\x06\x010"), "0403060130"},                // same as above "\x06\x010" = "\x06\x01" + "0"
		{ObjectIdentifier([]int{2, 999, 3}), "0603883703"}, // Example of ITU-T X.690
		{ObjectIdentifier([]int{0, 0}), "060100"},          // zero OID
	}
	for i, test := range marshalTestsOID {
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

func TestIssue11130(t *testing.T) {
	data := []byte("\x06\x010") // == \x06\x01\x30 == OID = 0 (the figure)
	var v interface{}
	// v has Zero value here and Elem() would panic
	_, err := Unmarshal(data, &v)
	if err != nil {
		t.Errorf("%v", err)
		return
	}
	if reflect.TypeOf(v).String() != reflect.TypeOf(ObjectIdentifier{}).String() {
		t.Errorf("marshal OID returned an invalid type")
		return
	}

	data1, err := Marshal(v)
	if err != nil {
		t.Errorf("%v", err)
		return
	}

	if !bytes.Equal(data, data1) {
		t.Errorf("got: %q, want: %q \n", data1, data)
		return
	}

	var v1 interface{}
	_, err = Unmarshal(data1, &v1)
	if err != nil {
		t.Errorf("%v", err)
		return
	}
	if !reflect.DeepEqual(v, v1) {
		t.Errorf("got: %#v data=%q , want : %#v data=%q\n ", v1, data1, v, data)
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

func TestSetEncoder(t *testing.T) {
	testStruct := struct {
		Strings []string `asn1:"set"`
	}{
		Strings: []string{"a", "aa", "b", "bb", "c", "cc"},
	}

	// Expected ordering of the SET should be:
	// a, b, c, aa, bb, cc

	output, err := Marshal(testStruct)
	if err != nil {
		t.Errorf("%v", err)
	}

	expectedOrder := []string{"a", "b", "c", "aa", "bb", "cc"}
	var resultStruct struct {
		Strings []string `asn1:"set"`
	}
	rest, err := Unmarshal(output, &resultStruct)
	if err != nil {
		t.Errorf("%v", err)
	}
	if len(rest) != 0 {
		t.Error("Unmarshal returned extra garbage")
	}
	if !reflect.DeepEqual(expectedOrder, resultStruct.Strings) {
		t.Errorf("Unexpected SET content. got: %s, want: %s", resultStruct.Strings, expectedOrder)
	}
}

func TestSetEncoderSETSliceSuffix(t *testing.T) {
	type testSetSET []string
	testSet := testSetSET{"a", "aa", "b", "bb", "c", "cc"}

	// Expected ordering of the SET should be:
	// a, b, c, aa, bb, cc

	output, err := Marshal(testSet)
	if err != nil {
		t.Errorf("%v", err)
	}

	expectedOrder := testSetSET{"a", "b", "c", "aa", "bb", "cc"}
	var resultSet testSetSET
	rest, err := Unmarshal(output, &resultSet)
	if err != nil {
		t.Errorf("%v", err)
	}
	if len(rest) != 0 {
		t.Error("Unmarshal returned extra garbage")
	}
	if !reflect.DeepEqual(expectedOrder, resultSet) {
		t.Errorf("Unexpected SET content. got: %s, want: %s", resultSet, expectedOrder)
	}
}
