// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptobyte

import (
	"bytes"
	encoding_asn1 "encoding/asn1"
	"math/big"
	"reflect"
	"testing"
	"time"

	"internal/x/crypto/cryptobyte/asn1"
)

type readASN1Test struct {
	name string
	in   []byte
	tag  asn1.Tag
	ok   bool
	out  interface{}
}

var readASN1TestData = []readASN1Test{
	{"valid", []byte{0x30, 2, 1, 2}, 0x30, true, []byte{1, 2}},
	{"truncated", []byte{0x30, 3, 1, 2}, 0x30, false, nil},
	{"zero length of length", []byte{0x30, 0x80}, 0x30, false, nil},
	{"invalid long form length", []byte{0x30, 0x81, 1, 1}, 0x30, false, nil},
	{"non-minimal length", append([]byte{0x30, 0x82, 0, 0x80}, make([]byte, 0x80)...), 0x30, false, nil},
	{"invalid tag", []byte{0xa1, 3, 0x4, 1, 1}, 31, false, nil},
	{"high tag", []byte{0x1f, 0x81, 0x80, 0x01, 2, 1, 2}, 0xff /* actually 0x4001, but tag is uint8 */, false, nil},
}

func TestReadASN1(t *testing.T) {
	for _, test := range readASN1TestData {
		t.Run(test.name, func(t *testing.T) {
			var in, out String = test.in, nil
			ok := in.ReadASN1(&out, test.tag)
			if ok != test.ok || ok && !bytes.Equal(out, test.out.([]byte)) {
				t.Errorf("in.ReadASN1() = %v, want %v; out = %v, want %v", ok, test.ok, out, test.out)
			}
		})
	}
}

func TestReadASN1Optional(t *testing.T) {
	var empty String
	var present bool
	ok := empty.ReadOptionalASN1(nil, &present, 0xa0)
	if !ok || present {
		t.Errorf("empty.ReadOptionalASN1() = %v, want true; present = %v want false", ok, present)
	}

	var in, out String = []byte{0xa1, 3, 0x4, 1, 1}, nil
	ok = in.ReadOptionalASN1(&out, &present, 0xa0)
	if !ok || present {
		t.Errorf("in.ReadOptionalASN1() = %v, want true, present = %v, want false", ok, present)
	}
	ok = in.ReadOptionalASN1(&out, &present, 0xa1)
	wantBytes := []byte{4, 1, 1}
	if !ok || !present || !bytes.Equal(out, wantBytes) {
		t.Errorf("in.ReadOptionalASN1() = %v, want true; present = %v, want true; out = %v, want = %v", ok, present, out, wantBytes)
	}
}

var optionalOctetStringTestData = []struct {
	readASN1Test
	present bool
}{
	{readASN1Test{"empty", []byte{}, 0xa0, true, []byte{}}, false},
	{readASN1Test{"invalid", []byte{0xa1, 3, 0x4, 2, 1}, 0xa1, false, []byte{}}, true},
	{readASN1Test{"missing", []byte{0xa1, 3, 0x4, 1, 1}, 0xa0, true, []byte{}}, false},
	{readASN1Test{"present", []byte{0xa1, 3, 0x4, 1, 1}, 0xa1, true, []byte{1}}, true},
}

func TestReadASN1OptionalOctetString(t *testing.T) {
	for _, test := range optionalOctetStringTestData {
		t.Run(test.name, func(t *testing.T) {
			in := String(test.in)
			var out []byte
			var present bool
			ok := in.ReadOptionalASN1OctetString(&out, &present, test.tag)
			if ok != test.ok || present != test.present || !bytes.Equal(out, test.out.([]byte)) {
				t.Errorf("in.ReadOptionalASN1OctetString() = %v, want %v; present = %v want %v; out = %v, want %v", ok, test.ok, present, test.present, out, test.out)
			}
		})
	}
}

const defaultInt = -1

var optionalIntTestData = []readASN1Test{
	{"empty", []byte{}, 0xa0, true, defaultInt},
	{"invalid", []byte{0xa1, 3, 0x2, 2, 127}, 0xa1, false, 0},
	{"missing", []byte{0xa1, 3, 0x2, 1, 127}, 0xa0, true, defaultInt},
	{"present", []byte{0xa1, 3, 0x2, 1, 42}, 0xa1, true, 42},
}

func TestReadASN1OptionalInteger(t *testing.T) {
	for _, test := range optionalIntTestData {
		t.Run(test.name, func(t *testing.T) {
			in := String(test.in)
			var out int
			ok := in.ReadOptionalASN1Integer(&out, test.tag, defaultInt)
			if ok != test.ok || ok && out != test.out.(int) {
				t.Errorf("in.ReadOptionalASN1Integer() = %v, want %v; out = %v, want %v", ok, test.ok, out, test.out)
			}
		})
	}
}

func TestReadASN1IntegerSigned(t *testing.T) {
	testData64 := []struct {
		in  []byte
		out int64
	}{
		{[]byte{2, 3, 128, 0, 0}, -0x800000},
		{[]byte{2, 2, 255, 0}, -256},
		{[]byte{2, 2, 255, 127}, -129},
		{[]byte{2, 1, 128}, -128},
		{[]byte{2, 1, 255}, -1},
		{[]byte{2, 1, 0}, 0},
		{[]byte{2, 1, 1}, 1},
		{[]byte{2, 1, 2}, 2},
		{[]byte{2, 1, 127}, 127},
		{[]byte{2, 2, 0, 128}, 128},
		{[]byte{2, 2, 1, 0}, 256},
		{[]byte{2, 4, 0, 128, 0, 0}, 0x800000},
	}
	for i, test := range testData64 {
		in := String(test.in)
		var out int64
		ok := in.ReadASN1Integer(&out)
		if !ok || out != test.out {
			t.Errorf("#%d: in.ReadASN1Integer() = %v, want true; out = %d, want %d", i, ok, out, test.out)
		}
	}

	// Repeat the same cases, reading into a big.Int.
	t.Run("big.Int", func(t *testing.T) {
		for i, test := range testData64 {
			in := String(test.in)
			var out big.Int
			ok := in.ReadASN1Integer(&out)
			if !ok || out.Int64() != test.out {
				t.Errorf("#%d: in.ReadASN1Integer() = %v, want true; out = %d, want %d", i, ok, out.Int64(), test.out)
			}
		}
	})

	// Repeat with the implicit-tagging functions
	t.Run("WithTag", func(t *testing.T) {
		for i, test := range testData64 {
			tag := asn1.Tag((i * 3) % 32).ContextSpecific()

			testData := make([]byte, len(test.in))
			copy(testData, test.in)

			// Alter the tag of the test case.
			testData[0] = uint8(tag)

			in := String(testData)
			var out int64
			ok := in.ReadASN1Int64WithTag(&out, tag)
			if !ok || out != test.out {
				t.Errorf("#%d: in.ReadASN1Int64WithTag() = %v, want true; out = %d, want %d", i, ok, out, test.out)
			}

			var b Builder
			b.AddASN1Int64WithTag(test.out, tag)
			result, err := b.Bytes()

			if err != nil {
				t.Errorf("#%d: AddASN1Int64WithTag failed: %s", i, err)
				continue
			}

			if !bytes.Equal(result, testData) {
				t.Errorf("#%d: AddASN1Int64WithTag: got %x, want %x", i, result, testData)
			}
		}
	})
}

func TestReadASN1IntegerUnsigned(t *testing.T) {
	testData := []struct {
		in  []byte
		out uint64
	}{
		{[]byte{2, 1, 0}, 0},
		{[]byte{2, 1, 1}, 1},
		{[]byte{2, 1, 2}, 2},
		{[]byte{2, 1, 127}, 127},
		{[]byte{2, 2, 0, 128}, 128},
		{[]byte{2, 2, 1, 0}, 256},
		{[]byte{2, 4, 0, 128, 0, 0}, 0x800000},
		{[]byte{2, 8, 127, 255, 255, 255, 255, 255, 255, 255}, 0x7fffffffffffffff},
		{[]byte{2, 9, 0, 128, 0, 0, 0, 0, 0, 0, 0}, 0x8000000000000000},
		{[]byte{2, 9, 0, 255, 255, 255, 255, 255, 255, 255, 255}, 0xffffffffffffffff},
	}
	for i, test := range testData {
		in := String(test.in)
		var out uint64
		ok := in.ReadASN1Integer(&out)
		if !ok || out != test.out {
			t.Errorf("#%d: in.ReadASN1Integer() = %v, want true; out = %d, want %d", i, ok, out, test.out)
		}
	}
}

func TestReadASN1IntegerInvalid(t *testing.T) {
	testData := []String{
		[]byte{3, 1, 0}, // invalid tag
		// truncated
		[]byte{2, 1},
		[]byte{2, 2, 0},
		// not minimally encoded
		[]byte{2, 2, 0, 1},
		[]byte{2, 2, 0xff, 0xff},
	}

	for i, test := range testData {
		var out int64
		if test.ReadASN1Integer(&out) {
			t.Errorf("#%d: in.ReadASN1Integer() = true, want false (out = %d)", i, out)
		}
	}
}

func TestASN1ObjectIdentifier(t *testing.T) {
	testData := []struct {
		in  []byte
		ok  bool
		out []int
	}{
		{[]byte{}, false, []int{}},
		{[]byte{6, 0}, false, []int{}},
		{[]byte{5, 1, 85}, false, []int{2, 5}},
		{[]byte{6, 1, 85}, true, []int{2, 5}},
		{[]byte{6, 2, 85, 0x02}, true, []int{2, 5, 2}},
		{[]byte{6, 4, 85, 0x02, 0xc0, 0x00}, true, []int{2, 5, 2, 0x2000}},
		{[]byte{6, 3, 0x81, 0x34, 0x03}, true, []int{2, 100, 3}},
		{[]byte{6, 7, 85, 0x02, 0xc0, 0x80, 0x80, 0x80, 0x80}, false, []int{}},
	}

	for i, test := range testData {
		in := String(test.in)
		var out encoding_asn1.ObjectIdentifier
		ok := in.ReadASN1ObjectIdentifier(&out)
		if ok != test.ok || ok && !out.Equal(test.out) {
			t.Errorf("#%d: in.ReadASN1ObjectIdentifier() = %v, want %v; out = %v, want %v", i, ok, test.ok, out, test.out)
			continue
		}

		var b Builder
		b.AddASN1ObjectIdentifier(out)
		result, err := b.Bytes()
		if builderOk := err == nil; test.ok != builderOk {
			t.Errorf("#%d: error from Builder.Bytes: %s", i, err)
			continue
		}
		if test.ok && !bytes.Equal(result, test.in) {
			t.Errorf("#%d: reserialisation didn't match, got %x, want %x", i, result, test.in)
			continue
		}
	}
}

func TestReadASN1GeneralizedTime(t *testing.T) {
	testData := []struct {
		in  string
		ok  bool
		out time.Time
	}{
		{"20100102030405Z", true, time.Date(2010, 01, 02, 03, 04, 05, 0, time.UTC)},
		{"20100102030405", false, time.Time{}},
		{"20100102030405+0607", true, time.Date(2010, 01, 02, 03, 04, 05, 0, time.FixedZone("", 6*60*60+7*60))},
		{"20100102030405-0607", true, time.Date(2010, 01, 02, 03, 04, 05, 0, time.FixedZone("", -6*60*60-7*60))},
		/* These are invalid times. However, the time package normalises times
		 * and they were accepted in some versions. See #11134. */
		{"00000100000000Z", false, time.Time{}},
		{"20101302030405Z", false, time.Time{}},
		{"20100002030405Z", false, time.Time{}},
		{"20100100030405Z", false, time.Time{}},
		{"20100132030405Z", false, time.Time{}},
		{"20100231030405Z", false, time.Time{}},
		{"20100102240405Z", false, time.Time{}},
		{"20100102036005Z", false, time.Time{}},
		{"20100102030460Z", false, time.Time{}},
		{"-20100102030410Z", false, time.Time{}},
		{"2010-0102030410Z", false, time.Time{}},
		{"2010-0002030410Z", false, time.Time{}},
		{"201001-02030410Z", false, time.Time{}},
		{"20100102-030410Z", false, time.Time{}},
		{"2010010203-0410Z", false, time.Time{}},
		{"201001020304-10Z", false, time.Time{}},
	}
	for i, test := range testData {
		in := String(append([]byte{byte(asn1.GeneralizedTime), byte(len(test.in))}, test.in...))
		var out time.Time
		ok := in.ReadASN1GeneralizedTime(&out)
		if ok != test.ok || ok && !reflect.DeepEqual(out, test.out) {
			t.Errorf("#%d: in.ReadASN1GeneralizedTime() = %v, want %v; out = %q, want %q", i, ok, test.ok, out, test.out)
		}
	}
}

func TestReadASN1BitString(t *testing.T) {
	testData := []struct {
		in  []byte
		ok  bool
		out encoding_asn1.BitString
	}{
		{[]byte{}, false, encoding_asn1.BitString{}},
		{[]byte{0x00}, true, encoding_asn1.BitString{}},
		{[]byte{0x07, 0x00}, true, encoding_asn1.BitString{Bytes: []byte{0}, BitLength: 1}},
		{[]byte{0x07, 0x01}, false, encoding_asn1.BitString{}},
		{[]byte{0x07, 0x40}, false, encoding_asn1.BitString{}},
		{[]byte{0x08, 0x00}, false, encoding_asn1.BitString{}},
		{[]byte{0xff}, false, encoding_asn1.BitString{}},
		{[]byte{0xfe, 0x00}, false, encoding_asn1.BitString{}},
	}
	for i, test := range testData {
		in := String(append([]byte{3, byte(len(test.in))}, test.in...))
		var out encoding_asn1.BitString
		ok := in.ReadASN1BitString(&out)
		if ok != test.ok || ok && (!bytes.Equal(out.Bytes, test.out.Bytes) || out.BitLength != test.out.BitLength) {
			t.Errorf("#%d: in.ReadASN1BitString() = %v, want %v; out = %v, want %v", i, ok, test.ok, out, test.out)
		}
	}
}
