// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding"
	"encoding/asn1"
	"math"
	"testing"
)

var oidTests = []struct {
	raw   []byte
	valid bool
	str   string
	ints  []uint64
}{
	{[]byte{}, false, "", nil},
	{[]byte{0x80, 0x01}, false, "", nil},
	{[]byte{0x01, 0x80, 0x01}, false, "", nil},

	{[]byte{1, 2, 3}, true, "0.1.2.3", []uint64{0, 1, 2, 3}},
	{[]byte{41, 2, 3}, true, "1.1.2.3", []uint64{1, 1, 2, 3}},
	{[]byte{86, 2, 3}, true, "2.6.2.3", []uint64{2, 6, 2, 3}},

	{[]byte{41, 255, 255, 255, 127}, true, "1.1.268435455", []uint64{1, 1, 268435455}},
	{[]byte{41, 0x87, 255, 255, 255, 127}, true, "1.1.2147483647", []uint64{1, 1, 2147483647}},
	{[]byte{41, 255, 255, 255, 255, 127}, true, "1.1.34359738367", []uint64{1, 1, 34359738367}},
	{[]byte{42, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "1.2.9223372036854775807", []uint64{1, 2, 9223372036854775807}},
	{[]byte{43, 0x81, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "1.3.18446744073709551615", []uint64{1, 3, 18446744073709551615}},
	{[]byte{44, 0x83, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "1.4.36893488147419103231", nil},
	{[]byte{85, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.5.1180591620717411303423", nil},
	{[]byte{85, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.5.19342813113834066795298815", nil},

	{[]byte{255, 255, 255, 127}, true, "2.268435375", []uint64{2, 268435375}},
	{[]byte{0x87, 255, 255, 255, 127}, true, "2.2147483567", []uint64{2, 2147483567}},
	{[]byte{255, 127}, true, "2.16303", []uint64{2, 16303}},
	{[]byte{255, 255, 255, 255, 127}, true, "2.34359738287", []uint64{2, 34359738287}},
	{[]byte{255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.9223372036854775727", []uint64{2, 9223372036854775727}},
	{[]byte{0x81, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.18446744073709551535", []uint64{2, 18446744073709551535}},
	{[]byte{0x83, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.36893488147419103151", nil},
	{[]byte{255, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.1180591620717411303343", nil},
	{[]byte{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127}, true, "2.19342813113834066795298735", nil},

	{[]byte{41, 0x80 | 66, 0x80 | 44, 0x80 | 11, 33}, true, "1.1.139134369", []uint64{1, 1, 139134369}},
	{[]byte{0x80 | 66, 0x80 | 44, 0x80 | 11, 33}, true, "2.139134289", []uint64{2, 139134289}},
}

func TestOID(t *testing.T) {
	for _, v := range oidTests {
		oid, ok := newOIDFromDER(v.raw)
		if ok != v.valid {
			t.Errorf("newOIDFromDER(%v) = (%v, %v); want = (OID, %v)", v.raw, oid, ok, v.valid)
			continue
		}

		if !ok {
			continue
		}

		if str := oid.String(); str != v.str {
			t.Errorf("(%#v).String() = %v, want; %v", oid, str, v.str)
		}

		var asn1OID asn1.ObjectIdentifier
		for _, v := range v.ints {
			if v > math.MaxInt32 {
				asn1OID = nil
				break
			}
			asn1OID = append(asn1OID, int(v))
		}

		o, ok := oid.toASN1OID()
		if shouldOk := asn1OID != nil; shouldOk != ok {
			t.Errorf("(%#v).toASN1OID() = (%v, %v); want = (%v, %v)", oid, o, ok, asn1OID, shouldOk)
			continue
		}

		if asn1OID != nil && !o.Equal(asn1OID) {
			t.Errorf("(%#v).toASN1OID() = (%v, true); want = (%v, true)", oid, o, asn1OID)
		}

		if v.ints != nil {
			oid2, err := OIDFromInts(v.ints)
			if err != nil {
				t.Errorf("OIDFromInts(%v) = (%v, %v); want = (%v, nil)", v.ints, oid2, err, oid)
			}
			if !oid2.Equal(oid) {
				t.Errorf("OIDFromInts(%v) = (%v, nil); want = (%v, nil)", v.ints, oid2, oid)
			}
		}
	}
}

func TestInvalidOID(t *testing.T) {
	cases := []struct {
		str  string
		ints []uint64
	}{
		{str: "", ints: []uint64{}},
		{str: "1", ints: []uint64{1}},
		{str: "3", ints: []uint64{3}},
		{str: "3.100.200", ints: []uint64{3, 100, 200}},
		{str: "1.81", ints: []uint64{1, 81}},
		{str: "1.81.200", ints: []uint64{1, 81, 200}},
	}

	for _, tt := range cases {
		oid, err := OIDFromInts(tt.ints)
		if err == nil {
			t.Errorf("OIDFromInts(%v) = (%v, %v); want = (OID{}, %v)", tt.ints, oid, err, errInvalidOID)
		}

		oid2, err := ParseOID(tt.str)
		if err == nil {
			t.Errorf("ParseOID(%v) = (%v, %v); want = (OID{}, %v)", tt.str, oid2, err, errInvalidOID)
		}

		var oid3 OID
		err = oid3.UnmarshalText([]byte(tt.str))
		if err == nil {
			t.Errorf("(*OID).UnmarshalText(%v) = (%v, %v); want = (OID{}, %v)", tt.str, oid3, err, errInvalidOID)
		}
	}
}

func TestOIDEqual(t *testing.T) {
	var cases = []struct {
		oid  OID
		oid2 OID
		eq   bool
	}{
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: mustNewOIDFromInts(t, []uint64{1, 2, 3}), eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: mustNewOIDFromInts(t, []uint64{1, 2, 4}), eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: mustNewOIDFromInts(t, []uint64{1, 2, 3, 4}), eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{2, 33, 22}), oid2: mustNewOIDFromInts(t, []uint64{2, 33, 23}), eq: false},
		{oid: OID{}, oid2: OID{}, eq: true},
		{oid: OID{}, oid2: mustNewOIDFromInts(t, []uint64{2, 33, 23}), eq: false},
	}

	for _, tt := range cases {
		if eq := tt.oid.Equal(tt.oid2); eq != tt.eq {
			t.Errorf("(%v).Equal(%v) = %v, want %v", tt.oid, tt.oid2, eq, tt.eq)
		}
	}
}

var (
	_ encoding.BinaryMarshaler   = OID{}
	_ encoding.BinaryUnmarshaler = new(OID)
	_ encoding.TextMarshaler     = OID{}
	_ encoding.TextUnmarshaler   = new(OID)
)

func TestOIDMarshal(t *testing.T) {
	cases := []struct {
		in  string
		out OID
		err error
	}{
		{in: "", err: errInvalidOID},
		{in: "0", err: errInvalidOID},
		{in: "1", err: errInvalidOID},
		{in: ".1", err: errInvalidOID},
		{in: ".1.", err: errInvalidOID},
		{in: "1.", err: errInvalidOID},
		{in: "1..", err: errInvalidOID},
		{in: "1.2.", err: errInvalidOID},
		{in: "1.2.333.", err: errInvalidOID},
		{in: "1.2.333..", err: errInvalidOID},
		{in: "1.2..", err: errInvalidOID},
		{in: "+1.2", err: errInvalidOID},
		{in: "-1.2", err: errInvalidOID},
		{in: "1.-2", err: errInvalidOID},
		{in: "1.2.+333", err: errInvalidOID},
	}

	for _, v := range oidTests {
		oid, ok := newOIDFromDER(v.raw)
		if !ok {
			continue
		}
		cases = append(cases, struct {
			in  string
			out OID
			err error
		}{
			in:  v.str,
			out: oid,
			err: nil,
		})
	}

	for _, tt := range cases {
		o, err := ParseOID(tt.in)
		if err != tt.err {
			t.Errorf("ParseOID(%q) = %v; want = %v", tt.in, err, tt.err)
			continue
		}

		var o2 OID
		err = o2.UnmarshalText([]byte(tt.in))
		if err != tt.err {
			t.Errorf("(*OID).UnmarshalText(%q) = %v; want = %v", tt.in, err, tt.err)
			continue
		}

		if err != nil {
			continue
		}

		if !o.Equal(tt.out) {
			t.Errorf("(*OID).UnmarshalText(%q) = %v; want = %v", tt.in, o, tt.out)
			continue
		}

		if !o2.Equal(tt.out) {
			t.Errorf("ParseOID(%q) = %v; want = %v", tt.in, o2, tt.out)
			continue
		}

		marshalled, err := o.MarshalText()
		if string(marshalled) != tt.in || err != nil {
			t.Errorf("(%#v).MarshalText() = (%v, %v); want = (%v, nil)", o, string(marshalled), err, tt.in)
			continue
		}

		binary, err := o.MarshalBinary()
		if err != nil {
			t.Errorf("(%#v).MarshalBinary() = %v; want = nil", o, err)
		}

		var o3 OID
		if err := o3.UnmarshalBinary(binary); err != nil {
			t.Errorf("(*OID).UnmarshalBinary(%v) = %v; want = nil", binary, err)
		}

		if !o3.Equal(tt.out) {
			t.Errorf("(*OID).UnmarshalBinary(%v) = %v; want = %v", binary, o3, tt.out)
			continue
		}
	}
}

func TestOIDEqualASN1OID(t *testing.T) {
	maxInt32PlusOne := int64(math.MaxInt32) + 1
	var cases = []struct {
		oid  OID
		oid2 asn1.ObjectIdentifier
		eq   bool
	}{
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: asn1.ObjectIdentifier{1, 2, 3}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: asn1.ObjectIdentifier{1, 2, 4}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 2, 3}), oid2: asn1.ObjectIdentifier{1, 2, 3, 4}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 22}), oid2: asn1.ObjectIdentifier{1, 33, 23}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 23}), oid2: asn1.ObjectIdentifier{1, 33, 22}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 127}), oid2: asn1.ObjectIdentifier{1, 33, 127}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 128}), oid2: asn1.ObjectIdentifier{1, 33, 127}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 128}), oid2: asn1.ObjectIdentifier{1, 33, 128}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 129}), oid2: asn1.ObjectIdentifier{1, 33, 129}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 128}), oid2: asn1.ObjectIdentifier{1, 33, 129}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 129}), oid2: asn1.ObjectIdentifier{1, 33, 128}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 255}), oid2: asn1.ObjectIdentifier{1, 33, 255}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 256}), oid2: asn1.ObjectIdentifier{1, 33, 256}, eq: true},
		{oid: mustNewOIDFromInts(t, []uint64{2, 33, 257}), oid2: asn1.ObjectIdentifier{2, 33, 256}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{2, 33, 256}), oid2: asn1.ObjectIdentifier{2, 33, 257}, eq: false},

		{oid: mustNewOIDFromInts(t, []uint64{1, 33}), oid2: asn1.ObjectIdentifier{1, 33, math.MaxInt32}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, math.MaxInt32}), oid2: asn1.ObjectIdentifier{1, 33}, eq: false},
		{oid: mustNewOIDFromInts(t, []uint64{1, 33, math.MaxInt32}), oid2: asn1.ObjectIdentifier{1, 33, math.MaxInt32}, eq: true},
		{
			oid:  mustNewOIDFromInts(t, []uint64{1, 33, math.MaxInt32 + 1}),
			oid2: asn1.ObjectIdentifier{1, 33 /*convert to int, so that it compiles on 32bit*/, int(maxInt32PlusOne)},
			eq:   false,
		},

		{oid: mustNewOIDFromInts(t, []uint64{1, 33, 256}), oid2: asn1.ObjectIdentifier{}, eq: false},
		{oid: OID{}, oid2: asn1.ObjectIdentifier{1, 33, 256}, eq: false},
		{oid: OID{}, oid2: asn1.ObjectIdentifier{}, eq: false},
	}

	for _, tt := range cases {
		if eq := tt.oid.EqualASN1OID(tt.oid2); eq != tt.eq {
			t.Errorf("(%v).EqualASN1OID(%v) = %v, want %v", tt.oid, tt.oid2, eq, tt.eq)
		}
	}
}

func TestOIDUnmarshalBinary(t *testing.T) {
	for _, tt := range oidTests {
		var o OID
		err := o.UnmarshalBinary(tt.raw)

		expectErr := errInvalidOID
		if tt.valid {
			expectErr = nil
		}

		if err != expectErr {
			t.Errorf("(o *OID).UnmarshalBinary(%v) = %v; want = %v; (o = %v)", tt.raw, err, expectErr, o)
		}
	}
}

func BenchmarkOIDMarshalUnmarshalText(b *testing.B) {
	oid := mustNewOIDFromInts(b, []uint64{1, 2, 3, 9999, 1024})
	for range b.N {
		text, err := oid.MarshalText()
		if err != nil {
			b.Fatal(err)
		}
		var o OID
		if err := o.UnmarshalText(text); err != nil {
			b.Fatal(err)
		}
	}
}

func mustNewOIDFromInts(t testing.TB, ints []uint64) OID {
	oid, err := OIDFromInts(ints)
	if err != nil {
		t.Fatalf("OIDFromInts(%v) unexpected error: %v", ints, err)
	}
	return oid
}
