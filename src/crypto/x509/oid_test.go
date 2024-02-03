// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding/asn1"
	"math"
	"testing"
)

func TestOID(t *testing.T) {
	var tests = []struct {
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
	}

	for _, v := range tests {
		oid, ok := newOIDFromDER(v.raw)
		if ok != v.valid {
			if ok {
				t.Errorf("%v: unexpected success while parsing: %v", v.raw, oid)
			} else {
				t.Errorf("%v: unexpected failure while parsing", v.raw)
			}
			continue
		}

		if !ok {
			continue
		}

		if str := oid.String(); str != v.str {
			t.Errorf("%v: oid.String() = %v, want; %v", v.raw, str, v.str)
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
			if ok {
				t.Errorf("%v: oid.toASN1OID() unexpected success", v.raw)
			} else {
				t.Errorf("%v: oid.toASN1OID() unexpected failure", v.raw)
			}
			continue
		}

		if asn1OID != nil {
			if !o.Equal(asn1OID) {
				t.Errorf("%v: oid.toASN1OID(asn1OID).Equal(oid) = false, want: true", v.raw)
			}
		}

		if v.ints != nil {
			oid2, err := OIDFromInts(v.ints)
			if err != nil {
				t.Errorf("%v: OIDFromInts() unexpected error: %v", v.raw, err)
			}
			if !oid2.Equal(oid) {
				t.Errorf("%v: %#v.Equal(%#v) = false, want: true", v.raw, oid2, oid)
			}
		}
	}
}

func mustNewOIDFromInts(t *testing.T, ints []uint64) OID {
	oid, err := OIDFromInts(ints)
	if err != nil {
		t.Fatalf("OIDFromInts(%v) unexpected error: %v", ints, err)
	}
	return oid
}
