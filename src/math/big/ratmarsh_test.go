// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"encoding/xml"
	"testing"
)

func TestRatGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for _, test := range encodingTests {
		medium.Reset() // empty buffer for each test case (in case of failures)
		var tx Rat
		tx.SetString(test + ".14159265")
		if err := enc.Encode(&tx); err != nil {
			t.Errorf("encoding of %s failed: %s", &tx, err)
			continue
		}
		var rx Rat
		if err := dec.Decode(&rx); err != nil {
			t.Errorf("decoding of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("transmission of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

// Sending a nil Rat pointer (inside a slice) on a round trip through gob should yield a zero.
// TODO: top-level nils.
func TestGobEncodingNilRatInSlice(t *testing.T) {
	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	dec := gob.NewDecoder(buf)

	var in = make([]*Rat, 1)
	err := enc.Encode(&in)
	if err != nil {
		t.Errorf("gob encode failed: %q", err)
	}
	var out []*Rat
	err = dec.Decode(&out)
	if err != nil {
		t.Fatalf("gob decode failed: %q", err)
	}
	if len(out) != 1 {
		t.Fatalf("wrong len; want 1 got %d", len(out))
	}
	var zero Rat
	if out[0].Cmp(&zero) != 0 {
		t.Fatalf("transmission of (*Int)(nil) failed: got %s want 0", out)
	}
}

var ratNums = []string{
	"-141592653589793238462643383279502884197169399375105820974944592307816406286",
	"-1415926535897932384626433832795028841971",
	"-141592653589793",
	"-1",
	"0",
	"1",
	"141592653589793",
	"1415926535897932384626433832795028841971",
	"141592653589793238462643383279502884197169399375105820974944592307816406286",
}

var ratDenoms = []string{
	"1",
	"718281828459045",
	"7182818284590452353602874713526624977572",
	"718281828459045235360287471352662497757247093699959574966967627724076630353",
}

func TestRatJSONEncoding(t *testing.T) {
	for _, num := range ratNums {
		for _, denom := range ratDenoms {
			var tx Rat
			tx.SetString(num + "/" + denom)
			b, err := json.Marshal(&tx)
			if err != nil {
				t.Errorf("marshaling of %s failed: %s", &tx, err)
				continue
			}
			var rx Rat
			if err := json.Unmarshal(b, &rx); err != nil {
				t.Errorf("unmarshaling of %s failed: %s", &tx, err)
				continue
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
			}
		}
	}
}

func TestRatXMLEncoding(t *testing.T) {
	for _, num := range ratNums {
		for _, denom := range ratDenoms {
			var tx Rat
			tx.SetString(num + "/" + denom)
			b, err := xml.Marshal(&tx)
			if err != nil {
				t.Errorf("marshaling of %s failed: %s", &tx, err)
				continue
			}
			var rx Rat
			if err := xml.Unmarshal(b, &rx); err != nil {
				t.Errorf("unmarshaling of %s failed: %s", &tx, err)
				continue
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("XML encoding of %s failed: got %s want %s", &tx, &rx, &tx)
			}
		}
	}
}

func TestRatGobDecodeShortBuffer(t *testing.T) {
	for _, tc := range [][]byte{
		[]byte{0x2},
		[]byte{0x2, 0x0, 0x0, 0x0, 0xff},
		[]byte{0x2, 0xff, 0xff, 0xff, 0xff},
	} {
		err := NewRat(1, 2).GobDecode(tc)
		if err == nil {
			t.Error("expected GobDecode to return error for malformed input")
		}
	}
}
