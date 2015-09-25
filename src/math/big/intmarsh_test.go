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

var encodingTests = []string{
	"-539345864568634858364538753846587364875430589374589",
	"-678645873",
	"-100",
	"-2",
	"-1",
	"0",
	"1",
	"2",
	"10",
	"42",
	"1234567890",
	"298472983472983471903246121093472394872319615612417471234712061",
}

func TestIntGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for _, test := range encodingTests {
		medium.Reset() // empty buffer for each test case (in case of failures)
		var tx Int
		tx.SetString(test, 10)
		if err := enc.Encode(&tx); err != nil {
			t.Errorf("encoding of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := dec.Decode(&rx); err != nil {
			t.Errorf("decoding of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("transmission of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

// Sending a nil Int pointer (inside a slice) on a round trip through gob should yield a zero.
// TODO: top-level nils.
func TestGobEncodingNilIntInSlice(t *testing.T) {
	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	dec := gob.NewDecoder(buf)

	var in = make([]*Int, 1)
	err := enc.Encode(&in)
	if err != nil {
		t.Errorf("gob encode failed: %q", err)
	}
	var out []*Int
	err = dec.Decode(&out)
	if err != nil {
		t.Fatalf("gob decode failed: %q", err)
	}
	if len(out) != 1 {
		t.Fatalf("wrong len; want 1 got %d", len(out))
	}
	var zero Int
	if out[0].Cmp(&zero) != 0 {
		t.Fatalf("transmission of (*Int)(nil) failed: got %s want 0", out)
	}
}

func TestIntJSONEncoding(t *testing.T) {
	for _, test := range encodingTests {
		var tx Int
		tx.SetString(test, 10)
		b, err := json.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := json.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

var intVals = []string{
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

func TestIntJSONEncodingTextMarshaller(t *testing.T) {
	for _, num := range intVals {
		var tx Int
		tx.SetString(num, 0)
		b, err := json.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := json.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

func TestIntXMLEncodingTextMarshaller(t *testing.T) {
	for _, num := range intVals {
		var tx Int
		tx.SetString(num, 0)
		b, err := xml.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := xml.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("XML encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}
