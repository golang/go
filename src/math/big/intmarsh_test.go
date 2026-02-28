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
	"0",
	"1",
	"2",
	"10",
	"1000",
	"1234567890",
	"298472983472983471903246121093472394872319615612417471234712061",
}

func TestIntGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for _, test := range encodingTests {
		for _, sign := range []string{"", "+", "-"} {
			x := sign + test
			medium.Reset() // empty buffer for each test case (in case of failures)
			var tx Int
			tx.SetString(x, 10)
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
		for _, sign := range []string{"", "+", "-"} {
			x := sign + test
			var tx Int
			tx.SetString(x, 10)
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
}

func TestIntJSONEncodingNil(t *testing.T) {
	var x *Int
	b, err := x.MarshalJSON()
	if err != nil {
		t.Fatalf("marshaling of nil failed: %s", err)
	}
	got := string(b)
	want := "null"
	if got != want {
		t.Fatalf("marshaling of nil failed: got %s want %s", got, want)
	}
}

func TestIntXMLEncoding(t *testing.T) {
	for _, test := range encodingTests {
		for _, sign := range []string{"", "+", "-"} {
			x := sign + test
			var tx Int
			tx.SetString(x, 0)
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
}

func TestIntAppendText(t *testing.T) {
	for _, test := range encodingTests {
		for _, sign := range []string{"", "+", "-"} {
			x := sign + test
			var tx Int
			tx.SetString(x, 10)
			buf := make([]byte, 4, 32)
			b, err := tx.AppendText(buf)
			if err != nil {
				t.Errorf("marshaling of %s failed: %s", &tx, err)
				continue
			}
			var rx Int
			if err := rx.UnmarshalText(b[4:]); err != nil {
				t.Errorf("unmarshaling of %s failed: %s", &tx, err)
				continue
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("AppendText of %s failed: got %s want %s", &tx, &rx, &tx)
			}
		}
	}
}

func TestIntAppendTextNil(t *testing.T) {
	var x *Int
	buf := make([]byte, 4, 16)
	data, _ := x.AppendText(buf)
	if string(data[4:]) != "<nil>" {
		t.Errorf("got %q, want <nil>", data[4:])
	}
}
