// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"reflect"
	"testing"
)

// Test values for the stream test.
// One of each JSON kind.
var streamTest = []interface{}{
	0.1,
	"hello",
	nil,
	true,
	false,
	[]interface{}{"a", "b", "c"},
	map[string]interface{}{"K": "Kelvin", "ß": "long s"},
	3.14, // another value to make sure something can follow map
}

var streamEncoded = `0.1
"hello"
null
true
false
["a","b","c"]
{"ß":"long s","K":"Kelvin"}
3.14
`

func TestEncoder(t *testing.T) {
	for i := 0; i <= len(streamTest); i++ {
		var buf bytes.Buffer
		enc := NewEncoder(&buf)
		for j, v := range streamTest[0:i] {
			if err := enc.Encode(v); err != nil {
				t.Fatalf("encode #%d: %v", j, err)
			}
		}
		if have, want := buf.String(), nlines(streamEncoded, i); have != want {
			t.Errorf("encoding %d items: mismatch", i)
			diff(t, []byte(have), []byte(want))
			break
		}
	}
}

func TestDecoder(t *testing.T) {
	for i := 0; i <= len(streamTest); i++ {
		// Use stream without newlines as input,
		// just to stress the decoder even more.
		// Our test input does not include back-to-back numbers.
		// Otherwise stripping the newlines would
		// merge two adjacent JSON values.
		var buf bytes.Buffer
		for _, c := range nlines(streamEncoded, i) {
			if c != '\n' {
				buf.WriteRune(c)
			}
		}
		out := make([]interface{}, i)
		dec := NewDecoder(&buf)
		for j := range out {
			if err := dec.Decode(&out[j]); err != nil {
				t.Fatalf("decode #%d/%d: %v", j, i, err)
			}
		}
		if !reflect.DeepEqual(out, streamTest[0:i]) {
			t.Errorf("decoding %d items: mismatch", i)
			for j := range out {
				if !reflect.DeepEqual(out[j], streamTest[j]) {
					t.Errorf("#%d: have %v want %v", j, out[j], streamTest[j])
				}
			}
			break
		}
	}
}

func nlines(s string, n int) string {
	if n <= 0 {
		return ""
	}
	for i, c := range s {
		if c == '\n' {
			if n--; n == 0 {
				return s[0 : i+1]
			}
		}
	}
	return s
}

func TestRawMessage(t *testing.T) {
	// TODO(rsc): Should not need the * in *RawMessage
	var data struct {
		X  float64
		Id *RawMessage
		Y  float32
	}
	const raw = `["\u0056",null]`
	const msg = `{"X":0.1,"Id":["\u0056",null],"Y":0.2}`
	err := Unmarshal([]byte(msg), &data)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if string([]byte(*data.Id)) != raw {
		t.Fatalf("Raw mismatch: have %#q want %#q", []byte(*data.Id), raw)
	}
	b, err := Marshal(&data)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != msg {
		t.Fatalf("Marshal: have %#q want %#q", b, msg)
	}
}
