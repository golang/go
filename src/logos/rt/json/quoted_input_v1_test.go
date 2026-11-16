// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	stdjson "encoding/json"
	"reflect"
	"testing"
)

type quotedMutableByte byte

func (p *quotedMutableByte) UnmarshalJSON(b []byte) error {
	*p = quotedMutableByte(b[0])
	b[0] = 'x'
	return nil
}

type quotedMutableTarget struct {
	Value   quotedMutableByte  `json:"value,string"`
	Pointer *quotedMutableByte `json:"pointer,string"`
}

func TestQuotedJSONMethodInputOwnership(t *testing.T) {
	var saved [2]string
	if err := stdjson.Unmarshal([]byte(`["7","8"]`), &saved); err != nil {
		t.Fatal(err)
	}
	const input = `{"value":"7","pointer":"8"}`
	var got, want quotedMutableTarget
	if err := stdjson.Unmarshal([]byte(input), &want); err != nil {
		t.Fatal(err)
	}
	s := NewScanner([]byte(input), (*quotedMutableTarget)(nil))
	if !s.Object() {
		t.Fatalf("Object: %v", s.End())
	}
	for s.Next() {
		if s.Key() == "value" {
			QuotedJSON(&s, &got.Value)
		} else {
			QuotedJSONPtr(&s, &got.Pointer)
		}
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("decoded = %#v, encoding/json = %#v", got, want)
	}
	if saved != [2]string{"7", "8"} {
		t.Errorf("method input mutation changed independent strings to %q", saved)
	}
}
