// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"reflect"
	"testing"
)

type Optionals struct {
	Sr string `json:"sr"`
	So string `json:"so,omitempty"`
	Sw string `json:"-"`

	Ir int `json:"omitempty"` // actually named omitempty, not an option
	Io int `json:"io,omitempty"`

	Slr []string `json:"slr,random"`
	Slo []string `json:"slo,omitempty"`

	Mr map[string]interface{} `json:"mr"`
	Mo map[string]interface{} `json:",omitempty"`
}

var optionalsExpected = `{
 "sr": "",
 "omitempty": 0,
 "slr": [],
 "mr": {}
}`

func TestOmitEmpty(t *testing.T) {
	var o Optionals
	o.Sw = "something"
	o.Mr = map[string]interface{}{}
	o.Mo = map[string]interface{}{}

	got, err := MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != optionalsExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, optionalsExpected)
	}
}

type StringTag struct {
	BoolStr bool   `json:",string"`
	IntStr  int64  `json:",string"`
	StrStr  string `json:",string"`
}

var stringTagExpected = `{
 "BoolStr": "true",
 "IntStr": "42",
 "StrStr": "\"xzbit\""
}`

func TestStringTag(t *testing.T) {
	var s StringTag
	s.BoolStr = true
	s.IntStr = 42
	s.StrStr = "xzbit"
	got, err := MarshalIndent(&s, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != stringTagExpected {
		t.Fatalf(" got: %s\nwant: %s\n", got, stringTagExpected)
	}

	// Verify that it round-trips.
	var s2 StringTag
	err = NewDecoder(bytes.NewBuffer(got)).Decode(&s2)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if !reflect.DeepEqual(s, s2) {
		t.Fatalf("decode didn't match.\nsource: %#v\nEncoded as:\n%s\ndecode: %#v", s, string(got), s2)
	}
}
