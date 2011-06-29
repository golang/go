// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import "testing"

type C struct {
	Name string
	Open bool
}

type A struct {
	XMLName Name `xml:"http://domain a"`
	C
	B      B
	FieldA string
}

type B struct {
	XMLName Name `xml:"b"`
	C
	FieldB string
}

const _1a = `
<?xml version="1.0" encoding="UTF-8"?>
<a xmlns="http://domain">
  <name>KmlFile</name>
  <open>1</open>
  <b>
    <name>Absolute</name>
    <open>0</open>
    <fieldb>bar</fieldb>
  </b>
  <fielda>foo</fielda>
</a>
`

// Tests that embedded structs are marshalled.
func TestEmbedded1(t *testing.T) {
	var a A
	if e := Unmarshal(StringReader(_1a), &a); e != nil {
		t.Fatalf("Unmarshal: %s", e)
	}
	if a.FieldA != "foo" {
		t.Fatalf("Unmarshal: expected 'foo' but found '%s'", a.FieldA)
	}
	if a.Name != "KmlFile" {
		t.Fatalf("Unmarshal: expected 'KmlFile' but found '%s'", a.Name)
	}
	if !a.Open {
		t.Fatal("Unmarshal: expected 'true' but found otherwise")
	}
	if a.B.FieldB != "bar" {
		t.Fatalf("Unmarshal: expected 'bar' but found '%s'", a.B.FieldB)
	}
	if a.B.Name != "Absolute" {
		t.Fatalf("Unmarshal: expected 'Absolute' but found '%s'", a.B.Name)
	}
	if a.B.Open {
		t.Fatal("Unmarshal: expected 'false' but found otherwise")
	}
}

type A2 struct {
	XMLName Name `xml:"http://domain a"`
	XY      string
	Xy      string
}

const _2a = `
<?xml version="1.0" encoding="UTF-8"?>
<a xmlns="http://domain">
  <xy>foo</xy>
</a>
`

// Tests that conflicting field names get excluded.
func TestEmbedded2(t *testing.T) {
	var a A2
	if e := Unmarshal(StringReader(_2a), &a); e != nil {
		t.Fatalf("Unmarshal: %s", e)
	}
	if a.XY != "" {
		t.Fatalf("Unmarshal: expected empty string but found '%s'", a.XY)
	}
	if a.Xy != "" {
		t.Fatalf("Unmarshal: expected empty string but found '%s'", a.Xy)
	}
}

type A3 struct {
	XMLName Name `xml:"http://domain a"`
	xy      string
}

// Tests that private fields are not set.
func TestEmbedded3(t *testing.T) {
	var a A3
	if e := Unmarshal(StringReader(_2a), &a); e != nil {
		t.Fatalf("Unmarshal: %s", e)
	}
	if a.xy != "" {
		t.Fatalf("Unmarshal: expected empty string but found '%s'", a.xy)
	}
}

type A4 struct {
	XMLName Name `xml:"http://domain a"`
	Any     string
}

// Tests that private fields are not set.
func TestEmbedded4(t *testing.T) {
	var a A4
	if e := Unmarshal(StringReader(_2a), &a); e != nil {
		t.Fatalf("Unmarshal: %s", e)
	}
	if a.Any != "foo" {
		t.Fatalf("Unmarshal: expected 'foo' but found '%s'", a.Any)
	}
}
