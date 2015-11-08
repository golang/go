// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package testdata

type StructTagTest struct {
	A   int "hello"            // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
	B   int "\tx:\"y\""        // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag key"
	C   int "x:\"y\"\tx:\"y\"" // ERROR "not compatible with reflect.StructTag.Get"
	D   int "x:`y`"            // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	E   int "ct\brl:\"char\""  // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
	F   int `:"emptykey"`      // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag key"
	G   int `x:"noEndQuote`    // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	H   int `x:"trunc\x0"`     // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	OK0 int `x:"y" u:"v" w:""`
	OK1 int `x:"y:z" u:"v" w:""` // note multiple colons.
	OK2 int "k0:\"values contain spaces\" k1:\"literal\ttabs\" k2:\"and\\tescaped\\tabs\""
	OK3 int `under_scores:"and" CAPS:"ARE_OK"`
}

type UnexportedEncodingTagTest struct {
	x int `json:"xx"` // ERROR "struct field x has json tag but is not exported"
	y int `xml:"yy"`  // ERROR "struct field y has xml tag but is not exported"
	z int
	A int `json:"aa" xml:"bb"`
}

type unexp struct{}

type JSONEmbeddedField struct {
	UnexportedEncodingTagTest `is:"embedded"`
	unexp                     `is:"embedded,notexported" json:"unexp"` // OK for now, see issue 7363
}

type DuplicateJSONFields struct {
	JSON              int `json:"a"`
	DuplicateJSON     int `json:"a"` // ERROR "struct field DuplicateJSON repeats json tag .a. also at testdata/structtag.go:39"
	IgnoredJSON       int `json:"-"`
	OtherIgnoredJSON  int `json:"-"`
	OmitJSON          int `json:",omitempty"`
	OtherOmitJSON     int `json:",omitempty"`
	DuplicateOmitJSON int `json:"a,omitempty"` // ERROR "struct field DuplicateOmitJSON repeats json tag .a. also at testdata/structtag.go:39"
	NonJSON           int `foo:"a"`
	DuplicateNonJSON  int `foo:"a"`
	Embedded          struct {
		DuplicateJSON int `json:"a"` // OK because its not in the same struct type
	}

	XML              int `xml:"a"`
	DuplicateXML     int `xml:"a"` // ERROR "struct field DuplicateXML repeats xml tag .a. also at testdata/structtag.go:52"
	IgnoredXML       int `xml:"-"`
	OtherIgnoredXML  int `xml:"-"`
	OmitXML          int `xml:",omitempty"`
	OtherOmitXML     int `xml:",omitempty"`
	DuplicateOmitXML int `xml:"a,omitempty"` // ERROR "struct field DuplicateOmitXML repeats xml tag .a. also at testdata/structtag.go:52"
	NonXML           int `foo:"a"`
	DuplicateNonXML  int `foo:"a"`
	Embedded         struct {
		DuplicateXML int `xml:"a"` // OK because its not in the same struct type
	}
}
