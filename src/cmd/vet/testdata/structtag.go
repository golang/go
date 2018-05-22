// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package testdata

import "encoding/xml"

type StructTagTest struct {
	A   int "hello"            // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
	B   int "\tx:\"y\""        // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag key"
	C   int "x:\"y\"\tx:\"y\"" // ERROR "not compatible with reflect.StructTag.Get"
	D   int "x:`y`"            // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	E   int "ct\brl:\"char\""  // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
	F   int `:"emptykey"`      // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag key"
	G   int `x:"noEndQuote`    // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	H   int `x:"trunc\x0"`     // ERROR "not compatible with reflect.StructTag.Get: bad syntax for struct tag value"
	I   int `x:"foo",y:"bar"`  // ERROR "not compatible with reflect.StructTag.Get: key:.value. pairs not separated by spaces"
	J   int `x:"foo"y:"bar"`   // ERROR "not compatible with reflect.StructTag.Get: key:.value. pairs not separated by spaces"
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

type AnonymousJSON struct{}
type AnonymousXML struct{}

type DuplicateJSONFields struct {
	JSON              int `json:"a"`
	DuplicateJSON     int `json:"a"` // ERROR "struct field DuplicateJSON repeats json tag .a. also at structtag.go:46"
	IgnoredJSON       int `json:"-"`
	OtherIgnoredJSON  int `json:"-"`
	OmitJSON          int `json:",omitempty"`
	OtherOmitJSON     int `json:",omitempty"`
	DuplicateOmitJSON int `json:"a,omitempty"` // ERROR "struct field DuplicateOmitJSON repeats json tag .a. also at structtag.go:46"
	NonJSON           int `foo:"a"`
	DuplicateNonJSON  int `foo:"a"`
	Embedded          struct {
		DuplicateJSON int `json:"a"` // OK because its not in the same struct type
	}
	AnonymousJSON `json:"a"` // ERROR "struct field AnonymousJSON repeats json tag .a. also at structtag.go:46"

	XML              int `xml:"a"`
	DuplicateXML     int `xml:"a"` // ERROR "struct field DuplicateXML repeats xml tag .a. also at structtag.go:60"
	IgnoredXML       int `xml:"-"`
	OtherIgnoredXML  int `xml:"-"`
	OmitXML          int `xml:",omitempty"`
	OtherOmitXML     int `xml:",omitempty"`
	DuplicateOmitXML int `xml:"a,omitempty"` // ERROR "struct field DuplicateOmitXML repeats xml tag .a. also at structtag.go:60"
	NonXML           int `foo:"a"`
	DuplicateNonXML  int `foo:"a"`
	Embedded         struct {
		DuplicateXML int `xml:"a"` // OK because its not in the same struct type
	}
	AnonymousXML `xml:"a"` // ERROR "struct field AnonymousXML repeats xml tag .a. also at structtag.go:60"
	Attribute    struct {
		XMLName     xml.Name `xml:"b"`
		NoDup       int      `xml:"b"`                // OK because XMLName above affects enclosing struct.
		Attr        int      `xml:"b,attr"`           // OK because <b b="0"><b>0</b></b> is valid.
		DupAttr     int      `xml:"b,attr"`           // ERROR "struct field DupAttr repeats xml attribute tag .b. also at structtag.go:76"
		DupOmitAttr int      `xml:"b,omitempty,attr"` // ERROR "struct field DupOmitAttr repeats xml attribute tag .b. also at structtag.go:76"

		AnonymousXML `xml:"b,attr"` // ERROR "struct field AnonymousXML repeats xml attribute tag .b. also at structtag.go:76"
	}
}

type UnexpectedSpacetest struct {
	A int `json:"a,omitempty"`
	B int `json:"b, omitempty"` // ERROR "suspicious space in struct tag value"
	C int `json:"c ,omitempty"`
	D int `json:"d,omitempty, string"` // ERROR "suspicious space in struct tag value"
	E int `xml:"e local"`
	F int `xml:"f "`                 // ERROR "suspicious space in struct tag value"
	G int `xml:" g"`                 // ERROR "suspicious space in struct tag value"
	H int `xml:"h ,omitempty"`       // ERROR "suspicious space in struct tag value"
	I int `xml:"i, omitempty"`       // ERROR "suspicious space in struct tag value"
	J int `xml:"j local ,omitempty"` // ERROR "suspicious space in struct tag value"
	K int `xml:"k local, omitempty"` // ERROR "suspicious space in struct tag value"
	L int `xml:" l local,omitempty"` // ERROR "suspicious space in struct tag value"
	M int `xml:"m  local,omitempty"` // ERROR "suspicious space in struct tag value"
	N int `xml:" "`                  // ERROR "suspicious space in struct tag value"
	O int `xml:""`
	P int `xml:","`
	Q int `foo:" doesn't care "`
}
