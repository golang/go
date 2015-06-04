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
