// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the structtag checker.

// This file contains the test for canonical struct tags.

package testdata

type StructTagTest struct {
	X int "hello" // ERROR "not compatible with reflect.StructTag.Get"
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
