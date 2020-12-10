// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for the new multiple-key struct tag format supported in 1.16.

package go16

type Go16StructTagTest struct {
	OK int `multiple keys can:"share a value"`
	OK2 int `json bson xml form:"field_1,omitempty" other:"value"`
}

type Go16UnexportedEncodingTagTest struct {
	F int `json xml:"ff"`

	// We currently always check json first, and return after an error.
	f1 int `json xml:"f1"` // want "struct field f1 has json tag but is not exported"
	f2 int `xml json:"f2"` // want "struct field f2 has json tag but is not exported"
	f3 int `xml bson:"f3"` // want "struct field f3 has xml tag but is not exported"
	f4 int `bson xml:"f4"` // want "struct field f4 has xml tag but is not exported"
}

type Go16DuplicateFields struct {
	JSONXML int `json xml:"c"`
	DuplicateJSONXML int `json xml:"c"` // want "struct field DuplicateJSONXML repeats json tag .c. also at go16.go:25" "struct field DuplicateJSONXML repeats xml tag .c. also at go16.go:25"
}
