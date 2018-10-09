// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for untagged struct literals.

package a

import (
	"flag"
	"go/scanner"
	"go/token"
	"image"
	"unicode"
)

var Okay1 = []string{
	"Name",
	"Usage",
	"DefValue",
}

var Okay2 = map[string]bool{
	"Name":     true,
	"Usage":    true,
	"DefValue": true,
}

var Okay3 = struct {
	X string
	Y string
	Z string
}{
	"Name",
	"Usage",
	"DefValue",
}

var Okay4 = []struct {
	A int
	B int
}{
	{1, 2},
	{3, 4},
}

type MyStruct struct {
	X string
	Y string
	Z string
}

var Okay5 = &MyStruct{
	"Name",
	"Usage",
	"DefValue",
}

var Okay6 = []MyStruct{
	{"foo", "bar", "baz"},
	{"aa", "bb", "cc"},
}

var Okay7 = []*MyStruct{
	{"foo", "bar", "baz"},
	{"aa", "bb", "cc"},
}

// Testing is awkward because we need to reference things from a separate package
// to trigger the warnings.

var goodStructLiteral = flag.Flag{
	Name:  "Name",
	Usage: "Usage",
}
var badStructLiteral = flag.Flag{ // want "unkeyed fields"
	"Name",
	"Usage",
	nil, // Value
	"DefValue",
}

var delta [3]rune

// SpecialCase is a named slice of CaseRange to test issue 9171.
var goodNamedSliceLiteral = unicode.SpecialCase{
	{Lo: 1, Hi: 2, Delta: delta},
	unicode.CaseRange{Lo: 1, Hi: 2, Delta: delta},
}
var badNamedSliceLiteral = unicode.SpecialCase{
	{1, 2, delta},                  // want "unkeyed fields"
	unicode.CaseRange{1, 2, delta}, // want "unkeyed fields"
}

// ErrorList is a named slice, so no warnings should be emitted.
var goodScannerErrorList = scanner.ErrorList{
	&scanner.Error{Msg: "foobar"},
}
var badScannerErrorList = scanner.ErrorList{
	&scanner.Error{token.Position{}, "foobar"}, // want "unkeyed fields"
}

// Check whitelisted structs: if vet is run with --compositewhitelist=false,
// this line triggers an error.
var whitelistedPoint = image.Point{1, 2}

// Do not check type from unknown package.
// See issue 15408.
var unknownPkgVar = unicode.NoSuchType{"foo", "bar"}

// A named pointer slice of CaseRange to test issue 23539. In
// particular, we're interested in how some slice elements omit their
// type.
var goodNamedPointerSliceLiteral = []*unicode.CaseRange{
	{Lo: 1, Hi: 2},
	&unicode.CaseRange{Lo: 1, Hi: 2},
}
var badNamedPointerSliceLiteral = []*unicode.CaseRange{
	{1, 2, delta},                   // want "unkeyed fields"
	&unicode.CaseRange{1, 2, delta}, // want "unkeyed fields"
}
