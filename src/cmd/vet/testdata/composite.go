// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for untagged struct literals.

package testdata

import (
	"flag"
	"go/scanner"
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

type MyStruct struct {
	X string
	Y string
	Z string
}

var Okay4 = MyStruct{
	"Name",
	"Usage",
	"DefValue",
}

// Testing is awkward because we need to reference things from a separate package
// to trigger the warnings.

var BadStructLiteralUsedInTests = flag.Flag{ // ERROR "unkeyed fields"
	"Name",
	"Usage",
	nil, // Value
	"DefValue",
}

// SpecialCase is an (aptly named) slice of CaseRange to test issue 9171.
var GoodNamedSliceLiteralUsedInTests = unicode.SpecialCase{
	{Lo: 1, Hi: 2},
}

// Used to test the check for slices and arrays: If that test is disabled and
// vet is run with --compositewhitelist=false, this line triggers an error.
// Clumsy but sufficient.
var scannerErrorListTest = scanner.ErrorList{nil, nil}
