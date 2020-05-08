// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for untagged struct literals.

package composite

import "flag"

// Testing is awkward because we need to reference things from a separate package
// to trigger the warnings.

var goodStructLiteral = flag.Flag{
	Name:  "Name",
	Usage: "Usage",
}

var badStructLiteral = flag.Flag{ // ERROR "unkeyed fields"
	"Name",
	"Usage",
	nil, // Value
	"DefValue",
}
