// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// To create a test case for a new export format version,
// build this package with the latest compiler and store
// the resulting .a file appropriately named in the versions
// directory. The VersionHandling test will pick it up.
//
// In the testdata/versions:
//
// go build -o test_go1.$X_$Y.a test.go
//
// with $X = Go version and $Y = export format version
// (add 'b' or 'i' to distinguish between binary and
// indexed format starting with 1.11 as long as both
// formats are supported).
//
// Make sure this source is extended such that it exercises
// whatever export format change has taken place.

package test

// Any release before and including Go 1.7 didn't encode
// the package for a blank struct field.
type BlankField struct {
	_ int
}
