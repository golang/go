// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we don't see spurious errors for ==
// for values with invalid types due to prior errors.

package p

var x struct {
	f *NotAType /* ERROR undeclared name */
}
var _ = x.f == nil // no error expected here

var y *NotAType  /* ERROR undeclared name */
var _ = y == nil // no error expected here
