// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for go.dev/issue/31092: Better synchronization of
// parser after seeing an := rather than an = in a const,
// type, or variable declaration.

package p

const _ /* ERROR unexpected := */ := 0
type _ /* ERROR unexpected := */ := int
var _ /* ERROR unexpected := */ := 0

const _ int /* ERROR unexpected := */ := 0
var _ int /* ERROR unexpected := */ := 0
