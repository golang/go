// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

type mybool bool

var x, y = 1, 2
var _ mybool = x < y && x < y // ERROR "cannot use"
var _ mybool = x < y || x < y // ERROR "cannot use"
