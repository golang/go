// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7153: array invalid index error duplicated on successive bad values

package p

var _ = []int{a: true, true} // ERROR "undefined: a" "cannot use true \(type bool\) as type int in array element"
