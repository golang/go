// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

var s [][10]int
const m = len(s[len(s)-1]) // ERROR "is not a constant" 

