// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int{
	3 // ERROR "need trailing comma before newline in composite literal|expecting comma or }"
}
