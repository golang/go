// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [10]int	// ok
var b [1e1]int	// ok
var c [1.5]int	// ERROR "truncated"
var d ["abc"]int	// ERROR "invalid array bound|not numeric"
var e [nil]int	// ERROR "invalid array bound|not numeric"
var f [e]int	// ERROR "invalid array bound|not constant"
var g [1<<65]int	// ERROR "overflows"
