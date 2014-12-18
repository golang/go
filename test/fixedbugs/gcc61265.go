// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61265: The gccgo middle-end failed to represent array composite literals
// where the elements are zero-sized values.
// This is a reduction of a program reported by GoSmith.

package p

var a = [1][0]int{B}[0]
var B = [0]int{}
var c = [1]struct{}{D}[0]
var D = struct{}{}
