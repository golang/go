// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 22305: gccgo failed to compile this file.

package main

var F func() [0]func()
var i = 2
var B = F()[i]

func main() {}
