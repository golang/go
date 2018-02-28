// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Verify that we can import package "a" containing an inlineable
// function F that declares a local interface with a non-exported
// method f.
import _ "./a"

func main() {}
