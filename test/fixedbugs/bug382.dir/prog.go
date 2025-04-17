// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

// Issue 2529

package main

import "./pkg"

var x = pkg.E

var fo = struct{ F pkg.T }{F: x}
