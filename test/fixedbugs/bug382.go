// $G $D/$F.dir/pkg.go && $G $D/$F.go || echo "Bug 382"

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

// Issue 2529

package main
import "./pkg"

var x = pkg.E

var fo = struct {F pkg.T}{F: x}
