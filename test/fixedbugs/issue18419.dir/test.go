// errorcheck -0 -m -l

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./other"

func InMyCode(e *other.Exported) {
	e.member() // ERROR "e\.member undefined .cannot refer to unexported field or method other\.\(\*Exported\)\.member.|unexported field or method"
}

func main() {}
