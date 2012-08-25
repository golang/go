// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use the functions in pkg2.go so that the inlined
// forms get type-checked.

package pkg3

import "./pkg2"

var x = pkg2.F()
var v = pkg2.V
