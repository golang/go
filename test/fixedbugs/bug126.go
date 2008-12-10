// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// same const identifier declared twice should not be accepted
const none = 0  // GCCGO_ERROR "previous"
const none = 1;  // ERROR "redeclared|redef"
