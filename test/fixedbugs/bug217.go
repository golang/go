// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash
// http://code.google.com/p/go/issues/detail?id=204

package main

func () x()	// ERROR "no receiver"

func (a b, c d) x()	// ERROR "multiple receiver"

