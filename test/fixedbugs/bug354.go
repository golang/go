// $G $D/$F.go || echo BUG: bug354

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 2086
// was calling makeclosure twice on the closure

package main

import (
	"os"
)

type Inner struct {
	F func() os.Error
}

type Outer struct {
	Inners []Inner
}

// calls makeclosure twice on same closure

var Foo = Outer{[]Inner{Inner{func() os.Error{ return nil }}}}
