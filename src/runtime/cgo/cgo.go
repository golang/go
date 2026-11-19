// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cgo contains runtime support for code generated
by the cgo tool.  See the documentation for the cgo command
for details on using cgo.
*/
package cgo

import "internal/runtime/sys"

// Incomplete is used specifically for the semantics of incomplete C types.
type Incomplete struct {
	_ sys.NotInHeap
}
