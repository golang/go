// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(mdempsky): Update error expectations for new parser.
// The new parser emits an extra "missing { after for clause" error.
// The old parser is supposed to emit this too, but it panics first
// due to a nil pointer dereference.

package main

func main() {
	for x		// GCCGO_ERROR "undefined"
	{		// ERROR "missing .*{.* after for clause|missing operand"
		z	// ERROR "undefined|missing { after for clause"
