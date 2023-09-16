// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

// Tags. Must be < 0.
const (
	// Objects
	packageTag = -(iota + 1)
	constTag
	typeTag
	varTag
	funcTag
	endTag
)
