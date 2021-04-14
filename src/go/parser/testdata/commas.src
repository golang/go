// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for error messages/parser synchronization
// after missing commas.

package p

var _ = []int{
	0/* ERROR HERE "missing ','" */
}

var _ = []int{
	0,
	1,
	2,
	3/* ERROR HERE "missing ','" */
}
