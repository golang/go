// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "C"

//export exportSliceIn
func exportSliceIn(s []byte) bool {
	return len(s) == cap(s)
}

//export exportSliceOut
func exportSliceOut() []byte {
	return []byte{1}
}

//export exportSliceInOut
func exportSliceInOut(s []byte) []byte {
	return s
}
