// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "C"

//export ReturnIntLong
func ReturnIntLong() (int, C.long) {
	return 1, 2
}
