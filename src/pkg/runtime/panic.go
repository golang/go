// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func panicindex() {
	panic(errorString("index out of range"))
}

func panicslice() {
	panic(errorString("slice bounds out of range"))
}
