// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

// golang.org/issue/12225
// The test is that this compiles at all.

func issue12225() {
	println(*(*int)(unsafe.Pointer(&convert("")[0])))
	println(*(*int)(unsafe.Pointer(&[]byte("")[0])))
}
