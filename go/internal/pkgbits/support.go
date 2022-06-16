// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

import "fmt"

func assert(b bool) {
	if !b {
		panic("assertion failed")
	}
}

func errorf(format string, args ...interface{}) {
	panic(fmt.Errorf(format, args...))
}
