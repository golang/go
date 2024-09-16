// errorcheck -0 -+ -p=runtime -m

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// A function that calls runtime.getcallersp()
// cannot be inlined, no matter how small it is.

func getcallersp() uintptr

func sp() uintptr {
	return getcallersp() + 3
}

func csp() uintptr { // ERROR "can inline csp"
	return sp() + 4
}
