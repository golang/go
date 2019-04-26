// errorcheck -0 -+ -p=runtime -m -newescape=true

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// A function that calls runtime.getcallerpc or runtime.getcallersp()
// cannot be inlined, no matter how small it is.

func getcallerpc() uintptr
func getcallersp() uintptr

func pc() uintptr {
	return getcallerpc() + 1
}

func cpc() uintptr { // ERROR "can inline cpc"
	return pc() + 2
}

func sp() uintptr {
	return getcallersp() + 3
}

func csp() uintptr { // ERROR "can inline csp"
	return sp() + 4
}
