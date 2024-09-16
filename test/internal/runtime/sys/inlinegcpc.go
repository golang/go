// errorcheck -0 -+ -p=internal/runtime/sys -m

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// A function that calls sys.GetCallerPC
// cannot be inlined, no matter how small it is.

func GetCallerPC() uintptr

func pc() uintptr {
	return GetCallerPC() + 1
}

func cpc() uintptr { // ERROR "can inline cpc"
	return pc() + 2
}
