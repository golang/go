// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import _ "unsafe" // for go:linkname

// Defined in the runtime package.
//
//go:linkname runtime_getm_for_test runtime.getm
func runtime_getm_for_test() uintptr
