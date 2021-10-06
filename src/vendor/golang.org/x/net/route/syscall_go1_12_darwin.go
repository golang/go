// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.12
// +build go1.12

package route

import _ "unsafe" // for linkname

//go:linkname sysctl syscall.sysctl
func sysctl(mib []int32, old *byte, oldlen *uintptr, new *byte, newlen uintptr) error
