// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import _ "unsafe"

//go:linkname lockedOSThread runtime.lockedOSThread
//extern runtime_lockedOSThread
func lockedOSThread() bool
