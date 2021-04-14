// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

// Issue 29563: internal linker fails on duplicate weak symbols.
// No runtime test; just make sure it compiles.

package cgotest

import _ "cgotest/issue29563"
