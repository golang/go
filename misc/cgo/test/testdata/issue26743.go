// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26743: typedef of uint leads to inconsistent typedefs error.
// No runtime test; just make sure it compiles.

package cgotest

import _ "cgotest/issue26743"
