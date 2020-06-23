// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can have two identical cgo packages in a single binary.
// No runtime test; just make sure it compiles.

package cgotest

import _ "cgotest/issue23555a"
import _ "cgotest/issue23555b"
