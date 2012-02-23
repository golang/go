// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that imports with backslashes are rejected by the compiler.
// Does not compile.
// TODO: make more thorough.

package main

import `net\http`  // ERROR "backslash"
