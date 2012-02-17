// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// import paths are slash-separated; reject backslash

package main

import `net\http`  // ERROR "backslash"
