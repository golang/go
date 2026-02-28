// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4663.
// Make sure 'not used' message is placed correctly.

package main

func a(b int) int64 {
  b // ERROR "not used"
  return 0
}
