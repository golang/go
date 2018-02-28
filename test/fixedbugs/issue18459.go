// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we have a line number for this error.

package main

//go:nowritebarrier // ERROR "//go:nowritebarrier only allowed in runtime"
func main() {
}
