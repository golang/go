// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	if true {
	} else ;  // ERROR "else must be followed by if or statement block|expected .if. or .{."
}
