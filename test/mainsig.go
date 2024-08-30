// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main(int)  {}           // ERROR "func main must have no arguments and no return values"
func main() int { return 1 } // ERROR "func main must have no arguments and no return values" "main redeclared in this block"

func init(int)  {}           // ERROR "func init must have no arguments and no return values"
func init() int { return 1 } // ERROR "func init must have no arguments and no return values"
