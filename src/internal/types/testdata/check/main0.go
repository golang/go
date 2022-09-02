// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main()
func main /* ERROR "no arguments and no return values" */ /* ERROR redeclared */ (int)
func main /* ERROR "no arguments and no return values" */ /* ERROR redeclared */ () int
