// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main()
func /* ERROR "no arguments and no return values" */ main /* ERROR redeclared */ (int)
func /* ERROR "no arguments and no return values" */ main /* ERROR redeclared */ () int
