// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func Println() {} // ERROR "Println redeclared in this block|Println already declared"

func main() {}
