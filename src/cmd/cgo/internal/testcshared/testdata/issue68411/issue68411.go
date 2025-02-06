// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

//export exportFuncWithNoParams
func exportFuncWithNoParams() {}

//export exportFuncWithParams
func exportFuncWithParams(a, b int) {}

func main() {}
