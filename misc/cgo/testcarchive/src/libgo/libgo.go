// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import _ "p"

import "C"

var (
	ranInit bool
	ranMain bool
)

func init() { ranInit = true }

func main() { ranMain = true }

//export DidInitRun
func DidInitRun() bool { return ranInit }

//export DidMainRun
func DidMainRun() bool { return ranMain }
