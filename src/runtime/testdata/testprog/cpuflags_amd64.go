// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func init() {
	register("CheckAVX", CheckAVX)
}

func CheckAVX() {
	checkAVX()
	fmt.Println("OK")
}

func checkAVX()
