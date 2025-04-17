// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Þfoo

var ÞbarV int = 101

func Þbar(x int) int {
	defer func() { ÞbarV += 3 }()
	return Þblix(x)
}

func Þblix(x int) int {
	defer func() { ÞbarV += 9 }()
	return ÞbarV + x
}
