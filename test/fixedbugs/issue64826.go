// build

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	f(g(false))
}
func g(b bool) string {
	if b {
		return "z"
	}
	return "q"
}
func f(x string) int {
	switch len(x) {
	case 4:
		return 4
	case 5:
		return 5
	case 6:
		return 6
	case 7:
		return 7
	case 8:
		return 8
	case 9:
		return 9
	case 10:
		return 10
	case 11:
		return 11
	}
	return 0
}
