// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

var gl int

type X struct {
	a int
	b int
}

func main() {
	print(gl)
}
