// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import p "./b"

var G int

func main() {
	if G == 101 {
		p.G(nil, nil)
	}
}
