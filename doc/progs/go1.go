// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains examples to embed in the Go 1 release notes document.

package main

import "log"

func main() {
	mapDelete()
}

func mapDelete() {
	m := map[string]int{"7": 7, "23": 23}
	k := "7"
	delete(m, k)
	if m["7"] != 0 || m["23"] != 23 {
		log.Fatal("mapDelete:", m)
	}
}
