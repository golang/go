// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	m := make(map[string][1000]byte)
	m["hi"] = [1000]byte{1}
	
	v := m["hi"]
	
	for k, vv := range m {
		if k != "hi" || string(v[:]) != string(vv[:]) {
			panic("bad iter")
		}
	}
}
