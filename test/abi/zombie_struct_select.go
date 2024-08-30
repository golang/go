// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type patchlist struct {
	head, tail uint32
}

type frag struct {
	i   uint32
	out patchlist
}

//go:noinline
//go:registerparams
func patch(l patchlist, i uint32) {
}

//go:noinline
//go:registerparams
func badbad(f1, f2 frag) frag {
	// concat of failure is failure
	if f1.i == 0 || f2.i == 0 { // internal compiler error: 'badbad': incompatible OpArgIntReg [4]: v42 and v26
		return frag{}
	}
	patch(f1.out, f2.i)
	return frag{f1.i, f2.out}
}

func main() {
	badbad(frag{i: 2}, frag{i: 3})
}
