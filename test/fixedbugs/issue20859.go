// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func use(b []byte, x byte) {
	b[0] = x
}

//go:noinline
func single(x byte) [16]byte {
	var ret [16]byte
	use(ret[:8], x)
	use(ret[8:], x+1)
	return ret
}

//go:noinline
func addrTaken(x byte) [16]byte {
	var ret [16]byte
	p := &ret
	p[1] = x
	return ret
}

//go:noinline
func captured(x byte) [16]byte {
	var ret [16]byte
	func() {
		ret[2] = x
	}()
	return ret
}

func check(name string, got, want [16]byte) {
	if got != want {
		panic(name)
	}
}

func main() {
	var want [16]byte

	want[0] = 10
	want[8] = 11
	check("single", single(10), want)

	want = [16]byte{}
	want[1] = 12
	check("addrTaken", addrTaken(12), want)

	want = [16]byte{}
	want[2] = 13
	check("captured", captured(13), want)
}
