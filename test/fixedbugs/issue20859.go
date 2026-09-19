// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var sink *[16]byte

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

//go:noinline
func multi(x byte, c bool) [16]byte {
	var ret [16]byte
	ret[4] = x
	if c {
		return ret
	}
	ret[5] = x + 1
	return ret
}

//go:noinline
func escaped(x byte, c bool) [16]byte {
	var ret [16]byte
	ret[3] = x
	sink = &ret
	if c {
		return ret
	}
	return [16]byte{4: 99}
}

//go:noinline
func partial(x byte, c bool) [16]byte {
	var ret [16]byte
	ret[6] = x
	if c {
		return ret
	}
	return [16]byte{7: 77}
}

//go:noinline
func partialNonCandidateLocal(x byte, c bool) [16]byte {
	var ret [16]byte
	ret[8] = x
	if c {
		return ret
	}
	other := [16]byte{9: 88}
	return other
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

	want = [16]byte{}
	want[4] = 15
	check("multi true", multi(15, true), want)
	want[5] = 16
	check("multi false", multi(15, false), want)

	want = [16]byte{}
	want[3] = 14
	check("escaped true", escaped(14, true), want)
	if *sink != want {
		panic("escaped pointer")
	}
	want = [16]byte{4: 99}
	check("escaped false", escaped(14, false), want)

	want = [16]byte{}
	want[6] = 17
	check("partial true", partial(17, true), want)
	want = [16]byte{7: 77}
	check("partial false", partial(17, false), want)

	want = [16]byte{}
	want[8] = 18
	check("partialNonCandidateLocal true", partialNonCandidateLocal(18, true), want)
	want = [16]byte{9: 88}
	check("partialNonCandidateLocal false", partialNonCandidateLocal(18, false), want)
}
