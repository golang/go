// build

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const big uint64 = 1<<63

func f(a uint64) uint64 {
	return a << big
}

func main() {
	f(1)
}

/*
main·f: doasm: notfound from=75 to=13 (82)    SHLQ    $-9223372036854775808,BX
main·f: doasm: notfound from=75 to=13 (82)    SHLQ    $-9223372036854775808,BX
main·f: doasm: notfound from=75 to=13 (82)    SHLQ    $-9223372036854775808,BX
*/
