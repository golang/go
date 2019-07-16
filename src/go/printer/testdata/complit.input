// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package complit

var (
	// Multi-line declarations
	V1 = T{
		F1: "hello",
		f2: 1,
	}
	V2 = T{
		f2: 1,
		F1: "hello",
	}
	V3 = T{
		F1: "hello",
		F2: T2{
			A: "world",
			b: "hidden",
		},
		f3: T2{
			A: "world",
		},
	}
	V4 = T{
		f2: 1,
	}

	// Single-line declarations
	V5 = T{F1: "hello", f2: 1}
	V6 = T{f2: 1, F1: "hello"}
	V7 = T{f2: 1}

	// Mixed-mode declarations
	V8 = T{
		F1: "hello", f2: 1,
		F3: "world",
		f4: 2}
	V9 = T{
	f2: 1, F1: "hello",}
	V10 = T{
		F1: "hello", f2: 1,
		f3: 2,
		F4: "world", f5: 3,
	}

	// Other miscellaneous declarations
	V11 = T{
		t{
			A: "world",
			b: "hidden",
		},
		f2: t{
			A: "world",
			b: "hidden",
		},
	}
	V12 = T{
		F1: make(chan int),
		f2: []int{},
		F3: make(map[int]string), f4: 1,
	}
)