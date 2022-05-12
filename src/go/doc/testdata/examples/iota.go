// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

const (
	a = iota
	b
)

const (
	c = 3
	d = 4
)

const (
	e = iota
	f
)

// The example refers to only one of the constants in the iota group, but we
// must keep all of them because of the iota. The second group of constants can
// be trimmed. The third has an iota, but is unused, so it can be eliminated.

func Example() {
	_ = b
	_ = d
}

// Need two examples to hit the playExample function.

func Example2() {
}
