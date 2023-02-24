// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const (
	zero = iota
	one
	two
	three
)

const iii int = 0x3

func f(v int) {
	switch v {
	case zero, one:
	case two, one: // ERROR "previous case at LINE-1|duplicate case .*in.* switch"

	case three:
	case 3: // ERROR "previous case at LINE-1|duplicate case .*in.* switch"
	case iii: // ERROR "previous case at LINE-2|duplicate case .*in.* switch"
	}
}

const b = "b"

var _ = map[string]int{
	"a": 0,
	b:   1,
	"a": 2, // ERROR "previous key at LINE-2|duplicate key.*in map literal"
	"b": 3, // GC_ERROR "previous key at LINE-2|duplicate key.*in map literal"
	"b": 4, // GC_ERROR "previous key at LINE-3|duplicate key.*in map literal"
}
