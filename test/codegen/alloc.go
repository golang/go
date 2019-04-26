// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check that allocating a 0-size object does not
// introduce a call to runtime.newobject.

package codegen

func zeroAllocNew1() *struct{} {
	// 386:-`CALL\truntime\.newobject`
	// amd64:-`CALL\truntime\.newobject`
	// arm:-`CALL\truntime\.newobject`
	// arm64:-`CALL\truntime\.newobject`
	return new(struct{})
}

func zeroAllocNew2() *[0]int {
	// 386:-`CALL\truntime\.newobject`
	// amd64:-`CALL\truntime\.newobject`
	// arm:-`CALL\truntime\.newobject`
	// arm64:-`CALL\truntime\.newobject`
	return new([0]int)
}

func zeroAllocSliceLit() []int {
	// 386:-`CALL\truntime\.newobject`
	// amd64:-`CALL\truntime\.newobject`
	// arm:-`CALL\truntime\.newobject`
	// arm64:-`CALL\truntime\.newobject`
	return []int{}
}
