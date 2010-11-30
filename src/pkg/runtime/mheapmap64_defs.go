// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	mHeapMap_Level1Bits = 18
	mHeapMap_Level2Bits = 18
	mHeapMap_Level3Bits = 16
	mHeapMap_TotalBits  = mHeapMap_Level1Bits + mHeapMap_Level2Bits + mHeapMap_Level3Bits

	mHeapMap_Level1Mask = (1 << mHeapMap_Level1Bits) - 1
	mHeapMap_Level2Mask = (1 << mHeapMap_Level2Bits) - 1
	mHeapMap_Level3Mask = (1 << mHeapMap_Level3Bits) - 1
)

type mHeapMap struct {
	allocator func(uintptr)
	p         [1 << mHeapMap_Level1Bits]*mHeapMapNode2
}


type mHeapMapNode2 struct {
	p [1 << mHeapMap_Level2Bits]*mHeapMapNode3
}


type mHeapMapNode3 struct {
	s [1 << mHeapMap_Level3Bits]*mSpan
}
