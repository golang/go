// run

// Test for 6g register move bug.  The optimizer gets confused
// about 32- vs 64-bit moves during splitContractIndex.

// Issue 3918.

package main

func main() {
	const c = 0x123400005678
	index, offset := splitContractIndex(c)
	if index != (c&0xffffffff)>>5 || offset != c+1 {
		println("BUG", index, offset)
	}
}

func splitContractIndex(ce uint64) (index uint32, offset uint64) {
	h := uint32(ce)
	return h >> 5, ce + 1
}
