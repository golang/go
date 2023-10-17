// run

// Test for 8g register move bug.  The optimizer gets confused
// about 16- vs 32-bit moves during splitContractIndex.

// Issue 3910.

package main

func main() {
	const c = 0x12345678
	index, n, offset := splitContractIndex(c)
	if index != int((c&0xffff)>>5) || n != int(c&(1<<5-1)) || offset != (c>>16)&(1<<14-1) {
		println("BUG", index, n, offset)
	}
}

func splitContractIndex(ce uint32) (index, n, offset int) {
	h := uint16(ce)
	return int(h >> 5), int(h & (1<<5 - 1)), int(ce>>16) & (1<<14 - 1)
}
