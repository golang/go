// asmcheck

package main

func scrub(key []byte) int {
	n := len(key)
	// amd64:"CALL.*memclrNoHeapPointers"
	// arm64:"CALL.*memclrNoHeapPointers"
	clear(key)
	return n
}

func scrubPtr(ptrs []*int) int {
	n := len(ptrs)
	// amd64:"CALL.*memclrHasPointers" -"CALL.*memclrNoHeapPointers"
	// arm64:"CALL.*memclrHasPointers" -"CALL.*memclrNoHeapPointers"
	clear(ptrs)
	return n
}

func main() {}
