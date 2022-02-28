package x

func indexByte(xs []byte, b byte) int { // ERROR "xs does not escape" "can inline indexByte"
	for i, x := range xs {
		if x == b {
			return i
		}
	}
	return -1
}
