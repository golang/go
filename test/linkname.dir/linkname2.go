package y

import _ "unsafe"

//go:linkname byteIndex test/linkname1.indexByte
func byteIndex(xs []byte, b byte) int // ERROR "leaking param: xs"

func ContainsSlash(data []byte) bool { // ERROR "leaking param: data" "can inline ContainsSlash"
	if byteIndex(data, '/') != -1 {
		return true
	}
	return false
}
