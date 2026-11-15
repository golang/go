// errorcheck -0

package p

var seed string = "0123456789abcdef0123456789abcdefTAIL"

//logos:zeroalloc
func sumBytes(s string) int {
	s = seed + s
	b := []byte(s)
	n := 0
	for _, c := range b {
		n += int(c)
	}
	return n
}
