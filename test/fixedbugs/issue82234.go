// compile

package p

func f(r []int) {
	for i := range uint64(1 << 63) {
		r[i] = 0
	}
}
