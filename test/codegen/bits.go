// asmcheck

package codegen

func bitcheck(a, b uint64) int {
	if a&(1<<(b&63)) != 0 { // amd64:"BTQ"
		return 1
	}
	return -1
}
