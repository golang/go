// asmcheck -gcflags=-spectre=ret

//go:build amd64

package codegen

func CallFunc(f func()) {
	// amd64:`CALL\truntime.retpoline`
	f()
}

func CallInterface(x interface{ M() }) {
	// amd64:`CALL\truntime.retpoline`
	x.M()
}

// Check to make sure that jump tables are disabled
// when retpoline is on. See issue 57097.
func noJumpTables(x int) int {
	switch x {
	case 0:
		return 0
	case 1:
		return 1
	case 2:
		return 2
	case 3:
		return 3
	case 4:
		return 4
	case 5:
		return 5
	case 6:
		return 6
	case 7:
		return 7
	case 8:
		return 8
	case 9:
		return 9
	}
	return 10
}
