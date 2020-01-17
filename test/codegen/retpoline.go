// +build amd64
// asmcheck -gcflags=-spectre=ret

package codegen

func CallFunc(f func()) {
	// amd64:`CALL\truntime.retpoline`
	f()
}

func CallInterface(x interface{ M() }) {
	// amd64:`CALL\truntime.retpoline`
	x.M()
}
