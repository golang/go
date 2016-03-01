// +build amd64
// errorcheck -0 -d=ssa/phiopt/debug=3

package main

func f0(a bool) bool {
	x := false
	if a {
		x = true
	} else {
		x = false
	}
	return x // ERROR "converted OpPhi to OpCopy$"
}

func f1(a bool) bool {
	x := false
	if a {
		x = false
	} else {
		x = true
	}
	return x // ERROR "converted OpPhi to OpNot$"
}

func f2(a, b int) bool {
	x := true
	if a == b {
		x = false
	}
	return x // ERROR "converted OpPhi to OpNot$"
}

func f3(a, b int) bool {
	x := false
	if a == b {
		x = true
	}
	return x // ERROR "converted OpPhi to OpCopy$"
}

func main() {
}
