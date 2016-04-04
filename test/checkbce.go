// +build amd64
// errorcheck -0 -d=ssa/check_bce/debug=3

package main

func f0(a []int) {
	a[0] = 1 // ERROR "Found IsInBounds$"
	a[0] = 1
	a[6] = 1 // ERROR "Found IsInBounds$"
	a[6] = 1
	a[5] = 1
	a[5] = 1
}

func f1(a [256]int, i int) {
	useInt(a[i])     // ERROR "Found IsInBounds$"
	useInt(a[i%256]) // ERROR "Found IsInBounds$"
	useInt(a[i&255])
	useInt(a[i&17])

	if 4 <= i && i < len(a) {
		useInt(a[i])
		useInt(a[i-1]) // ERROR "Found IsInBounds$"
		useInt(a[i-4]) // ERROR "Found IsInBounds$"
	}
}

func f2(a [256]int, i uint) {
	useInt(a[i]) // ERROR "Found IsInBounds$"
	useInt(a[i%256])
	useInt(a[i&255])
	useInt(a[i&17])
}

func f3(a [256]int, i uint8) {
	useInt(a[i])
	useInt(a[i+10])
	useInt(a[i+14])
}

func f4(a [27]int, i uint8) {
	useInt(a[i%15])
	useInt(a[i%19])
	useInt(a[i%27])
}

func f5(a []int) {
	if len(a) > 5 {
		useInt(a[5])
		useSlice(a[6:])
		useSlice(a[:6]) // ERROR "Found IsSliceInBounds$"
	}
}

func g1(a []int) {
	for i := range a {
		a[i] = i
		useSlice(a[:i+1])
		useSlice(a[:i])
	}
}

func g2(a []int) {
	useInt(a[3]) // ERROR "Found IsInBounds$"
	useInt(a[2])
	useInt(a[1])
	useInt(a[0])
}

func g3(a []int) {
	for i := range a[:256] { // ERROR "Found IsSliceInBounds$"
		useInt(a[i]) // ERROR "Found IsInBounds$"
	}
	b := a[:256]
	for i := range b {
		useInt(b[i])
	}
}

//go:noinline
func useInt(a int) {
}

//go:noinline
func useSlice(a []int) {
}

func main() {
}
