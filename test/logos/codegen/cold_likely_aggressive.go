// errorcheck -O=3 -0 -d=ssa/likelyadjust/debug=1

package main

// M13-S2 synthetic PGO (SPEC T-105): at the aggressive level an explicit
// panic path is marked BranchUnlikely with no source change, so the
// likelyadjust prediction agrees with the previous mark.

//go:noinline
func h(n int) int { return n * 3 }

func panics(n int) int {
	if n < 0 { // ERROR "Branch prediction rule ret < exit \(agrees with previous\)$"
		panic("negative")
	}
	return h(n)
}

func main() {
	println(panics(3))
}
