// errorcheck -O=2 -0 -d=ssa/likelyadjust/debug=1

package main

import (
	"logos/rt"
	"os"
)

// M13-S2 synthetic PGO (SPEC T-105): the Cold intrinsic emits no code and
// marks the branch into its block BranchUnlikely before likelyadjust, so
// the pass reports that its own prediction agrees with the previous one.
// An unmarked branch gets the same prediction without that suffix. At the
// standard level an os.Exit path is marked with no source change and an
// explicit panic path is not. The likelyadjust debug line sits on the
// line of the if.

//go:noinline
func g(n int) int { return n * 2 }

//go:noinline
func h(n int) int { return n * 3 }

func marked(n int) int {
	if n < 0 { // ERROR "Branch prediction rule default < call \(agrees with previous\)$"
		rt.Cold()
		g(n)
	}
	return h(n)
}

func unmarked(n int) int {
	if n < 0 { // ERROR "Branch prediction rule default < call$"
		g(n)
	}
	return h(n)
}

func exits(n int) int {
	if n < 0 { // ERROR "Branch prediction rule default < call \(agrees with previous\)$"
		os.Exit(1)
	}
	return h(n)
}

func panics(n int) int {
	if n < 0 { // ERROR "Branch prediction rule ret < exit$"
		panic("negative")
	}
	return h(n)
}

func main() {
	println(marked(3), unmarked(3), exits(3), panics(3))
}
