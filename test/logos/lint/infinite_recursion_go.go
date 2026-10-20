// errorcheck

// correctness::infinite_recursion on a Go package (SPEC Sec 1.2, #157,
// R-U1): a function whose every path from entry calls itself with identical
// arguments before any return. `logos lint` runs the rule at deny here, so
// a //logos:allow directive silences it. Twin of
// infinite_recursion.lgo with the same boundary cases.

package p

func selfDirect(n int) int { return selfDirect(n) } // ERROR "correctness::infinite_recursion"

func selfAfterCall(n int) int { // ERROR "correctness::infinite_recursion"
	if n > 0 {
		selfAfterCall(n)
		return n
	}
	return selfAfterCall(n)
}

func selfFallthrough(n int) int { // ERROR "correctness::infinite_recursion"
	if n > 0 {
		selfFallthrough(n)
	}
	return selfFallthrough(n)
}

func selfSwitch(n int) int { // ERROR "correctness::infinite_recursion"
	switch n {
	case 0:
		return selfSwitch(n)
	default:
		return selfSwitch(n)
	}
}

func selfSwitchFallthrough(n int) int { // ERROR "correctness::infinite_recursion"
	switch n {
	case 0:
		return selfSwitchFallthrough(n)
	case 1:
		return selfSwitchFallthrough(n)
	}
	return selfSwitchFallthrough(n)
}

func selfNested(n int) int { // ERROR "correctness::infinite_recursion"
	if n > 0 {
		if n > 1 {
			return selfNested(n)
		}
	}
	return selfNested(n)
}

func nestedNoCall(n int) int {
	if n > 0 {
		if n > 1 {
			return n
		}
	}
	return nestedNoCall(n)
}

func guarded(n int) int {
	if n > 0 {
		return n
	}
	return guarded(n)
}

func partialSwitch(n int) int {
	switch n {
	case 0:
		return 0
	default:
		return partialSwitch(n)
	}
}

func switchNoDefault(n int) int {
	switch n {
	case 0:
		return switchNoDefault(n)
	}
	return 0
}

func differentArgs(n int) int { return differentArgs(n - 1) }

func reassigned(n int) int {
	n--
	return reassigned(n)
}

//logos:allow(correctness::infinite_recursion)
func allowedDirect(n int) int { return allowedDirect(n) }

// Cases pinned by the M9-S2 callee-rule slice.

func loop(n int) int { return loop(n) } // ERROR "infinite_recursion"

func tick() { tick() } // ERROR "infinite_recursion"

type counter struct{ n int }

func (c counter) next(n int) int { return c.next(n) } // ERROR "infinite_recursion"

func guardedDecrement(n int) int {
	if n <= 0 {
		return 0
	}
	return guardedDecrement(n - 1)
}

func countdown(n int) int {
	if n == 0 {
		return 0
	}
	return countdown(n - 1)
}

func bump(n int) int {
	n++
	return bump(n)
}

func ping(n int) int { return pong(n) }

func pong(n int) int { return ping(n) }

func read(n int) int {
	println(n)
	return n
}
