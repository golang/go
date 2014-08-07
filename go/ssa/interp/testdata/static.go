package main

// Static tests of SSA builder (via the sanity checker).
// Dynamic semantics are not exercised.

func init() {
	// Regression test for issue 6806.
	ch := make(chan int)
	select {
	case n, _ := <-ch:
		_ = n
	default:
		// The default case disables the simplification of
		// select to a simple receive statement.
	}

	// value,ok-form receive where TypeOf(ok) is a named boolean.
	type mybool bool
	var x int
	var y mybool
	select {
	case x, y = <-ch:
	default:
		// The default case disables the simplification of
		// select to a simple receive statement.
	}
	_ = x
	_ = y
}

var a int

// Regression test for issue 7840 (covered by SSA sanity checker).
func bug7840() bool {
	// This creates a single-predecessor block with a φ-node.
	return false && a == 0 && a == 0
}

// A blocking select (sans "default:") cannot fall through.
// Regression test for issue 7022.
func bug7022() int {
	var c1, c2 chan int
	select {
	case <-c1:
		return 123
	case <-c2:
		return 456
	}
}

// Parens should not prevent intrinsic treatment of built-ins.
// (Regression test for a crash.)
func init() {
	_ = (new)(int)
	_ = (make)([]int, 0)
}

func main() {}
