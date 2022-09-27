//go:build go1.11
// +build go1.11

package bad

// See #36637
type stateFunc func() stateFunc //@item(stateFunc, "stateFunc", "func() stateFunc", "type")

var a unknown //@item(global_a, "a", "unknown", "var"),diag("unknown", "compiler", "(undeclared name|undefined): unknown", "error")

func random() int { //@item(random, "random", "func() int", "func")
	//@complete("", global_a, bob, random, random2, random3, stateFunc, stuff)
	return 0
}

func random2(y int) int { //@item(random2, "random2", "func(y int) int", "func"),item(bad_y_param, "y", "int", "var")
	x := 6       //@item(x, "x", "int", "var"),diag("x", "compiler", "x declared (and|but) not used", "error")
	var q blah   //@item(q, "q", "blah", "var"),diag("q", "compiler", "q declared (and|but) not used", "error"),diag("blah", "compiler", "(undeclared name|undefined): blah", "error")
	var t **blob //@item(t, "t", "**blob", "var"),diag("t", "compiler", "t declared (and|but) not used", "error"),diag("blob", "compiler", "(undeclared name|undefined): blob", "error")
	//@complete("", q, t, x, bad_y_param, global_a, bob, random, random2, random3, stateFunc, stuff)

	return y
}

func random3(y ...int) { //@item(random3, "random3", "func(y ...int)", "func"),item(y_variadic_param, "y", "[]int", "var")
	//@complete("", y_variadic_param, global_a, bob, random, random2, random3, stateFunc, stuff)

	var ch chan (favType1)   //@item(ch, "ch", "chan (favType1)", "var"),diag("ch", "compiler", "ch declared (and|but) not used", "error"),diag("favType1", "compiler", "(undeclared name|undefined): favType1", "error")
	var m map[keyType]int    //@item(m, "m", "map[keyType]int", "var"),diag("m", "compiler", "m declared (and|but) not used", "error"),diag("keyType", "compiler", "(undeclared name|undefined): keyType", "error")
	var arr []favType2       //@item(arr, "arr", "[]favType2", "var"),diag("arr", "compiler", "arr declared (and|but) not used", "error"),diag("favType2", "compiler", "(undeclared name|undefined): favType2", "error")
	var fn1 func() badResult //@item(fn1, "fn1", "func() badResult", "var"),diag("fn1", "compiler", "fn1 declared (and|but) not used", "error"),diag("badResult", "compiler", "(undeclared name|undefined): badResult", "error")
	var fn2 func(badParam)   //@item(fn2, "fn2", "func(badParam)", "var"),diag("fn2", "compiler", "fn2 declared (and|but) not used", "error"),diag("badParam", "compiler", "(undeclared name|undefined): badParam", "error")
	//@complete("", arr, ch, fn1, fn2, m, y_variadic_param, global_a, bob, random, random2, random3, stateFunc, stuff)
}
