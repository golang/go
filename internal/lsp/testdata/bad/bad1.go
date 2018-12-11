// +build go1.11

package bad

// import (
// 	"github.com/bob/pkg" //@diag("\"github.com/bob/pkg\"", "unable to import "\"github.com/bob/pkg\"")
// )

var a unknown //@item(global_a, "a", "unknown", "var"),diag("unknown", "undeclared name: unknown")

func random() int { //@item(random, "random()", "int", "func")
	//@complete("", global_a, bob, random, random2, random3, stuff)
	return 0
}

func random2(y int) int { //@item(random2, "random2(y int)", "int", "func"),item(bad_y_param, "y", "int", "var")
	x := 6     //@item(x, "x", "int", "var"),diag("x", "x declared but not used")
	var q blah //@item(q, "q", "blah", "var"),diag("q", "q declared but not used"),diag("blah", "undeclared name: blah")
	var t blob //@item(t, "t", "blob", "var"),diag("t", "t declared but not used"),diag("blob", "undeclared name: blob")
	//@complete("", q, t, x, bad_y_param, global_a, bob, random, random2, random3, stuff)

	return y
}

func random3(y ...int) { //@item(random3, "random3(y ...int)", "", "func"),item(y_variadic_param, "y", "[]int", "var")
	//@complete("", y_variadic_param, global_a, bob, random, random2, random3, stuff)
}
