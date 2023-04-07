//go:build go1.11 && !go1.21
// +build go1.11,!go1.21

package bad

import _ "golang.org/lsptests/assign/internal/secret" //@diag("\"golang.org/lsptests/assign/internal/secret\"", "compiler", "could not import golang.org/lsptests/assign/internal/secret \\(invalid use of internal package \"golang.org/lsptests/assign/internal/secret\"\\)", "error")

func stuff() { //@item(stuff, "stuff", "func()", "func")
	x := "heeeeyyyy"
	random2(x) //@diag("x", "compiler", "cannot use x \\(variable of type string\\) as int value in argument to random2", "error")
	random2(1) //@complete("dom", random, random2, random3)
	y := 3     //@diag("y", "compiler", "y declared (and|but) not used", "error")
}

type bob struct { //@item(bob, "bob", "struct{...}", "struct")
	x int
}

func _() {
	var q int
	_ = &bob{
		f: q, //@diag("f: q", "compiler", "unknown field f in struct literal", "error")
	}
}
