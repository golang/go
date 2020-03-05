package undeclared

func m() int {
	z, _ := 1+y, 11 //@diag("y", "undeclaredname", "undeclared name: y", "error"),suggestedfix("y", "quickfix")
	if 100 < 90 {
		z = 1
	} else if 100 > n+2 { //@diag("n", "undeclaredname", "undeclared name: n", "error"),suggestedfix("n", "quickfix")
		z = 4
	}
	for i < 200 { //@diag("i", "undeclaredname", "undeclared name: i", "error"),suggestedfix("i", "quickfix")
	}
	r() //@diag("r", "compiler", "undeclared name: r", "error")
	return z
}
