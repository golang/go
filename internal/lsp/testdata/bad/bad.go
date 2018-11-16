package bad

func stuff() {
	x := "heeeeyyyy"
	random2(x) //@diag("x", "cannot use x (variable of type string) as int value in argument to random2")
	random2(1)
	y := 3 //@diag("y", "y declared but not used")
}

type bob struct {
	x int
}

func _() {
	_ = &bob{
		f: 0, //@diag("f", "unknown field f in struct literal")
	}
}
