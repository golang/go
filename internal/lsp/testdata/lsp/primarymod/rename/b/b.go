package b

var c int //@rename("int", "uint")

func _() {
	a := 1 //@rename("a", "error")
	a = 2
	_ = a
}
