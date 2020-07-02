package b

var c int //@rename("int", "uint")

func _() {
	a := 1 //@rename("a", "error")
	a = 2
	_ = a
}

var (
	// Hello there.
	// Foo does the thing.
	Foo int //@rename("Foo", "Bob")
)

/*
Hello description
*/
func Hello() {} //@rename("Hello", "Goodbye")
