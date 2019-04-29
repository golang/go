package snippets

func foo(i int, b bool) {} //@item(snipFoo, "foo(i int, b bool)", "", "func")
func bar(fn func()) func()    {} //@item(snipBar, "bar(fn func())", "", "func")

type Foo struct {
	Bar int //@item(snipFieldBar, "Bar", "int", "field")
}

func (Foo) Baz() func() {} //@item(snipMethodBaz, "Baz()", "func()", "field")

func _() {
	f //@snippet(" //", snipFoo, "oo(${1})", "oo(${1:i int}, ${2:b bool})")

	bar //@snippet(" //", snipBar, "(${1})", "(${1:fn func()})")

	bar(nil) //@snippet("(", snipBar, "", "")
	bar(ba) //@snippet(")", snipBar, "r(${1})", "r(${1:fn func()})")
	var f Foo
	bar(f.Ba) //@snippet(")", snipMethodBaz, "z()", "z()")

	Foo{
		B //@snippet(" //", snipFieldBar, "ar: ${1},", "ar: ${1:int},")
	}

	Foo{B} //@snippet("}", snipFieldBar, "ar: ${1}", "ar: ${1:int}")
	Foo{} //@snippet("}", snipFieldBar, "Bar: ${1}", "Bar: ${1:int}")

	Foo{Foo{}.B} //@snippet("} ", snipFieldBar, "ar", "ar")
}
