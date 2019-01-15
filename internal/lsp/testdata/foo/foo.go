package foo

type StructFoo struct { //@item(StructFoo, "StructFoo", "struct{...}", "struct")
	Value int //@item(Value, "Value", "int", "field")
}

// Pre-set this marker, as we don't have a "source" for it in this package.
/* Error() */ //@item(Error, "Error()", "string", "method")

func Foo() { //@item(Foo, "Foo()", "", "func")
	var err error
	err.Error() //@complete("E", Error)
}

func _() {
	var sFoo StructFoo           //@complete("t", StructFoo)
	if x := sFoo; x.Value == 1 { //@complete("V", Value),typdef("sFoo", StructFoo)
		return
	}
}

//@complete("", Foo, IntFoo, StructFoo)
type IntFoo int //@item(IntFoo, "IntFoo", "int", "type")
