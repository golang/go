package foo

type StructFoo struct { //@mark(StructFoo, "StructFoo"),item(StructFoo, "StructFoo", "struct{...}", "struct")
	Value int //@mark(Value, "Value"),item(Value, "Value", "int", "field")
}

// TODO(rstambler): Create pre-set builtins?
//@mark(Error, ""),item(Error, "Error()", "string", "method")

func Foo() { //@mark(Foo, "Foo"),item(Foo, "Foo()", "", "func")
	var err error
	err.Error() //@complete("E", Error)
}

func _() {
	var sFoo StructFoo           //@complete("t", StructFoo)
	if x := sFoo; x.Value == 1 { //@complete("V", Value)
		return
	}
}

//@complete("", Foo, IntFoo, StructFoo)
type IntFoo int //@mark(IntFoo, "IntFoo"),item(IntFoo, "IntFoo", "int", "type")
