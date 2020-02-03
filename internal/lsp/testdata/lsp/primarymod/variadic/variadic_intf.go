package variadic

type baz interface {
	baz()
}

func wantsBaz(...baz) {}

type bazImpl int

func (bazImpl) baz() {}

func _() {
	var (
		impls []bazImpl //@item(vImplSlice, "impls", "[]bazImpl", "var")
		impl  bazImpl   //@item(vImpl, "impl", "bazImpl", "var")
		bazes []baz     //@item(vIntfSlice, "bazes", "[]baz", "var")
	)

	wantsBaz() //@rank(")", vImpl, vImplSlice),rank(")", vIntfSlice, vImplSlice)
}
