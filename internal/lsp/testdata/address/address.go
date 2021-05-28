package address

func wantsPtr(*int)            {}
func wantsVariadicPtr(...*int) {}

func wantsVariadic(...int) {}

type foo struct{ c int } //@item(addrFieldC, "c", "int", "field")

func _() {
	var (
		a string //@item(addrA, "a", "string", "var")
		b int    //@item(addrB, "b", "int", "var")
	)

	wantsPtr()   //@rank(")", addrB, addrA),snippet(")", addrB, "&b", "&b")
	wantsPtr(&b) //@snippet(")", addrB, "b", "b")

	wantsVariadicPtr() //@rank(")", addrB, addrA),snippet(")", addrB, "&b", "&b")

	var s foo
	s.c          //@item(addrDeepC, "s.c", "int", "field")
	wantsPtr()   //@snippet(")", addrDeepC, "&s.c", "&s.c")
	wantsPtr(s)  //@snippet(")", addrDeepC, "&s.c", "&s.c")
	wantsPtr(&s) //@snippet(")", addrDeepC, "s.c", "s.c")

	// don't add "&" in item (it gets added as an additional edit)
	wantsPtr(&s.c) //@snippet(")", addrFieldC, "c", "c")

	// check dereferencing as well
	var c *int    //@item(addrCPtr, "c", "*int", "var")
	var _ int = _ //@rank("_ //", addrCPtr, addrA),snippet("_ //", addrCPtr, "*c", "*c")

	wantsVariadic() //@rank(")", addrCPtr, addrA),snippet(")", addrCPtr, "*c", "*c")

	var d **int   //@item(addrDPtr, "d", "**int", "var")
	var _ int = _ //@rank("_ //", addrDPtr, addrA),snippet("_ //", addrDPtr, "**d", "**d")

	type namedPtr *int
	var np namedPtr //@item(addrNamedPtr, "np", "namedPtr", "var")

	var _ int = _ //@rank("_ //", addrNamedPtr, addrA)

	// don't get tripped up by recursive pointer type
	type dontMessUp *dontMessUp
	var dmu *dontMessUp //@item(addrDMU, "dmu", "*dontMessUp", "var")

	var _ int = dmu //@complete(" //", addrDMU)
}

func (f foo) ptr() *foo { return &f }

func _() {
	getFoo := func() foo { return foo{} }

	// not addressable
	getFoo().c //@item(addrGetFooC, "getFoo().c", "int", "field")

	// addressable
	getFoo().ptr().c //@item(addrGetFooPtrC, "getFoo().ptr().c", "int", "field")

	wantsPtr()   //@rank(addrGetFooPtrC, addrGetFooC),snippet(")", addrGetFooPtrC, "&getFoo().ptr().c", "&getFoo().ptr().c")
	wantsPtr(&g) //@rank(addrGetFooPtrC, addrGetFooC),snippet(")", addrGetFooPtrC, "getFoo().ptr().c", "getFoo().ptr().c")
}

type nested struct {
	f foo
}

func _() {
	getNested := func() nested { return nested{} }

	getNested().f.c       //@item(addrNestedC, "getNested().f.c", "int", "field")
	getNested().f.ptr().c //@item(addrNestedPtrC, "getNested().f.ptr().c", "int", "field")

	// addrNestedC is not addressable, so rank lower
	wantsPtr(getNestedfc) //@fuzzy(")", addrNestedPtrC, addrNestedC)
}
