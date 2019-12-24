package address

func wantsPtr(*int) {}

type foo struct{ c int } //@item(addrFieldC, "c", "int", "field")

func _() {
	var (
		a string //@item(addrA, "a", "string", "var")
		b int    //@item(addrB, "b", "int", "var")
	)

	&b //@item(addrBRef, "&b", "int", "var")

	wantsPtr()   //@rank(")", addrBRef, addrA),snippet(")", addrBRef, "&b", "&b")
	wantsPtr(&b) //@snippet(")", addrB, "b", "b")

	var s foo
	s.c          //@item(addrDeepC, "s.c", "int", "field")
	&s.c         //@item(addrDeepCRef, "&s.c", "int", "field")
	wantsPtr()   //@snippet(")", addrDeepCRef, "&s.c", "&s.c")
	wantsPtr(s)  //@snippet(")", addrDeepCRef, "&s.c", "&s.c")
	wantsPtr(&s) //@snippet(")", addrDeepC, "s.c", "s.c")

	// don't add "&" in item (it gets added as an additional edit)
	wantsPtr(&s.c) //@snippet(")", addrFieldC, "c", "c")
}

func (f foo) ptr() *foo { return &f }

func _() {
	getFoo := func() foo { return foo{} }

	// not addressable
	getFoo().c //@item(addrGetFooC, "getFoo().c", "int", "field")

	// addressable
	getFoo().ptr().c  //@item(addrGetFooPtrC, "getFoo().ptr().c", "int", "field")
	&getFoo().ptr().c //@item(addrGetFooPtrCRef, "&getFoo().ptr().c", "int", "field")

	wantsPtr()   //@rank(addrGetFooPtrCRef, addrGetFooC),snippet(")", addrGetFooPtrCRef, "&getFoo().ptr().c", "&getFoo().ptr().c")
	wantsPtr(&g) //@rank(addrGetFooPtrC, addrGetFooC),snippet(")", addrGetFooPtrC, "getFoo().ptr().c", "getFoo().ptr().c")
}

type nested struct {
	f foo
}

func _() {
	getNested := func() nested { return nested{} }

	getNested().f.c        //@item(addrNestedC, "getNested().f.c", "int", "field")
	&getNested().f.ptr().c //@item(addrNestedPtrC, "&getNested().f.ptr().c", "int", "field")

	// addrNestedC is not addressable, so rank lower
	wantsPtr(getNestedfc) //@fuzzy(")", addrNestedPtrC, addrNestedC)
}
