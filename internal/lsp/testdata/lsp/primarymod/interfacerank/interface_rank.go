package interfacerank

type foo interface {
	foo()
}

type fooImpl int

func (*fooImpl) foo() {}

func wantsFoo(foo) {}

func _() {
	var (
		aa string   //@item(irAA, "aa", "string", "var")
		ab *fooImpl //@item(irAB, "ab", "*fooImpl", "var")
	)

	wantsFoo(a) //@complete(")", irAB, irAA)

	var ac fooImpl //@item(irAC, "ac", "fooImpl", "var")
	wantsFoo(&a)   //@complete(")", irAC, irAA, irAB)
}
