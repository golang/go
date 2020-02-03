package funcsig

type someType int //@item(sigSomeType, "someType", "int", "type")

// Don't complete "foo" in signature.
func (foo someType) _() { //@item(sigFoo, "foo", "someType", "var"),complete(") {", sigSomeType)

	//@complete("", sigFoo, sigSomeType)
}
