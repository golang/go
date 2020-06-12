package fillstruct

type StructB struct {
	StructC
}

type StructC struct {
	unexportedInt int
}

func nested() {
	c := StructB{
		StructC: StructC{}, //@suggestedfix("}", "refactor.rewrite")
	}
}
