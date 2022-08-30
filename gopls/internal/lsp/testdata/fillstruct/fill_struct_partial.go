package fillstruct

type StructPartialA struct {
	PrefilledInt int
	UnfilledInt  int
	StructPartialB
}

type StructPartialB struct {
	PrefilledInt int
	UnfilledInt  int
}

func fill() {
	a := StructPartialA{
		PrefilledInt: 5,
	} //@suggestedfix("}", "refactor.rewrite", "Fill")
	b := StructPartialB{
		/* this comment should disappear */
		PrefilledInt: 7, // This comment should be blown away.
		/* As should
		this one */
	} //@suggestedfix("}", "refactor.rewrite", "Fill")
}
