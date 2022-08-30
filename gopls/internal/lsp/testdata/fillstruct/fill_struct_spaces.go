package fillstruct

type StructD struct {
	ExportedIntField int
}

func spaces() {
	d := StructD{} //@suggestedfix("}", "refactor.rewrite", "Fill")
}
