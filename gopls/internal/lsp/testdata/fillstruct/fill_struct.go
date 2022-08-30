package fillstruct

type StructA struct {
	unexportedIntField int
	ExportedIntField   int
	MapA               map[int]string
	Array              []int
	StructB
}

type StructA2 struct {
	B *StructB
}

type StructA3 struct {
	B StructB
}

func fill() {
	a := StructA{}  //@suggestedfix("}", "refactor.rewrite", "Fill")
	b := StructA2{} //@suggestedfix("}", "refactor.rewrite", "Fill")
	c := StructA3{} //@suggestedfix("}", "refactor.rewrite", "Fill")
	if true {
		_ = StructA3{} //@suggestedfix("}", "refactor.rewrite", "Fill")
	}
}
