package fillstruct

type StructAnon struct {
	a struct{}
	b map[string]interface{}
	c map[string]struct {
		d int
		e bool
	}
}

func fill() {
	_ := StructAnon{} //@suggestedfix("}", "refactor.rewrite")
}
