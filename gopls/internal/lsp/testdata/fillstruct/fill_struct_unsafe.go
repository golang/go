package fillstruct

import "unsafe"

type unsafeStruct struct {
	x int
	p unsafe.Pointer
}

func fill() {
	_ := unsafeStruct{} //@suggestedfix("}", "refactor.rewrite", "Fill")
}
