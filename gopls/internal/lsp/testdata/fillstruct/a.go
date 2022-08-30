package fillstruct

import (
	"golang.org/lsptests/fillstruct/data"
)

type basicStruct struct {
	foo int
}

var _ = basicStruct{} //@suggestedfix("}", "refactor.rewrite", "Fill")

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} //@suggestedfix("}", "refactor.rewrite", "Fill")

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} //@suggestedfix("}", "refactor.rewrite", "Fill")

var _ = data.B{} //@suggestedfix("}", "refactor.rewrite", "Fill")
