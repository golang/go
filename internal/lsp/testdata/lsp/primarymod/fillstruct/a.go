package fillstruct

import (
	"golang.org/x/tools/internal/lsp/fillstruct/data"
)

type basicStruct struct {
	foo int
}

var _ = basicStruct{} //@suggestedfix("}", "refactor.rewrite")

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} //@suggestedfix("}", "refactor.rewrite")

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} //@suggestedfix("}", "refactor.rewrite")

var _ = data.B{} //@suggestedfix("}", "refactor.rewrite")
