package fillstruct

import (
	"golang.org/x/tools/internal/lsp/fillstruct/data"
)

func unexported() {
	a := data.A{} //@suggestedfix("}", "refactor.rewrite")
}
