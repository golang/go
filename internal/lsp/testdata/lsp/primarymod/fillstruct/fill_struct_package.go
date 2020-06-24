package fillstruct

import (
	h2 "net/http"

	"golang.org/x/tools/internal/lsp/fillstruct/data"
)

func unexported() {
	a := data.A{}   //@suggestedfix("}", "refactor.rewrite")
	_ = h2.Client{} //@suggestedfix("}", "refactor.rewrite")
}
