package fillstruct

import (
	h2 "net/http"

	"golang.org/lsptests/fillstruct/data"
)

func unexported() {
	a := data.B{}   //@suggestedfix("}", "refactor.rewrite", "Fill")
	_ = h2.Client{} //@suggestedfix("}", "refactor.rewrite", "Fill")
}
