package fillstruct

import (
	data "golang.org/x/tools/internal/lsp/fillstruct/data"
)

func unexported() {
	a := data.A{} //@refactorrewrite("}", "Fill struct")
}
