package circular //@diag("", "go list", "import cycle not allowed")

import (
	"golang.org/x/tools/internal/lsp/circular"
)

func print() {
	Test()
}

func Test() {
}
