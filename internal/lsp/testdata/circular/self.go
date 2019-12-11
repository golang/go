package circular //@diag("", "go list", "import cycle not allowed: import stack: [golang.org/x/tools/internal/lsp/circular golang.org/x/tools/internal/lsp/circular]")

import (
	"golang.org/x/tools/internal/lsp/circular"
)

func print() {
	Test()
}

func Test() {
}
