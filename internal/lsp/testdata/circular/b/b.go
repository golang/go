package b //@diag("", "go list", "import cycle not allowed")

import (
	"golang.org/x/tools/internal/lsp/circular/one"
)

func Test1() {
	one.Test()
}
