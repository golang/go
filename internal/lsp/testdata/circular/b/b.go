package b //@diag("", "go list", "import cycle not allowed: import stack: [golang.org/x/tools/internal/lsp/circular/b golang.org/x/tools/internal/lsp/circular/one golang.org/x/tools/internal/lsp/circular/b]")

import (
	"golang.org/x/tools/internal/lsp/circular/one"
)

func Test1() {
	one.Test()
}
