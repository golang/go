package a

import (
	_ "golang.org/x/tools/internal/lsp/circular/triple/b" //@diag("_ \"golang.org/x/tools/internal/lsp/circular/triple/b\"", "go list", "import cycle not allowed")
)
