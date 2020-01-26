package a

import (
	_ "golang.org/x/tools/internal/lsp/circular/triple/b" //@diag("_ \"golang.org/x/tools/internal/lsp/circular/triple/b\"", "compiler", "import cycle not allowed")
)
