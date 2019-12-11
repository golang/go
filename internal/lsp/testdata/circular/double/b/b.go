package b

import (
	_ "golang.org/x/tools/internal/lsp/circular/double/one" //@diag("_ \"golang.org/x/tools/internal/lsp/circular/double/one\"", "go list", "import cycle not allowed")
)
