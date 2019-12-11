package circular

import (
	_ "golang.org/x/tools/internal/lsp/circular" //@diag("_ \"golang.org/x/tools/internal/lsp/circular\"", "go list", "import cycle not allowed")
)
