package circular

import (
	_ "golang.org/x/tools/internal/lsp/circular" //@diag("_ \"golang.org/x/tools/internal/lsp/circular\"", "compiler", "import cycle not allowed", "error"),diag("\"golang.org/x/tools/internal/lsp/circular\"", "compiler", "could not import golang.org/x/tools/internal/lsp/circular (no package for import golang.org/x/tools/internal/lsp/circular)", "error")
)
