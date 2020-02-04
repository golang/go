package b

import (
	_ "golang.org/x/tools/internal/lsp/circular/double/one" //@diag("_ \"golang.org/x/tools/internal/lsp/circular/double/one\"", "compiler", "import cycle not allowed", "error"),diag("\"golang.org/x/tools/internal/lsp/circular/double/one\"", "compiler", "could not import golang.org/x/tools/internal/lsp/circular/double/one (no package for import golang.org/x/tools/internal/lsp/circular/double/one)", "error")
)
