package bad

import (
	_ "nosuchpkg" //@diag("_", "LSP", "could not import nosuchpkg (no package data for import path nosuchpkg)")
)
