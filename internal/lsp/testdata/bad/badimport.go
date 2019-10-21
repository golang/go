package bad

import (
	_ "nosuchpkg" //@diag("_", "compiler", "could not import nosuchpkg (no package data for import nosuchpkg)")
)
