package stub

import (
	"golang.org/lsptests/stub/other"
)

// This file tests that if an interface
// method references an import from its own package
// that the concrete type does not yet import, and that import happens
// to be renamed, then we prefer the renaming of the interface.
var _ other.Interface = &otherInterfaceImpl{} //@suggestedfix("&otherInterfaceImpl", "refactor.rewrite", "")

type otherInterfaceImpl struct{}
