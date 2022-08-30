package stub

import (
	"compress/zlib"
	. "io"
	_ "io"
)

// This file tests that dot-imports and underscore imports
// are properly ignored and that a new import is added to
// reference method types

var (
	_ Reader
	_ zlib.Resetter = (*ignoredResetter)(nil) //@suggestedfix("(", "refactor.rewrite", "")
)

type ignoredResetter struct{}
