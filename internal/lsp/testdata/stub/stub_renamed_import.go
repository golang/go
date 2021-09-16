package stub

import (
	"compress/zlib"
	myio "io"
)

var _ zlib.Resetter = &myIO{} //@suggestedfix("&", "refactor.rewrite")
var _ myio.Reader

type myIO struct{}
