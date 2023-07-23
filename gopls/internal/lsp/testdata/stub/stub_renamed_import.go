package stub

import (
	"compress/zlib"
	myio "io"
)

var _ zlib.Resetter = &myIO{} //@suggestedfix("&", "quickfix", "")
var _ myio.Reader

type myIO struct{}
