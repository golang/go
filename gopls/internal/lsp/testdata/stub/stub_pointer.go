package stub

import "io"

func getReaderFrom() io.ReaderFrom {
	return &pointerImpl{} //@suggestedfix("&", "quickfix", "")
}

type pointerImpl struct{}
