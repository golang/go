package stub

import "io"

func getReaderFrom() io.ReaderFrom {
	return &pointerImpl{} //@suggestedfix("&", "refactor.rewrite", "")
}

type pointerImpl struct{}
