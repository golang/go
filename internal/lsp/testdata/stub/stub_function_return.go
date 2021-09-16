package stub

import (
	"io"
)

func newCloser() io.Closer {
	return closer{} //@suggestedfix("c", "refactor.rewrite")
}

type closer struct{}
