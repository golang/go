package stub

import (
	"io"
)

var _ io.Writer = writer{} //@suggestedfix("w", "refactor.rewrite", "")

type writer struct{}
