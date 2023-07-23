package stub

import (
	"io"
)

var _ io.Writer = writer{} //@suggestedfix("w", "quickfix", "")

type writer struct{}
