package stub

import (
	"io"
	"sort"
)

var _ embeddedInterface = (*embeddedConcrete)(nil) //@suggestedfix("(", "quickfix", "")

type embeddedConcrete struct{}

type embeddedInterface interface {
	sort.Interface
	io.Reader
}
