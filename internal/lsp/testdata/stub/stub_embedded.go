package stub

import (
	"io"
	"sort"
)

var _ embeddedInterface = (*embeddedConcrete)(nil) //@suggestedfix("(", "refactor.rewrite")

type embeddedConcrete struct{}

type embeddedInterface interface {
	sort.Interface
	io.Reader
}
