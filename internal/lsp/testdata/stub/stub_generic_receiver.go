//go:build go1.18
// +build go1.18

package stub

import "io"

// This file tests that that the stub method generator accounts for concrete
// types that have type parameters defined.
var _ io.ReaderFrom = &genReader[string, int]{} //@suggestedfix("&genReader", "refactor.rewrite")

type genReader[T, Y any] struct {
	T T
	Y Y
}
