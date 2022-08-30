package stub

import "io"

// This file tests that if an interface
// method references a type from its own package
// then our implementation must add the import/package selector
// in the concrete method if the concrete type is outside of the interface
// package
var _ io.ReaderFrom = &readerFrom{} //@suggestedfix("&readerFrom", "refactor.rewrite", "")

type readerFrom struct{}
