package stub

import "io"

// This test ensures that a variable declaration that
// has multiple values on the same line can still be
// analyzed correctly to target the interface implementation
// diagnostic.
var one, two, three io.Reader = nil, &multiVar{}, nil //@suggestedfix("&", "refactor.rewrite")

type multiVar struct{}
