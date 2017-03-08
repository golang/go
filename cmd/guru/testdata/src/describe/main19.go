package describe

// The behavior of "describe" on a non-existent import changed
// when go/types started returning fake packages, so this test
// is executed only under go1.9.

import (
	"nosuchpkg"            // @describe badimport1 "nosuchpkg"
	nosuchpkg2 "nosuchpkg" // @describe badimport2 "nosuchpkg2"
)

var _ nosuchpkg.T
var _ nosuchpkg2.T
