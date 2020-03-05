// Package missingdep does something
package missingdep

import (
	"example.com/extramodule/pkg" //@diag("\"example.com/extramodule/pkg\"", "go mod tidy", "example.com/extramodule is not in your go.mod file.", "warning"),suggestedfix("\"example.com/extramodule/pkg\"", "quickfix")
)

func Yo() {
	_ = pkg.Test
}
