// Package missingdep does something
package missingdep

import (
	"fmt"

	"example.com/extramodule/pkg" //@diag("\"example.com/extramodule/pkg\"", "go mod tidy", "example.com/extramodule is not in your go.mod file", "warning"),suggestedfix("\"example.com/extramodule/pkg\"", "quickfix")
)

func Yo() {
	_ = pkg.Test
	fmt.Printf("%s") //@diag("fmt.Printf(\"%s\")", "printf", "Printf format %s reads arg #1, but call has 0 args", "warning")
}
