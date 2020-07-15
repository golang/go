// Package missingtwodep does something
package missingtwodep

import (
	"example.com/anothermodule/hey" //@diag("\"example.com/anothermodule/hey\"", "go mod tidy", "example.com/anothermodule is not in your go.mod file", "warning"),suggestedfix("\"example.com/anothermodule/hey\"", "quickfix")
	"example.com/extramodule/pkg"   //@diag("\"example.com/extramodule/pkg\"", "go mod tidy", "example.com/extramodule is not in your go.mod file", "warning"),suggestedfix("\"example.com/extramodule/pkg\"", "quickfix")
	"example.com/extramodule/yo"    //@diag("\"example.com/extramodule/yo\"", "go mod tidy", "example.com/extramodule is not in your go.mod file", "warning"),suggestedfix("\"example.com/extramodule/yo\"", "quickfix")
)

func Yo() {
	_ = pkg.Test
	_ = yo.Test
	_ = hey.Test
}
