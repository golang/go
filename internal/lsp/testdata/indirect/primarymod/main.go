// Package indirect does something
package indirect

import (
	"example.com/extramodule/pkg"
)

func Yo() {
	var _ pkg.Test
}
