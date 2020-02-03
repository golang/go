// Package indirect does something
package indirect

import (
	"golang.org/x/tools/go/packages"
)

func Yo() {
	var _ packages.Config
}
