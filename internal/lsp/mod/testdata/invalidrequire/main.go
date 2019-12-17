// Package invalidrequire does something
package invalidrequire

import (
	"golang.org/x/tools/go/packages"
)

func Yo() {
	var _ packages.Config
}
