// Package invalidgo does something
package invalidgo

import (
	"golang.org/x/tools/go/packages"
)

func Yo() {
	var _ packages.Config
}
