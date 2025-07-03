package test

import (
	"cmd/go/internal/imports/testdata/test/child"
	"fmt"
)

func F() {
	fmt.Println(child.V)
}
