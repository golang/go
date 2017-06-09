// The package doc comment
package initial

import (
	"fmt"

	"domain.name/importdecl"
)

type t int // type1

// const1
const c = 1 // const2

func foo() {
	fmt.Println(importdecl.F())
}

// zinit
const (
	z1 = iota // z1
	z2        // z2
) // zend
