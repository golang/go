// Package a is a package for testing go to definition.
package a //@mark(aPackage, "a "),hover("a ", aPackage)

import (
	"fmt"
	"go/types"
	"sync"
)

var (
	// x is a variable.
	x string //@x,hover("x", x)
)

// z is a variable too.
var z string //@z,hover("z", z)

type A string //@mark(AString, "A")

func AStuff() { //@AStuff
	x := 5
	Random2(x) //@godef("dom2", Random2)
	Random()   //@godef("()", Random)

	var err error         //@err
	fmt.Printf("%v", err) //@godef("err", err)

	var y string       //@string,hover("string", string)
	_ = make([]int, 0) //@make,hover("make", make)

	var mu sync.Mutex
	mu.Lock() //@Lock,hover("Lock", Lock)

	var typ *types.Named //@mark(typesImport, "types"),hover("types", typesImport)
	typ.Obj().Name()     //@Name,hover("Name", Name)
}
