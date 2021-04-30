package b

import (
	"fmt"

	"golang.org/x/tools/internal/lsp/godef/a"
)

func useThings() {
	t := a.Thing{}      //@mark(bStructType, "ing")
	fmt.Print(t.Member) //@mark(bMember, "ember")
	fmt.Print(a.Other)  //@mark(bVar, "ther")
	a.Things()          //@mark(bFunc, "ings")
}

/*@
godef(bStructType, Thing)
godef(bMember, Member)
godef(bVar, Other)
godef(bFunc, Things)
*/

func _() {
	var x interface{}      //@mark(eInterface, "interface{}")
	switch x := x.(type) { //@hoverdef("x", eInterface)
	case string: //@mark(eString, "string")
		fmt.Println(x) //@hoverdef("x", eString)
	case int: //@mark(eInt, "int")
		fmt.Println(x) //@hoverdef("x", eInt)
	}
}
