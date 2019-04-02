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
definition(bStructType, "", Thing, "$file:$line:$col-$ecol: defined here as type a.Thing struct{Member string}")
definition(bStructType, "-emulate=guru", Thing, "$file:$line:$col: defined here as type $$a.Thing")

definition(bMember, "", Member, "$file:$line:$col-$ecol: defined here as field Member string")
definition(bMember, "-emulate=guru", Member, "$file:$line:$col: defined here as field Member string")

definition(bVar, "", Other, "$file:$line:$col-$ecol: defined here as var a.Other a.Thing")
definition(bVar, "-emulate=guru", Other, "$file:$line:$col: defined here as var $$a.Other")

definition(bFunc, "", Things, "$file:$line:$col-$ecol: defined here as func a.Things(val []string) []a.Thing")
definition(bFunc, "-emulate=guru", Things, "$file:$line:$col: defined here as func $$a.Things")
*/
