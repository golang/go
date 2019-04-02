package a

import "fmt"

type Thing struct { //@Thing
	Member string //@Member
}

var Other Thing //@Other

func Things(val []string) []Thing { //@Things
	return nil
}

func (t Thing) Method(i int) string { //@Method
	return t.Member
}

func useThings() {
	t := Thing{}        //@mark(aStructType, "ing")
	fmt.Print(t.Member) //@mark(aMember, "ember")
	fmt.Print(Other)    //@mark(aVar, "ther")
	Things()            //@mark(aFunc, "ings")
	t.Method()          //@mark(aMethod, "eth")
}

/*@
definition(aStructType, "", Thing, "$file:$line:$col-$ecol: defined here as type Thing struct{Member string}")
definition(aStructType, "-emulate=guru", Thing, "$file:$line:$col: defined here as type Thing")

definition(aMember, "", Member, "$file:$line:$col-$ecol: defined here as field Member string")
definition(aMember, "-emulate=guru", Member, "$file:$line:$col: defined here as field Member string")

definition(aVar, "", Other, "$file:$line:$col-$ecol: defined here as var Other Thing")
definition(aVar, "-emulate=guru", Other, "$file:$line:$col: defined here as var Other")

definition(aFunc, "", Things, "$file:$line:$col-$ecol: defined here as func Things(val []string) []Thing")
definition(aFunc, "-emulate=guru", Things, "$file:$line:$col: defined here as func Things")

definition(aMethod, "", Method, "$file:$line:$col-$ecol: defined here as func (Thing).Method(i int) string")
definition(aMethod, "-emulate=guru", Method, "$file:$line:$col: defined here as func (Thing).Method(i int) string")

//param
//package name
//const
//anon field

// JSON tests

definition(aStructType, "-json", Thing, `{
	"span": {
		"uri": "$euri",
		"start": {
			"line": $line,
			"column": $col,
			"offset": $offset
		},
		"end": {
			"line": $eline,
			"column": $ecol,
			"offset": $eoffset
		}
	},
	"description": "type Thing struct{Member string}"
}`)
definition(aStructType, "-json -emulate=guru", Thing, `{
	"objpos": "$efile:$line:$col",
	"desc": "type Thing$$"
}`)
*/
