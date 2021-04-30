package a //@mark(a, "a "),hoverdef("a ", a)

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
	t := Thing{ //@mark(aStructType, "ing")
		Member: "string", //@mark(fMember, "ember")
	}
	fmt.Print(t.Member) //@mark(aMember, "ember")
	fmt.Print(Other)    //@mark(aVar, "ther")
	Things()            //@mark(aFunc, "ings")
	t.Method()          //@mark(aMethod, "eth")
}

/*@
godef(aStructType, Thing)
godef(aMember, Member)
godef(aVar, Other)
godef(aFunc, Things)
godef(aMethod, Method)
godef(fMember, Member)
godef(Member, Member)

//param
//package name
//const
//anon field

*/
