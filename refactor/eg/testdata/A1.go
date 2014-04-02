// +build ignore

package A1

import (
	. "fmt"
	myfmt "fmt"
	"os"
	"strings"
)

func example(n int) {
	x := "foo" + strings.Repeat("\t", n)
	// Match, despite named import.
	myfmt.Errorf("%s", x)

	// Match, despite dot import.
	Errorf("%s", x)

	// Match: multiple matches in same function are possible.
	myfmt.Errorf("%s", x)

	// No match: wildcarded operand has the wrong type.
	myfmt.Errorf("%s", 3)

	// No match: function operand doesn't match.
	myfmt.Printf("%s", x)

	// No match again, dot import.
	Printf("%s", x)

	// Match.
	myfmt.Fprint(os.Stderr, myfmt.Errorf("%s", x+"foo"))

	// No match: though this literally matches the template,
	// fmt doesn't resolve to a package here.
	var fmt struct{ Errorf func(string, string) }
	fmt.Errorf("%s", x)

	// Recursive matching:

	// Match: both matches are well-typed, so both succeed.
	myfmt.Errorf("%s", myfmt.Errorf("%s", x+"foo").Error())

	// Outer match succeeds, inner doesn't: 3 has wrong type.
	myfmt.Errorf("%s", myfmt.Errorf("%s", 3).Error())

	// Inner match succeeds, outer doesn't: the inner replacement
	// has the wrong type (error not string).
	myfmt.Errorf("%s", myfmt.Errorf("%s", x+"foo"))
}
