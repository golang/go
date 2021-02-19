package C1

import "strings"

func example() {
	x := "foo"
	println(x[:len(x)])

	// Match, but the transformation is not sound w.r.t. possible side effects.
	println(strings.Repeat("*", 3)[:len(strings.Repeat("*", 3))])

	// No match, since second use of wildcard doesn't match first.
	println(strings.Repeat("*", 3)[:len(strings.Repeat("*", 2))])

	// Recursive match demonstrating bottom-up rewrite:
	// only after the inner replacement occurs does the outer syntax match.
	println((x[:len(x)])[:len(x[:len(x)])])
	// -> (x[:len(x)])
	// -> x
}
