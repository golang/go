package initial

import "fmt" // this comment should not be visible

// init functions are not renamed
func init() { foo() }

// Type S.
type S struct {
	t
	u int
} /* multi-line
comment
*/

// non-associated comment

/*
	non-associated comment2
*/

// Function bar.
func bar(s *S) {
	fmt.Println(s.t, s.u) // comment inside function
}

// file-end comment
