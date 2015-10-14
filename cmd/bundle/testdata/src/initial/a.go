package initial

import "fmt"

// init functions are not renamed
func init() { foo() }

// Type S.
type S struct {
	t
	u int
}

// Function bar.
func bar(s *S) { fmt.Println(s.t, s.u) }
