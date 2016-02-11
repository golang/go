package main // @what pkgdecl "main"

// Tests of 'what' queries.
// See go.tools/guru/guru_test.go for explanation.
// See what.golden for expected query results.

func main() {
	f()             // @what call "f"
	var ch chan int // @what var "var"
	<-ch            // @what recv "ch"
}
