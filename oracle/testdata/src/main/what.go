package what // @what pkgdecl "what"

// Tests of 'what' queries.
// See go.tools/oracle/oracle_test.go for explanation.
// See what.golden for expected query results.

func main() {
	f()             // @what call "f"
	var ch chan int // @what var "var"
	<-ch            // @what recv "ch"
}
