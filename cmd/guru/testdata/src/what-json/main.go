package main

// Tests of 'what' queries, -format=json.
// See go.tools/guru/guru_test.go for explanation.
// See what-json.golden for expected query results.

func main() {
	f() // @what call "f"
}
