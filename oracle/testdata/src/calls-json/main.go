package main

// Tests of call-graph queries, -format=json.
// See go.tools/oracle/oracle_test.go for explanation.
// See calls-json.golden for expected query results.

func call(f func()) {
	f() // @callees @callees-f "f"
}

func main() {
	call(func() {
		// @callers callers-main.anon "^"
		// @callstack callstack-main.anon "^"
	})
}
