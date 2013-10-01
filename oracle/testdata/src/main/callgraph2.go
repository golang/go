package main

// Tests of call-graph queries.
// See go.tools/oracle/oracle_test.go for explanation.
// See callgraph2.golden for expected query results.

// (Regression test for pointer analysis: programs that use reflection
// create some cgnodes before the root of the callgraph.)
import _ "reflect"

func f() {}
func main() {
	f()
}

// @callgraph callgraph "^"
