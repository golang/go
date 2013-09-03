package main

// Tests of call-graph queries.
// See go.tools/oracle/oracle_test.go for explanation.
// See callgraph.golden for expected query results.

func A() {}

func B() {}

// call is not (yet) treated context-sensitively.
func call(f func()) {
	f()
}

// nop *is* treated context-sensitively.
func nop() {}

func call2(f func()) {
	f()
	f()
}

func main() {
	call(A)
	call(B)

	nop()
	nop()

	call2(func() {
		// called twice from main.call2,
		// but call2 is not context sensitive (yet).
	})

	print("builtin")
	_ = string("type conversion")
	call(nil)
	if false {
		main()
	}
	var nilFunc func()
	nilFunc()
	var i interface {
		f()
	}
	i.f()
}

func deadcode() {
	main()
}

// @callgraph callgraph "^"
