package main

// Regression test for https://golang.org/issue/23925

type stringFlagImpl string

func (*stringFlagImpl) Set(s string) error { return nil }

type boolFlagImpl bool

func (*boolFlagImpl) Set(s string) error { return nil }
func (*boolFlagImpl) extra()             {}

// A copy of flag.boolFlag interface, without a dependency.
// Must appear first, so that it becomes the owner of the Set methods.
type boolFlag interface {
	flagValue
	extra()
}

// A copy of flag.Value, without adding a dependency.
type flagValue interface {
	Set(string) error
}

func main() {
	var x flagValue = new(stringFlagImpl)
	x.Set("")

	var y boolFlag = new(boolFlagImpl)
	y.Set("")
}

// WANT:
// Dynamic calls
//   main --> (*boolFlagImpl).Set
//   main --> (*boolFlagImpl).Set
//   main --> (*stringFlagImpl).Set
