package main

// Tests of defer.  (Deferred recover() belongs is recover.go.)

import "fmt"

func deferMutatesResults(noArgReturn bool) (a, b int) {
	defer func() {
		if a != 1 || b != 2 {
			panic(fmt.Sprint(a, b))
		}
		a, b = 3, 4
	}()
	if noArgReturn {
		a, b = 1, 2
		return
	}
	return 1, 2
}

func init() {
	a, b := deferMutatesResults(true)
	if a != 3 || b != 4 {
		panic(fmt.Sprint(a, b))
	}
	a, b = deferMutatesResults(false)
	if a != 3 || b != 4 {
		panic(fmt.Sprint(a, b))
	}
}

// We concatenate init blocks to make a single function, but we must
// run defers at the end of each block, not the combined function.
var deferCount = 0

func init() {
	deferCount = 1
	defer func() {
		deferCount++
	}()
	// defer runs HERE
}

func init() {
	// Strictly speaking the spec says deferCount may be 0 or 2
	// since the relative order of init blocks is unspecified.
	if deferCount != 2 {
		panic(deferCount) // defer call has not run!
	}
}

func main() {
}
