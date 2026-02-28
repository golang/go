// errorcheck -0 -d=ssa/prove/debug=1

//go:build amd64 || arm64

package main

func invert(b func(), n int) {
	for i := 0; i < n; i++ { // ERROR "(Inverted loop iteration|Induction variable: limits \[0,\?\), increment 1)"
		b()
	}
}
