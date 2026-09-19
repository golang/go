// errorcheck

package main

type a = b[int] // ERROR "invalid recursive type a\n.*a refers to b\n.*b refers to a"
type b[_ any] = a