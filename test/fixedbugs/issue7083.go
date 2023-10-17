// run

package main

import "runtime/debug"

func f(m map[int]*string, i int) {
	s := ""
	m[i] = &s
}

func main() {
	debug.SetGCPercent(0)
	m := map[int]*string{}
	for i := 0; i < 40; i++ {
		f(m, i)
		if len(*m[i]) != 0 {
			println("bad length", i, m[i], len(*m[i]))
			panic("bad length")
		}
	}
}
