package main

import "testing"

func BenchmarkA(b *testing.B) {
	benchmarkB(b)
}
func benchmarkB(b *testing.B) {

	for i := 0; true; {
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
	}
}
