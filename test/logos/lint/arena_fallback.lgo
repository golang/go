// errorcheck -goexperiment logosarena,logosescape -0

package p

import "logos/arena"

func Store() {
	r := arena.Open()
	p := arena.New[int](r) // ERROR "\[perf::arena_fallback\].*arena\.New\[int\].*falls back to the heap: function contains goto"
	i := 0
loop:
	i++
	if i < 3 {
		goto loop
	}
	_ = p
	_ = r
}
