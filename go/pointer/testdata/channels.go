//go:build ignore
// +build ignore

package main

func incr(x int) int { return x + 1 }

func decr(x int) int { return x - 1 }

var unknown bool // defeat dead-code elimination

func chan1() {
	chA := make(chan func(int) int, 0) // @line c1makeA
	chB := make(chan func(int) int, 0) // @line c1makeB
	chA <- incr
	chB <- decr
	chB <- func(int) int { return 1 }

	print(chA)   // @pointsto makechan@c1makeA:13
	print(<-chA) // @pointsto command-line-arguments.incr

	print(chB)   // @pointsto makechan@c1makeB:13
	print(<-chB) // @pointsto command-line-arguments.decr | command-line-arguments.chan1$1
}

func chan2() {
	chA := make(chan func(int) int, 0) // @line c2makeA
	chB := make(chan func(int) int, 0) // @line c2makeB
	chA <- incr
	chB <- decr
	chB <- func(int) int { return 1 }

	// Channels flow together.
	// Labelsets remain distinct but elements are merged.
	chAB := chA
	if unknown {
		chAB = chB
	}

	print(chA)   // @pointsto makechan@c2makeA:13
	print(<-chA) // @pointsto command-line-arguments.incr

	print(chB)   // @pointsto makechan@c2makeB:13
	print(<-chB) // @pointsto command-line-arguments.decr | command-line-arguments.chan2$1

	print(chAB)   // @pointsto makechan@c2makeA:13 | makechan@c2makeB:13
	print(<-chAB) // @pointsto command-line-arguments.incr | command-line-arguments.decr | command-line-arguments.chan2$1

	(<-chA)(3)
}

// @calls command-line-arguments.chan2 -> command-line-arguments.incr

func chan3() {
	chA := make(chan func(int) int, 0) // @line c3makeA
	chB := make(chan func(int) int, 0) // @line c3makeB
	chA <- incr
	chB <- decr
	chB <- func(int) int { return 1 }
	print(chA)   // @pointsto makechan@c3makeA:13
	print(<-chA) // @pointsto command-line-arguments.incr
	print(chB)   // @pointsto makechan@c3makeB:13
	print(<-chB) // @pointsto command-line-arguments.decr | command-line-arguments.chan3$1

	(<-chA)(3)
}

// @calls command-line-arguments.chan3 -> command-line-arguments.incr

func chan4() {
	chA := make(chan func(int) int, 0) // @line c4makeA
	chB := make(chan func(int) int, 0) // @line c4makeB

	select {
	case chA <- incr:
	case chB <- decr:
	case a := <-chA:
		print(a) // @pointsto command-line-arguments.incr
	case b := <-chB:
		print(b) // @pointsto command-line-arguments.decr
	default:
		print(chA) // @pointsto makechan@c4makeA:13
		print(chB) // @pointsto makechan@c4makeB:13
	}

	for k := range chA {
		print(k) // @pointsto command-line-arguments.incr
	}
	// Exercise constraint generation (regtest for a crash).
	for range chA {
	}
}

// Multi-word channel value in select with multiple receive cases.
// (Regtest for a crash.)
func chan5() {
	type T struct {
		x *int
		y interface{}
	}
	ch := make(chan T)
	ch <- T{new(int), incr} // @line ch5new
	select {
	case a := <-ch:
		print(a.x) // @pointsto new@ch5new:13
		print(a.y) // @types func(x int) int
	case b := <-ch:
		print(b.x) // @pointsto new@ch5new:13
		print(b.y) // @types func(x int) int
	}
}

func main() {
	chan1()
	chan2()
	chan3()
	chan4()
	chan5()
}
