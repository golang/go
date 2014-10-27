package main

func main() {
	// Regression test for golang issue 9002.
	//
	// The two-result "value,ok" receive operation generated a
	// too-wide constraint loading (value int, ok bool), not bool,
	// from the channel.
	//
	// This bug manifested itself in an out-of-bounds array access
	// when the makechan object was the highest-numbered node, as in
	// this program.
	//
	// In more realistic programs it silently resulted in bogus
	// constraints.
	_, _ = <-make(chan int)
}
