package main

// Tests of channel 'peers' query.
// See go.tools/guru/guru_test.go for explanation.
// See peers.golden for expected query results.

var a2 int

func main() {
	chA := make(chan *int)
	a1 := 1
	chA <- &a1

	chA2 := make(chan *int, 2)
	if a2 == 0 {
		chA = chA2
	}

	chB := make(chan *int)
	b := 3
	chB <- &b

	<-chA  // @pointsto pointsto-chA "chA"
	<-chA2 // @pointsto pointsto-chA2 "chA2"
	<-chB  // @pointsto pointsto-chB "chB"

	select {
	case rA := <-chA: // @peers peer-recv-chA "<-"
		_ = rA // @pointsto pointsto-rA "rA"
	case rB := <-chB: // @peers peer-recv-chB "<-"
		_ = rB // @pointsto pointsto-rB "rB"

	case <-chA: // @peers peer-recv-chA' "<-"

	case chA2 <- &a2: // @peers peer-send-chA' "<-"
	}

	for range chA {
	}

	close(chA) // @peers peer-close-chA "chA"

	chC := make(chan *int)
	(close)(chC) // @peers peer-close-chC "chC"

	close := func(ch chan *int) chan *int {
		return ch
	}

	close(chC) <- &b // @peers peer-send-chC "chC"
	<-close(chC)     // @peers peer-recv-chC "chC"
}
