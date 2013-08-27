package peers

// Tests of channel 'peers' query.
// See go.tools/oracle/oracle_test.go for explanation.
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

	<-chA  // @describe describe-chA "chA"
	<-chA2 // @describe describe-chA2 "chA2"
	<-chB  // @describe describe-chB "chB"

	select {
	case rA := <-chA: // @peers peer-recv-chA "<-"
		_ = rA // @describe describe-rA "rA"
	case rB := <-chB: // @peers peer-recv-chB "<-"
		_ = rB // @describe describe-rB "rB"

	case <-chA: // @peers peer-recv-chA' "<-"

	case chA2 <- &a2: // @peers peer-send-chA' "<-"
	}

	for _ = range chA {
	}
}
