package main

// Tests of channel 'peers' query, -format=json.
// See go.tools/oracle/oracle_test.go for explanation.
// See peers-json.golden for expected query results.

func main() {
	chA := make(chan *int)
	<-chA
	select {
	case <-chA: // @peers peer-recv-chA "<-"
	}
}
