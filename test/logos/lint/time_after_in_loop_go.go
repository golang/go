// errorcheck -0

// perf::time_after_in_loop on a Go package (R-U1): logos lint applies the
// rule to Go syntax and Go types alone, so the Go twin of the loGo/s attack
// file reports the same loops.
//
// correctness::unbounded_channel_op also reports the channel receives of this
// file. They are not its subject, so each carries //logos:allow for that rule
// and the file checks perf::time_after_in_loop alone;
// test/logos/lint/unbounded_channel_op_go.go covers the rule.

package p

import "time"

type clock struct{}

func (clock) After(d time.Duration) <-chan time.Time { return nil }

func After(d time.Duration) <-chan time.Time { return nil }

func selectLoop(d time.Duration, ch <-chan int) {
	for {
		//logos:allow(correctness::unbounded_channel_op)
		select {
		case <-ch:
			return
		case <-time.After(d): // ERROR "\[perf::time_after_in_loop\] time.After in a loop body"
		}
	}
}

func threeClause(d time.Duration, n int) {
	for i := 0; i < n; i++ {
		//logos:allow(correctness::unbounded_channel_op)
		<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
	}
}

func rangeLoop(d time.Duration, xs []int) {
	for range xs {
		//logos:allow(correctness::unbounded_channel_op)
		<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
	}
	for i := range 3 {
		_ = i
		//logos:allow(correctness::unbounded_channel_op)
		<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
	}
}

func conditionLoop(d time.Duration, ok func() bool) {
	for ok() {
		//logos:allow(correctness::unbounded_channel_op)
		<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
	}
}

func nestedLoops(d time.Duration) {
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			//logos:allow(correctness::unbounded_channel_op)
			<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
		}
		//logos:allow(correctness::unbounded_channel_op)
		<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
	}
}

func closureInLoop(d time.Duration) {
	for i := 0; i < 2; i++ {
		func() {
			//logos:allow(correctness::unbounded_channel_op)
			<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
		}()
	}
}

func loopInClosure(d time.Duration) func() {
	return func() {
		for {
			//logos:allow(correctness::unbounded_channel_op)
			<-time.After(d) // ERROR "\[perf::time_after_in_loop\]"
		}
	}
}

func twoCallsOneBody(d time.Duration) {
	for {
		a := time.After(d) // ERROR "\[perf::time_after_in_loop\]"
		b := time.After(d) // ERROR "\[perf::time_after_in_loop\]"
		_, _ = a, b
	}
}

//logos:allow(correctness::unbounded_channel_op)
func drain(c <-chan time.Time) { <-c }

func argumentInLoop(d time.Duration) {
	for i := 0; i < 2; i++ {
		drain(time.After(d)) // ERROR "\[perf::time_after_in_loop\]"
	}
}

func outsideLoop(d time.Duration) {
	//logos:allow(correctness::unbounded_channel_op)
	<-time.After(d)
	t := time.NewTimer(d)
	for i := 0; i < 3; i++ {
		t.Reset(d)
		//logos:allow(correctness::unbounded_channel_op)
		<-t.C
	}
	_ = time.AfterFunc(d, func() {})
}

func headerRunsOnce(d time.Duration) {
	for t := time.After(d); ; {
		//logos:allow(correctness::unbounded_channel_op)
		<-t
		break
	}
	for range time.After(d) {
		break
	}
}

func notTimeAfter(d time.Duration) {
	var c clock
	now := time.Now()
	for i := 0; i < 2; i++ {
		_ = now.After(now)
		//logos:allow(correctness::unbounded_channel_op)
		<-c.After(d)
		//logos:allow(correctness::unbounded_channel_op)
		<-After(d)
		_ = time.Tick
		f := time.After
		_ = f
	}
}

func allowed(d time.Duration) {
	for {
		//logos:allow(correctness::unbounded_channel_op)
		//logos:allow(perf::time_after_in_loop)
		<-time.After(d)
	}
}
