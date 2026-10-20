// errorcheck

package p

import (
	"context"
	"time"
)

type hub struct{ ch chan int }

type task struct{ done chan struct{} }

func (t task) Done() <-chan struct{} { return t.done }

const slots = 2

func paramSend(ch chan int) {
	ch <- 1 // ERROR "unbounded_channel_op"
}

func paramRecv(ch chan int) int {
	return <-ch // ERROR "unbounded_channel_op"
}

func assignRecv(ch chan int) int {
	v := <-ch // ERROR "unbounded_channel_op"
	return v
}

func commaOkRecv(ch chan int) bool {
	_, ok := <-ch // ERROR "unbounded_channel_op"
	return ok
}

func statementRecv(ch chan int) {
	<-ch // ERROR "unbounded_channel_op"
}

func fieldSend(h *hub) {
	h.ch <- 1 // ERROR "unbounded_channel_op"
}

func indexSend(chs []chan int) {
	chs[0] <- 1 // ERROR "unbounded_channel_op"
}

func selectNoDone(a, b chan int) int {
	select {
	case v := <-a: // ERROR "unbounded_channel_op"
		return v
	case b <- 1: // ERROR "unbounded_channel_op"
		return 0
	}
}

func selectDefault(a chan int) {
	select {
	case a <- 1: // ERROR "unbounded_channel_op"
	default:
	}
}

func bodyInDoneSelect(ctx context.Context, in, out chan int) {
	for {
		select {
		case <-ctx.Done():
			return
		case v := <-in:
			out <- v // ERROR "unbounded_channel_op"
		}
	}
}

func doneSelect(ctx context.Context, in chan int) int {
	select {
	case <-ctx.Done():
		return 0
	case v := <-in:
		return v
	}
}

func doneSelectSend(ctx context.Context, out chan int) {
	select {
	case out <- 1:
	case <-ctx.Done():
	}
}

func doneAssign(ctx context.Context, in chan int) {
	select {
	case _, ok := <-ctx.Done():
		_ = ok
	case <-in:
	}
}

func customDone(t task, in chan int) int {
	select {
	case <-t.Done():
		return 0
	case v := <-in:
		return v
	}
}

func waitDone(ctx context.Context) {
	<-ctx.Done()
}

func belowCap() int {
	ch := make(chan int, 2)
	ch <- 1
	return <-ch
}

func atCap() int {
	ch := make(chan int, 2)
	ch <- 1
	ch <- 2
	return <-ch + <-ch
}

func overCap() int {
	ch := make(chan int, 2)
	ch <- 1 // ERROR "unbounded_channel_op"
	ch <- 2 // ERROR "unbounded_channel_op"
	ch <- 3 // ERROR "unbounded_channel_op"
	return <-ch // ERROR "unbounded_channel_op"
}

func zeroCap() int {
	ch := make(chan int, 0)
	ch <- 1 // ERROR "unbounded_channel_op"
	return <-ch // ERROR "unbounded_channel_op"
}

func unbufferedNoSend() int {
	ch := make(chan int)
	return <-ch
}

func sendInLoop() {
	ch := make(chan int, 2)
	for i := 0; i < 2; i++ {
		ch <- i // ERROR "unbounded_channel_op"
	}
	<-ch // ERROR "unbounded_channel_op"
}

func sendInRange() {
	ch := make(chan int, 2)
	for _, v := range []int{1, 2} {
		ch <- v // ERROR "unbounded_channel_op"
	}
}

func recvInLoop() {
	ch := make(chan int, 2)
	ch <- 1
	for i := 0; i < 2; i++ {
		<-ch
	}
}

func dynamicCap(n int) {
	ch := make(chan int, n)
	ch <- 1 // ERROR "unbounded_channel_op"
}

func constCap() {
	ch := make(chan int, slots)
	ch <- 1
	ch <- 2
}

func exprCap() {
	ch := make(chan int, 1+1)
	ch <- 1
	ch <- 2
}

func reassignedChan() {
	ch := make(chan int, 2)
	ch = make(chan int, 2)
	ch <- 1 // ERROR "unbounded_channel_op"
}

func closureOver() {
	ch := make(chan int, 2)
	ch <- 1 // ERROR "unbounded_channel_op"
	ch <- 2 // ERROR "unbounded_channel_op"
	func() {
		ch <- 3 // ERROR "unbounded_channel_op"
	}()
}

func closureWithin() {
	ch := make(chan int, 2)
	ch <- 1
	func() {
		ch <- 2
	}()
}

func closureInLoop() {
	ch := make(chan int, 4)
	for i := 0; i < 2; i++ {
		func() {
			ch <- i // ERROR "unbounded_channel_op"
		}()
	}
}

func selectCounts() {
	ch := make(chan int, 1)
	ch <- 1 // ERROR "unbounded_channel_op"
	select {
	case ch <- 2: // ERROR "unbounded_channel_op"
	default:
	}
}

func varDecl() {
	var ch = make(chan int, 1)
	ch <- 1
}

func varDeclType() {
	var ch chan int = make(chan int, 1)
	ch <- 1
}

func declNoMake() {
	var ch chan int
	ch <- 1 // ERROR "unbounded_channel_op"
}

func rangeChan(ch chan int) int {
	n := 0
	for v := range ch {
		n += v
	}
	return n
}

func closeAndLen(ch chan int) int {
	close(ch)
	return len(ch)
}

//logos:allow(correctness::unbounded_channel_op)
func allowedSend(ch chan int) {
	ch <- 1
}

func timerRecv() {
	<-time.After(time.Second) // ERROR "unbounded_channel_op"
}

func timerSelect(in chan int) int {
	select {
	case v := <-in: // ERROR "unbounded_channel_op"
		return v
	case <-time.After(time.Second): // ERROR "unbounded_channel_op"
		return 0
	}
}

func tickRecv() {
	<-time.Tick(time.Second) // ERROR "unbounded_channel_op"
}
