// errorcheck

package p

import "logos/sync"

type Box struct {
	n int
}

func (b *Box) Touch() int {
	return b.n
}

func useBox(b *Box) {}

var leaked *Box

var holder struct {
	b *Box
}

var boxCh chan *Box

var getBox func() *Box

func storeVar(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		leaked = b // ERROR "\[correctness::withlock_leak\]"
	})
}

func storeCaptured(l *sync.Locked[Box]) *Box {
	held := &Box{}
	l.WithLock(func(b *Box) {
		held = b // ERROR "\[correctness::withlock_leak\]"
	})
	return held
}

func storeField(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		holder.b = b // ERROR "\[correctness::withlock_leak\]"
	})
}

func storeChan(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		//logos:allow(correctness::unbounded_channel_op)
		boxCh <- b // ERROR "\[correctness::withlock_leak\]"
	})
}

func storeAppend(l *sync.Locked[Box]) []*Box {
	buf := []*Box{}
	l.WithLock(func(b *Box) {
		buf = append(buf, b) // ERROR "\[correctness::withlock_leak\]"
	})
	return buf
}

func storeGo(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		go useBox(b) // ERROR "\[correctness::withlock_leak\]"
	})
}

func storeGetter(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		getBox = func() *Box { // ERROR "\[correctness::withlock_leak\]"
			return b // ERROR "\[correctness::withlock_leak\]"
		}
	})
}

func readField(l *sync.Locked[Box]) int {
	n := 0
	l.WithLock(func(b *Box) {
		n = b.n
	})
	return n
}

func readCopy(l *sync.Locked[Box]) int {
	n := 0
	l.WithLock(func(b *Box) {
		v := *b
		n = v.n
	})
	return n
}

func callMethod(l *sync.Locked[Box]) int {
	n := 0
	l.WithLock(func(b *Box) {
		n = b.Touch()
	})
	return n
}

func callArg(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		useBox(b)
	})
}

func reassignAlias(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		c := b
		c = &Box{}
		leaked = c
	})
}

func useBeforeReassign(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		c := b
		leaked = c // ERROR "\[correctness::withlock_leak\]"
		c = &Box{}
	})
}

func reassignInOtherBlock(l *sync.Locked[Box]) {
	l.WithLock(func(b *Box) {
		c := b
		if b.n > 0 {
			c = &Box{}
		}
		leaked = c // ERROR "\[correctness::withlock_leak\]"
	})
}
