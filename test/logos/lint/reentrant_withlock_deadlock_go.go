// errorcheck

package p

import "logos/sync"

// Go twin of reentrant_withlock_deadlock.lgo: concurrency::reentrant_withlock_deadlock
// (SPEC Sec 1.2, #167) on a Go package, where it runs at deny and the
// callback is a function literal. Depth 0 and depth 1 fail; depth 2 and a
// different receiver pass.

var global sync.Locked[int]

var other sync.Locked[int]

type Counter struct {
	mu sync.Locked[int]
	n  int
}

type Wrapper struct {
	inner Counter
	mu    sync.Locked[int]
}

type Fake struct{}

func (f *Fake) WithLock(fn func(*int)) {
	var n int
	fn(&n)
}

func lockGlobal() {
	global.WithLock(func(n *int) { *n++ })
}

func lockGlobalTwice() {
	lockGlobal()
	lockGlobal()
}

func lockParam(l *sync.Locked[int]) {
	l.WithLock(func(n *int) { *n++ })
}

func lockByValue(l sync.Locked[int]) {
	l.WithLock(func(n *int) { *n++ })
}

func lockParamTwoDeep(l *sync.Locked[int]) {
	lockParam(l)
}

func lockOther() {
	other.WithLock(func(n *int) { *n++ })
}

func (c *Counter) inc() {
	c.mu.WithLock(func(n *int) { *n++ })
}

func (c *Counter) incTwoDeep() {
	c.inc()
}

func (c *Counter) read() int {
	return c.n
}

func (c *Counter) outerDeadlocks() {
	c.mu.WithLock(func(n *int) {
		c.inc() // ERROR "reentrant_withlock_deadlock"
	})
}

func (c *Counter) outerTwoDeep() {
	c.mu.WithLock(func(n *int) {
		c.incTwoDeep()
	})
}

func (c *Counter) outerOtherMethod() {
	c.mu.WithLock(func(n *int) {
		*n += c.read()
	})
}

func directDepthZero() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		l.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			*m++
		})
	})
}

func directParenthesizedLiteral() {
	var l sync.Locked[int]
	l.WithLock((func(n *int) {
		l.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			*m++
		})
	}))
}

func directOnGlobal() {
	global.WithLock(func(n *int) {
		global.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			*m++
		})
	})
}

func directOnField() {
	var c Counter
	c.mu.WithLock(func(n *int) {
		c.mu.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			*m++
		})
	})
}

func directThroughPointer() {
	var l sync.Locked[int]
	p := &l
	p.WithLock(func(n *int) {
		p.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			*m++
		})
	})
}

func directNamedCallback() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		l.WithLock(setOne) // ERROR "reentrant_withlock_deadlock"
	})
}

func directDeeplyNested() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		if *n > 0 {
			for range 2 {
				l.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
					*m++
				})
			}
		}
	})
}

func directThroughOtherLock() {
	var a, b sync.Locked[int]
	a.WithLock(func(n *int) {
		b.WithLock(func(m *int) {
			a.WithLock(func(k *int) { // ERROR "reentrant_withlock_deadlock"
				*k++
			})
		})
	})
}

func directThreeLevels() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		l.WithLock(func(m *int) { // ERROR "reentrant_withlock_deadlock"
			l.WithLock(func(k *int) { // ERROR "reentrant_withlock_deadlock"
				*k++
			})
		})
	})
}

func depthOneGlobal() {
	global.WithLock(func(n *int) {
		lockGlobal() // ERROR "reentrant_withlock_deadlock"
	})
}

func depthOneParameter() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		lockParam(&l) // ERROR "reentrant_withlock_deadlock"
	})
}

func depthOneMethodOnField() {
	var c Counter
	c.mu.WithLock(func(n *int) {
		c.inc() // ERROR "reentrant_withlock_deadlock"
	})
}

func depthTwoParameter() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		lockParamTwoDeep(&l)
	})
}

func depthTwoGlobal() {
	global.WithLock(func(n *int) {
		lockGlobalTwice()
	})
}

func passDifferentReceiver() {
	var a, b sync.Locked[int]
	a.WithLock(func(n *int) {
		b.WithLock(func(m *int) {
			*m += *n
		})
	})
}

func passDifferentGlobals() {
	global.WithLock(func(n *int) {
		lockOther()
	})
}

func passDifferentParameter() {
	var a, b sync.Locked[int]
	a.WithLock(func(n *int) {
		lockParam(&b)
	})
}

func passByValueCopy() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		lockByValue(l)
	})
}

func passDifferentCounters() {
	var a, b Counter
	a.mu.WithLock(func(n *int) {
		b.inc()
	})
}

func passDifferentField() {
	var w Wrapper
	w.mu.WithLock(func(n *int) {
		w.inner.inc()
	})
}

func passSequential() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) { *n++ })
	l.WithLock(func(n *int) { *n++ })
	lockParam(&l)
}

func passAfterLambda() {
	global.WithLock(func(n *int) { *n++ })
	lockGlobal()
}

func passShadowedReceiver() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		var l sync.Locked[int]
		l.WithLock(func(m *int) { *m++ })
	})
}

func passFakeWithLock() {
	var f Fake
	f.WithLock(func(n *int) {
		f.WithLock(func(m *int) { *m++ })
	})
}

func passDeferredNestedLiteral() {
	var l sync.Locked[int]
	l.WithLock(func(n *int) {
		inner := func() {
			l.WithLock(func(m *int) { *m++ })
		}
		_ = inner
	})
}

func passNoLiteral() {
	var l sync.Locked[int]
	l.WithLock(setOne)
}

func passUnmatchedReceiver() {
	var ls [2]sync.Locked[int]
	ls[0].WithLock(func(n *int) {
		ls[1].WithLock(func(m *int) { *m++ })
	})
}

func setOne(n *int) {
	*n = 1
}
