// errorcheck -0 -d=ssa/opt/debug=3

package main

import (
	"crypto/sha1"
	"errors"
	"fmt"
	"sync"
)

func f0() {
	v := errors.New("error string")
	_ = v.Error() // ERROR "de-virtualizing call$"
}

func f1() {
	h := sha1.New()
	buf := make([]byte, 4)
	h.Write(buf)   // ERROR "de-virtualizing call$"
	_ = h.Sum(nil) // ERROR "de-virtualizing call$"
}

func f2() {
	// trickier case: make sure we see this is *sync.rlocker
	// instead of *sync.RWMutex,
	// even though they are the same pointers
	var m sync.RWMutex
	r := m.RLocker()

	// deadlock if the type of 'r' is improperly interpreted
	// as *sync.RWMutex
	r.Lock() // ERROR "de-virtualizing call$"
	m.RLock()
	r.Unlock() // ERROR "de-virtualizing call$"
	m.RUnlock()
}

type multiword struct{ a, b, c int }

func (m multiword) Error() string { return fmt.Sprintf("%d, %d, %d", m.a, m.b, m.c) }

func f3() {
	// can't de-virtualize this one yet;
	// it passes through a call to iconvT2I
	var err error
	err = multiword{1, 2, 3}
	if err.Error() != "1, 2, 3" {
		panic("bad call")
	}

	// ... but we can do this one
	err = &multiword{1, 2, 3}
	if err.Error() != "1, 2, 3" { // ERROR "de-virtualizing call$"
		panic("bad call")
	}
}

func main() {
	f0()
	f1()
	f2()
	f3()
}
