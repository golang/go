package testdata

import (
	"sync"
	"sync/atomic"
	"unsafe"
	. "unsafe"
	unsafe1 "unsafe"
)

func OkFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y

	var z = sync.Mutex{}
	w := sync.Mutex{}

	w = sync.Mutex{}
	q := struct{ L sync.Mutex }{
		L: sync.Mutex{},
	}

	yy := []Tlock{
		Tlock{},
		Tlock{
			once: sync.Once{},
		},
	}

	nl := new(sync.Mutex)
	mx := make([]sync.Mutex, 10)
	xx := struct{ L *sync.Mutex }{
		L: new(sync.Mutex),
	}
}

type Tlock struct {
	once sync.Once
}

func BadFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
	*p = *x // ERROR "assignment copies lock value to \*p: sync.Mutex"

	var t Tlock
	var tp *Tlock
	tp = &t
	*tp = t // ERROR "assignment copies lock value to \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	t = *tp // ERROR "assignment copies lock value to t: testdata.Tlock contains sync.Once contains sync.Mutex"

	y := *x   // ERROR "assignment copies lock value to y: sync.Mutex"
	var z = t // ERROR "variable declaration copies lock value to z: testdata.Tlock contains sync.Once contains sync.Mutex"

	w := struct{ L sync.Mutex }{
		L: *x, // ERROR "literal copies lock value from \*x: sync.Mutex"
	}
	var q = map[int]Tlock{
		1: t,   // ERROR "literal copies lock value from t: testdata.Tlock contains sync.Once contains sync.Mutex"
		2: *tp, // ERROR "literal copies lock value from \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	}
	yy := []Tlock{
		t,   // ERROR "literal copies lock value from t: testdata.Tlock contains sync.Once contains sync.Mutex"
		*tp, // ERROR "literal copies lock value from \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	}

	// override 'new' keyword
	new := func(interface{}) {}
	new(t) // ERROR "call of new copies lock value: testdata.Tlock contains sync.Once contains sync.Mutex"

	// copy of array of locks
	var muA [5]sync.Mutex
	muB := muA        // ERROR "assignment copies lock value to muB: sync.Mutex"
	muA = muB         // ERROR "assignment copies lock value to muA: sync.Mutex"
	muSlice := muA[:] // OK

	// multidimensional array
	var mmuA [5][5]sync.Mutex
	mmuB := mmuA        // ERROR "assignment copies lock value to mmuB: sync.Mutex"
	mmuA = mmuB         // ERROR "assignment copies lock value to mmuA: sync.Mutex"
	mmuSlice := mmuA[:] // OK

	// slice copy is ok
	var fmuA [5][][5]sync.Mutex
	fmuB := fmuA        // OK
	fmuA = fmuB         // OK
	fmuSlice := fmuA[:] // OK
}

func LenAndCapOnLockArrays() {
	var a [5]sync.Mutex
	aLen := len(a) // OK
	aCap := cap(a) // OK

	// override 'len' and 'cap' keywords

	len := func(interface{}) {}
	len(a) // ERROR "call of len copies lock value: sync.Mutex"

	cap := func(interface{}) {}
	cap(a) // ERROR "call of cap copies lock value: sync.Mutex"
}

func SizeofMutex() {
	var mu sync.Mutex
	unsafe.Sizeof(mu)  // OK
	unsafe1.Sizeof(mu) // OK
	Sizeof(mu)         // OK
	unsafe := struct{ Sizeof func(interface{}) }{}
	unsafe.Sizeof(mu) // ERROR "call of unsafe.Sizeof copies lock value: sync.Mutex"
	Sizeof := func(interface{}) {}
	Sizeof(mu) // ERROR "call of Sizeof copies lock value: sync.Mutex"
}

// SyncTypesCheck checks copying of sync.* types except sync.Mutex
func SyncTypesCheck() {
	// sync.RWMutex copying
	var rwmuX sync.RWMutex
	var rwmuXX = sync.RWMutex{}
	rwmuX1 := new(sync.RWMutex)
	rwmuY := rwmuX     // ERROR "assignment copies lock value to rwmuY: sync.RWMutex"
	rwmuY = rwmuX      // ERROR "assignment copies lock value to rwmuY: sync.RWMutex"
	var rwmuYY = rwmuX // ERROR "variable declaration copies lock value to rwmuYY: sync.RWMutex"
	rwmuP := &rwmuX
	rwmuZ := &sync.RWMutex{}

	// sync.Cond copying
	var condX sync.Cond
	var condXX = sync.Cond{}
	condX1 := new(sync.Cond)
	condY := condX     // ERROR "assignment copies lock value to condY: sync.Cond contains sync.noCopy"
	condY = condX      // ERROR "assignment copies lock value to condY: sync.Cond contains sync.noCopy"
	var condYY = condX // ERROR "variable declaration copies lock value to condYY: sync.Cond contains sync.noCopy"
	condP := &condX
	condZ := &sync.Cond{
		L: &sync.Mutex{},
	}
	condZ = sync.NewCond(&sync.Mutex{})

	// sync.WaitGroup copying
	var wgX sync.WaitGroup
	var wgXX = sync.WaitGroup{}
	wgX1 := new(sync.WaitGroup)
	wgY := wgX     // ERROR "assignment copies lock value to wgY: sync.WaitGroup contains sync.noCopy"
	wgY = wgX      // ERROR "assignment copies lock value to wgY: sync.WaitGroup contains sync.noCopy"
	var wgYY = wgX // ERROR "variable declaration copies lock value to wgYY: sync.WaitGroup contains sync.noCopy"
	wgP := &wgX
	wgZ := &sync.WaitGroup{}

	// sync.Pool copying
	var poolX sync.Pool
	var poolXX = sync.Pool{}
	poolX1 := new(sync.Pool)
	poolY := poolX     // ERROR "assignment copies lock value to poolY: sync.Pool contains sync.noCopy"
	poolY = poolX      // ERROR "assignment copies lock value to poolY: sync.Pool contains sync.noCopy"
	var poolYY = poolX // ERROR "variable declaration copies lock value to poolYY: sync.Pool contains sync.noCopy"
	poolP := &poolX
	poolZ := &sync.Pool{}

	// sync.Once copying
	var onceX sync.Once
	var onceXX = sync.Once{}
	onceX1 := new(sync.Once)
	onceY := onceX     // ERROR "assignment copies lock value to onceY: sync.Once contains sync.Mutex"
	onceY = onceX      // ERROR "assignment copies lock value to onceY: sync.Once contains sync.Mutex"
	var onceYY = onceX // ERROR "variable declaration copies lock value to onceYY: sync.Once contains sync.Mutex"
	onceP := &onceX
	onceZ := &sync.Once{}
}

// AtomicTypesCheck checks copying of sync/atomic types
func AtomicTypesCheck() {
	// atomic.Value copying
	var vX atomic.Value
	var vXX = atomic.Value{}
	vX1 := new(atomic.Value)
	// These are OK because the value has not been used yet.
	// (And vet can't tell whether it has been used, so they're always OK.)
	vY := vX
	vY = vX
	var vYY = vX
	vP := &vX
	vZ := &atomic.Value{}
}
