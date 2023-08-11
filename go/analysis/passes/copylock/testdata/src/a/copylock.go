package a

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

	var pz = (sync.Mutex{})
	pw := (sync.Mutex{})
}

type Tlock struct {
	once sync.Once
}

func BadFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
	*p = *x // want `assignment copies lock value to \*p: sync.Mutex`

	var t Tlock
	var tp *Tlock
	tp = &t
	*tp = t // want `assignment copies lock value to \*tp: a.Tlock contains sync.Once contains sync\b.*`
	t = *tp // want `assignment copies lock value to t: a.Tlock contains sync.Once contains sync\b.*`

	y := *x   // want "assignment copies lock value to y: sync.Mutex"
	var z = t // want `variable declaration copies lock value to z: a.Tlock contains sync.Once contains sync\b.*`

	w := struct{ L sync.Mutex }{
		L: *x, // want `literal copies lock value from \*x: sync.Mutex`
	}
	var q = map[int]Tlock{
		1: t,   // want `literal copies lock value from t: a.Tlock contains sync.Once contains sync\b.*`
		2: *tp, // want `literal copies lock value from \*tp: a.Tlock contains sync.Once contains sync\b.*`
	}
	yy := []Tlock{
		t,   // want `literal copies lock value from t: a.Tlock contains sync.Once contains sync\b.*`
		*tp, // want `literal copies lock value from \*tp: a.Tlock contains sync.Once contains sync\b.*`
	}

	// override 'new' keyword
	new := func(interface{}) {}
	new(t) // want `call of new copies lock value: a.Tlock contains sync.Once contains sync\b.*`

	// copy of array of locks
	var muA [5]sync.Mutex
	muB := muA        // want "assignment copies lock value to muB: sync.Mutex"
	muA = muB         // want "assignment copies lock value to muA: sync.Mutex"
	muSlice := muA[:] // OK

	// multidimensional array
	var mmuA [5][5]sync.Mutex
	mmuB := mmuA        // want "assignment copies lock value to mmuB: sync.Mutex"
	mmuA = mmuB         // want "assignment copies lock value to mmuA: sync.Mutex"
	mmuSlice := mmuA[:] // OK

	// slice copy is ok
	var fmuA [5][][5]sync.Mutex
	fmuB := fmuA        // OK
	fmuA = fmuB         // OK
	fmuSlice := fmuA[:] // OK

	// map access by single and tuple copies prohibited
	type mut struct{ mu sync.Mutex }
	muM := map[string]mut{
		"a": mut{},
	}
	mumA := muM["a"]    // want "assignment copies lock value to mumA: a.mut contains sync.Mutex"
	mumB, _ := muM["a"] // want "assignment copies lock value to mumB: \\(a.mut, bool\\) contains a.mut contains sync.Mutex"
}

func LenAndCapOnLockArrays() {
	var a [5]sync.Mutex
	aLen := len(a) // OK
	aCap := cap(a) // OK

	// override 'len' and 'cap' keywords

	len := func(interface{}) {}
	len(a) // want "call of len copies lock value: sync.Mutex"

	cap := func(interface{}) {}
	cap(a) // want "call of cap copies lock value: sync.Mutex"
}

func SizeofMutex() {
	var mu sync.Mutex
	unsafe.Sizeof(mu)  // OK
	unsafe1.Sizeof(mu) // OK
	Sizeof(mu)         // OK
	unsafe := struct{ Sizeof func(interface{}) }{}
	unsafe.Sizeof(mu) // want "call of unsafe.Sizeof copies lock value: sync.Mutex"
	Sizeof := func(interface{}) {}
	Sizeof(mu) // want "call of Sizeof copies lock value: sync.Mutex"
}

func OffsetofMutex() {
	type T struct {
		f  int
		mu sync.Mutex
	}
	unsafe.Offsetof(T{}.mu) // OK
	unsafe := struct{ Offsetof func(interface{}) }{}
	unsafe.Offsetof(T{}.mu) // want "call of unsafe.Offsetof copies lock value: sync.Mutex"
}

func AlignofMutex() {
	type T struct {
		f  int
		mu sync.Mutex
	}
	unsafe.Alignof(T{}.mu) // OK
	unsafe := struct{ Alignof func(interface{}) }{}
	unsafe.Alignof(T{}.mu) // want "call of unsafe.Alignof copies lock value: sync.Mutex"
}

// SyncTypesCheck checks copying of sync.* types except sync.Mutex
func SyncTypesCheck() {
	// sync.RWMutex copying
	var rwmuX sync.RWMutex
	var rwmuXX = sync.RWMutex{}
	rwmuX1 := new(sync.RWMutex)
	rwmuY := rwmuX     // want "assignment copies lock value to rwmuY: sync.RWMutex"
	rwmuY = rwmuX      // want "assignment copies lock value to rwmuY: sync.RWMutex"
	var rwmuYY = rwmuX // want "variable declaration copies lock value to rwmuYY: sync.RWMutex"
	rwmuP := &rwmuX
	rwmuZ := &sync.RWMutex{}

	// sync.Cond copying
	var condX sync.Cond
	var condXX = sync.Cond{}
	condX1 := new(sync.Cond)
	condY := condX     // want "assignment copies lock value to condY: sync.Cond contains sync.noCopy"
	condY = condX      // want "assignment copies lock value to condY: sync.Cond contains sync.noCopy"
	var condYY = condX // want "variable declaration copies lock value to condYY: sync.Cond contains sync.noCopy"
	condP := &condX
	condZ := &sync.Cond{
		L: &sync.Mutex{},
	}
	condZ = sync.NewCond(&sync.Mutex{})

	// sync.WaitGroup copying
	var wgX sync.WaitGroup
	var wgXX = sync.WaitGroup{}
	wgX1 := new(sync.WaitGroup)
	wgY := wgX     // want "assignment copies lock value to wgY: sync.WaitGroup contains sync.noCopy"
	wgY = wgX      // want "assignment copies lock value to wgY: sync.WaitGroup contains sync.noCopy"
	var wgYY = wgX // want "variable declaration copies lock value to wgYY: sync.WaitGroup contains sync.noCopy"
	wgP := &wgX
	wgZ := &sync.WaitGroup{}

	// sync.Pool copying
	var poolX sync.Pool
	var poolXX = sync.Pool{}
	poolX1 := new(sync.Pool)
	poolY := poolX     // want "assignment copies lock value to poolY: sync.Pool contains sync.noCopy"
	poolY = poolX      // want "assignment copies lock value to poolY: sync.Pool contains sync.noCopy"
	var poolYY = poolX // want "variable declaration copies lock value to poolYY: sync.Pool contains sync.noCopy"
	poolP := &poolX
	poolZ := &sync.Pool{}

	// sync.Once copying
	var onceX sync.Once
	var onceXX = sync.Once{}
	onceX1 := new(sync.Once)
	onceY := onceX     // want `assignment copies lock value to onceY: sync.Once contains sync\b.*`
	onceY = onceX      // want `assignment copies lock value to onceY: sync.Once contains sync\b.*`
	var onceYY = onceX // want `variable declaration copies lock value to onceYY: sync.Once contains sync\b.*`
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

// PointerRhsCheck checks that exceptions are made for pointer return values of
// function calls. These may be zero initialized so they are considered OK.
func PointerRhsCheck() {
	newMutex := func() *sync.Mutex { return new(sync.Mutex) }
	d := *newMutex()
	pd := *(newMutex())
}
