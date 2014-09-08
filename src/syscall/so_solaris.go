// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// soError describes reasons for shared library load failures.
type soError struct {
	Err     error
	ObjName string
	Msg     string
}

func (e *soError) Error() string { return e.Msg }

// Implemented in asm_solaris_amd64.s.
func rawSysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func sysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func dlclose(handle uintptr) (err Errno)
func dlopen(name *uint8, mode uintptr) (handle uintptr, err Errno)
func dlsym(handle uintptr, name *uint8) (proc uintptr, err Errno)

// A so implements access to a single shared library object.
type so struct {
	Name   string
	Handle uintptr
}

// loadSO loads shared library file into memory.
func loadSO(name string) (*so, error) {
	namep, err := BytePtrFromString(name)
	if err != nil {
		return nil, err
	}
	h, e := dlopen(namep, 1) // RTLD_LAZY
	use(unsafe.Pointer(namep))
	if e != 0 {
		return nil, &soError{
			Err:     e,
			ObjName: name,
			Msg:     "Failed to load " + name + ": " + e.Error(),
		}
	}
	d := &so{
		Name:   name,
		Handle: uintptr(h),
	}
	return d, nil
}

// mustLoadSO is like loadSO but panics if load operation fails.
func mustLoadSO(name string) *so {
	d, e := loadSO(name)
	if e != nil {
		panic(e)
	}
	return d
}

// FindProc searches shared library d for procedure named name and returns
// *proc if found. It returns an error if the search fails.
func (d *so) FindProc(name string) (*proc, error) {
	namep, err := BytePtrFromString(name)
	if err != nil {
		return nil, err
	}
	a, _ := dlsym(uintptr(d.Handle), namep)
	use(unsafe.Pointer(namep))
	if a == 0 {
		return nil, &soError{
			Err:     ENOSYS,
			ObjName: name,
			Msg:     "Failed to find " + name + " procedure in " + d.Name,
		}
	}
	p := &proc{
		SO:   d,
		Name: name,
		addr: a,
	}
	return p, nil
}

// MustFindProc is like FindProc but panics if search fails.
func (d *so) MustFindProc(name string) *proc {
	p, e := d.FindProc(name)
	if e != nil {
		panic(e)
	}
	return p
}

// Release unloads shared library d from memory.
func (d *so) Release() (err error) {
	return dlclose(d.Handle)
}

// A proc implements access to a procedure inside a shared library.
type proc struct {
	SO   *so
	Name string
	addr uintptr
}

// Addr returns the address of the procedure represented by p.
// The return value can be passed to Syscall to run the procedure.
func (p *proc) Addr() uintptr {
	return p.addr
}

// Call executes procedure p with arguments a. It will panic, if more then
// 6 arguments are supplied.
//
// The returned error is always non-nil, constructed from the result of
// GetLastError.  Callers must inspect the primary return value to decide
// whether an error occurred (according to the semantics of the specific
// function being called) before consulting the error. The error will be
// guaranteed to contain syscall.Errno.
func (p *proc) Call(a ...uintptr) (r1, r2 uintptr, lastErr error) {
	switch len(a) {
	case 0:
		return sysvicall6(p.Addr(), uintptr(len(a)), 0, 0, 0, 0, 0, 0)
	case 1:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], 0, 0, 0, 0, 0)
	case 2:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], a[1], 0, 0, 0, 0)
	case 3:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], 0, 0, 0)
	case 4:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], 0, 0)
	case 5:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], 0)
	case 6:
		return sysvicall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5])
	default:
		panic("Call " + p.Name + " with too many arguments " + itoa(len(a)) + ".")
	}
	return
}

// A lazySO implements access to a single shared library.  It will delay
// the load of the shared library until the first call to its Handle method
// or to one of its lazyProc's Addr method.
type lazySO struct {
	mu   sync.Mutex
	so   *so // non nil once SO is loaded
	Name string
}

// Load loads single shared file d.Name into memory. It returns an error if
// fails.  Load will not try to load SO, if it is already loaded into memory.
func (d *lazySO) Load() error {
	// Non-racy version of:
	// if d.so == nil {
	if atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&d.so))) == nil {
		d.mu.Lock()
		defer d.mu.Unlock()
		if d.so == nil {
			so, e := loadSO(d.Name)
			if e != nil {
				return e
			}
			// Non-racy version of:
			// d.so = so
			atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&d.so)), unsafe.Pointer(so))
		}
	}
	return nil
}

// mustLoad is like Load but panics if search fails.
func (d *lazySO) mustLoad() {
	e := d.Load()
	if e != nil {
		panic(e)
	}
}

// Handle returns d's module handle.
func (d *lazySO) Handle() uintptr {
	d.mustLoad()
	return uintptr(d.so.Handle)
}

// NewProc returns a lazyProc for accessing the named procedure in the SO d.
func (d *lazySO) NewProc(name string) *lazyProc {
	return &lazyProc{l: d, Name: name}
}

// newLazySO creates new lazySO associated with SO file.
func newLazySO(name string) *lazySO {
	return &lazySO{Name: name}
}

// A lazyProc implements access to a procedure inside a lazySO.
// It delays the lookup until the Addr method is called.
type lazyProc struct {
	mu   sync.Mutex
	Name string
	l    *lazySO
	proc *proc
}

// Find searches the shared library for procedure named p.Name. It returns an
// error if search fails. Find will not search procedure, if it is already
// found and loaded into memory.
func (p *lazyProc) Find() error {
	// Non-racy version of:
	// if p.proc == nil {
	if atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&p.proc))) == nil {
		p.mu.Lock()
		defer p.mu.Unlock()
		if p.proc == nil {
			e := p.l.Load()
			if e != nil {
				return e
			}
			proc, e := p.l.so.FindProc(p.Name)
			if e != nil {
				return e
			}
			// Non-racy version of:
			// p.proc = proc
			atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&p.proc)), unsafe.Pointer(proc))
		}
	}
	return nil
}

// mustFind is like Find but panics if search fails.
func (p *lazyProc) mustFind() {
	e := p.Find()
	if e != nil {
		panic(e)
	}
}

// Addr returns the address of the procedure represented by p.
// The return value can be passed to Syscall to run the procedure.
func (p *lazyProc) Addr() uintptr {
	p.mustFind()
	return p.proc.Addr()
}

// Call executes procedure p with arguments a. It will panic, if more then
// 6 arguments are supplied.
//
// The returned error is always non-nil, constructed from the result of
// GetLastError.  Callers must inspect the primary return value to decide
// whether an error occurred (according to the semantics of the specific
// function being called) before consulting the error. The error will be
// guaranteed to contain syscall.Errno.
func (p *lazyProc) Call(a ...uintptr) (r1, r2 uintptr, lastErr error) {
	p.mustFind()
	return p.proc.Call(a...)
}
