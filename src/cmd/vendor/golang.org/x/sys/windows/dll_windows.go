// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// We need to use LoadLibrary and GetProcAddress from the Go runtime, because
// the these symbols are loaded by the system linker and are required to
// dynamically load additional symbols. Note that in the Go runtime, these
// return syscall.Handle and syscall.Errno, but these are the same, in fact,
// as windows.Handle and windows.Errno, and we intend to keep these the same.

//go:linkname syscall_loadlibrary syscall.loadlibrary
func syscall_loadlibrary(filename *uint16) (handle Handle, err Errno)

//go:linkname syscall_getprocaddress syscall.getprocaddress
func syscall_getprocaddress(handle Handle, procname *uint8) (proc uintptr, err Errno)

// DLLError describes reasons for DLL load failures.
type DLLError struct {
	Err     error
	ObjName string
	Msg     string
}

func (e *DLLError) Error() string { return e.Msg }

// A DLL implements access to a single DLL.
type DLL struct {
	Name   string
	Handle Handle
}

// LoadDLL loads DLL file into memory.
//
// Warning: using LoadDLL without an absolute path name is subject to
// DLL preloading attacks. To safely load a system DLL, use LazyDLL
// with System set to true, or use LoadLibraryEx directly.
func LoadDLL(name string) (dll *DLL, err error) {
	namep, err := UTF16PtrFromString(name)
	if err != nil {
		return nil, err
	}
	h, e := syscall_loadlibrary(namep)
	if e != 0 {
		return nil, &DLLError{
			Err:     e,
			ObjName: name,
			Msg:     "Failed to load " + name + ": " + e.Error(),
		}
	}
	d := &DLL{
		Name:   name,
		Handle: h,
	}
	return d, nil
}

// MustLoadDLL is like LoadDLL but panics if load operation failes.
func MustLoadDLL(name string) *DLL {
	d, e := LoadDLL(name)
	if e != nil {
		panic(e)
	}
	return d
}

// FindProc searches DLL d for procedure named name and returns *Proc
// if found. It returns an error if search fails.
func (d *DLL) FindProc(name string) (proc *Proc, err error) {
	namep, err := BytePtrFromString(name)
	if err != nil {
		return nil, err
	}
	a, e := syscall_getprocaddress(d.Handle, namep)
	if e != 0 {
		return nil, &DLLError{
			Err:     e,
			ObjName: name,
			Msg:     "Failed to find " + name + " procedure in " + d.Name + ": " + e.Error(),
		}
	}
	p := &Proc{
		Dll:  d,
		Name: name,
		addr: a,
	}
	return p, nil
}

// MustFindProc is like FindProc but panics if search fails.
func (d *DLL) MustFindProc(name string) *Proc {
	p, e := d.FindProc(name)
	if e != nil {
		panic(e)
	}
	return p
}

// Release unloads DLL d from memory.
func (d *DLL) Release() (err error) {
	return FreeLibrary(d.Handle)
}

// A Proc implements access to a procedure inside a DLL.
type Proc struct {
	Dll  *DLL
	Name string
	addr uintptr
}

// Addr returns the address of the procedure represented by p.
// The return value can be passed to Syscall to run the procedure.
func (p *Proc) Addr() uintptr {
	return p.addr
}

//go:uintptrescapes

// Call executes procedure p with arguments a. It will panic, if more than 15 arguments
// are supplied.
//
// The returned error is always non-nil, constructed from the result of GetLastError.
// Callers must inspect the primary return value to decide whether an error occurred
// (according to the semantics of the specific function being called) before consulting
// the error. The error will be guaranteed to contain windows.Errno.
func (p *Proc) Call(a ...uintptr) (r1, r2 uintptr, lastErr error) {
	switch len(a) {
	case 0:
		return syscall.Syscall(p.Addr(), uintptr(len(a)), 0, 0, 0)
	case 1:
		return syscall.Syscall(p.Addr(), uintptr(len(a)), a[0], 0, 0)
	case 2:
		return syscall.Syscall(p.Addr(), uintptr(len(a)), a[0], a[1], 0)
	case 3:
		return syscall.Syscall(p.Addr(), uintptr(len(a)), a[0], a[1], a[2])
	case 4:
		return syscall.Syscall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], 0, 0)
	case 5:
		return syscall.Syscall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], 0)
	case 6:
		return syscall.Syscall6(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5])
	case 7:
		return syscall.Syscall9(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], 0, 0)
	case 8:
		return syscall.Syscall9(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], 0)
	case 9:
		return syscall.Syscall9(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
	case 10:
		return syscall.Syscall12(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], 0, 0)
	case 11:
		return syscall.Syscall12(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], 0)
	case 12:
		return syscall.Syscall12(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11])
	case 13:
		return syscall.Syscall15(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], 0, 0)
	case 14:
		return syscall.Syscall15(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], 0)
	case 15:
		return syscall.Syscall15(p.Addr(), uintptr(len(a)), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14])
	default:
		panic("Call " + p.Name + " with too many arguments " + itoa(len(a)) + ".")
	}
}

// A LazyDLL implements access to a single DLL.
// It will delay the load of the DLL until the first
// call to its Handle method or to one of its
// LazyProc's Addr method.
type LazyDLL struct {
	Name string

	// System determines whether the DLL must be loaded from the
	// Windows System directory, bypassing the normal DLL search
	// path.
	System bool

	mu  sync.Mutex
	dll *DLL // non nil once DLL is loaded
}

// Load loads DLL file d.Name into memory. It returns an error if fails.
// Load will not try to load DLL, if it is already loaded into memory.
func (d *LazyDLL) Load() error {
	// Non-racy version of:
	// if d.dll != nil {
	if atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(&d.dll))) != nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.dll != nil {
		return nil
	}

	// kernel32.dll is special, since it's where LoadLibraryEx comes from.
	// The kernel already special-cases its name, so it's always
	// loaded from system32.
	var dll *DLL
	var err error
	if d.Name == "kernel32.dll" {
		dll, err = LoadDLL(d.Name)
	} else {
		dll, err = loadLibraryEx(d.Name, d.System)
	}
	if err != nil {
		return err
	}

	// Non-racy version of:
	// d.dll = dll
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&d.dll)), unsafe.Pointer(dll))
	return nil
}

// mustLoad is like Load but panics if search fails.
func (d *LazyDLL) mustLoad() {
	e := d.Load()
	if e != nil {
		panic(e)
	}
}

// Handle returns d's module handle.
func (d *LazyDLL) Handle() uintptr {
	d.mustLoad()
	return uintptr(d.dll.Handle)
}

// NewProc returns a LazyProc for accessing the named procedure in the DLL d.
func (d *LazyDLL) NewProc(name string) *LazyProc {
	return &LazyProc{l: d, Name: name}
}

// NewLazyDLL creates new LazyDLL associated with DLL file.
func NewLazyDLL(name string) *LazyDLL {
	return &LazyDLL{Name: name}
}

// NewLazySystemDLL is like NewLazyDLL, but will only
// search Windows System directory for the DLL if name is
// a base name (like "advapi32.dll").
func NewLazySystemDLL(name string) *LazyDLL {
	return &LazyDLL{Name: name, System: true}
}

// A LazyProc implements access to a procedure inside a LazyDLL.
// It delays the lookup until the Addr method is called.
type LazyProc struct {
	Name string

	mu   sync.Mutex
	l    *LazyDLL
	proc *Proc
}

// Find searches DLL for procedure named p.Name. It returns
// an error if search fails. Find will not search procedure,
// if it is already found and loaded into memory.
func (p *LazyProc) Find() error {
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
			proc, e := p.l.dll.FindProc(p.Name)
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
func (p *LazyProc) mustFind() {
	e := p.Find()
	if e != nil {
		panic(e)
	}
}

// Addr returns the address of the procedure represented by p.
// The return value can be passed to Syscall to run the procedure.
// It will panic if the procedure cannot be found.
func (p *LazyProc) Addr() uintptr {
	p.mustFind()
	return p.proc.Addr()
}

//go:uintptrescapes

// Call executes procedure p with arguments a. It will panic, if more than 15 arguments
// are supplied. It will also panic if the procedure cannot be found.
//
// The returned error is always non-nil, constructed from the result of GetLastError.
// Callers must inspect the primary return value to decide whether an error occurred
// (according to the semantics of the specific function being called) before consulting
// the error. The error will be guaranteed to contain windows.Errno.
func (p *LazyProc) Call(a ...uintptr) (r1, r2 uintptr, lastErr error) {
	p.mustFind()
	return p.proc.Call(a...)
}

var canDoSearchSystem32Once struct {
	sync.Once
	v bool
}

func initCanDoSearchSystem32() {
	// https://msdn.microsoft.com/en-us/library/ms684179(v=vs.85).aspx says:
	// "Windows 7, Windows Server 2008 R2, Windows Vista, and Windows
	// Server 2008: The LOAD_LIBRARY_SEARCH_* flags are available on
	// systems that have KB2533623 installed. To determine whether the
	// flags are available, use GetProcAddress to get the address of the
	// AddDllDirectory, RemoveDllDirectory, or SetDefaultDllDirectories
	// function. If GetProcAddress succeeds, the LOAD_LIBRARY_SEARCH_*
	// flags can be used with LoadLibraryEx."
	canDoSearchSystem32Once.v = (modkernel32.NewProc("AddDllDirectory").Find() == nil)
}

func canDoSearchSystem32() bool {
	canDoSearchSystem32Once.Do(initCanDoSearchSystem32)
	return canDoSearchSystem32Once.v
}

func isBaseName(name string) bool {
	for _, c := range name {
		if c == ':' || c == '/' || c == '\\' {
			return false
		}
	}
	return true
}

// loadLibraryEx wraps the Windows LoadLibraryEx function.
//
// See https://msdn.microsoft.com/en-us/library/windows/desktop/ms684179(v=vs.85).aspx
//
// If name is not an absolute path, LoadLibraryEx searches for the DLL
// in a variety of automatic locations unless constrained by flags.
// See: https://msdn.microsoft.com/en-us/library/ff919712%28VS.85%29.aspx
func loadLibraryEx(name string, system bool) (*DLL, error) {
	loadDLL := name
	var flags uintptr
	if system {
		if canDoSearchSystem32() {
			const LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
			flags = LOAD_LIBRARY_SEARCH_SYSTEM32
		} else if isBaseName(name) {
			// WindowsXP or unpatched Windows machine
			// trying to load "foo.dll" out of the system
			// folder, but LoadLibraryEx doesn't support
			// that yet on their system, so emulate it.
			systemdir, err := GetSystemDirectory()
			if err != nil {
				return nil, err
			}
			loadDLL = systemdir + "\\" + name
		}
	}
	h, err := LoadLibraryEx(loadDLL, 0, flags)
	if err != nil {
		return nil, err
	}
	return &DLL{Name: name, Handle: h}, nil
}

type errString string

func (s errString) Error() string { return string(s) }
