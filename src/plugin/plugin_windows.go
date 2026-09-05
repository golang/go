// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package plugin

import (
	"errors"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

// Windows plugin loading mirrors the dlopen-based path used on
// Linux/Darwin/FreeBSD, but uses a host-independent design: the DLL
// is built with -buildmode=plugin (no link-time host binding) and
// carries its own copy of the runtime/stdlib code. Host-shared
// symbols the compiler emitted as external -dynlink references
// (R_PEIMPORT on windows/amd64) are rewritten by the linker into
// indirect calls through a slot table embedded in the DLL, described
// by a meta blob (see cmd/link/internal/ld/pe_plugin.go).
//
// At load time we LoadDLL the plugin, walk the meta blob, and fill
// each slot with GetProcAddress(<host module>, <name>) so subsequent
// calls into shared symbols reach the host's implementation. The
// plugin's moduledata is then attached to the host's
// firstmoduledata chain via runtime.pluginAddmoduledata, and
// runtime.plugin_lastmoduleinit completes type unification, itab
// merge, and surfaces the plugin's symbol table.

func open(name string) (*Plugin, error) {
	abs, err := filepath.Abs(name)
	if err != nil {
		return nil, errors.New(`plugin.Open("` + name + `"): ` + err.Error())
	}

	pluginsMu.Lock()
	if p := plugins[abs]; p != nil {
		pluginsMu.Unlock()
		if p.err != "" {
			return nil, errors.New(`plugin.Open("` + name + `"): ` + p.err + ` (previous failure)`)
		}
		<-p.loaded
		return p, nil
	}

	dll, err := syscall.LoadDLL(abs)
	if err != nil {
		pluginsMu.Unlock()
		return nil, errors.New(`plugin.Open("` + name + `"): ` + err.Error())
	}

	// Resolve the plugin's host-shared symbol slots against the
	// running host EXE module. This is the windows analogue of
	// dlopen(RTLD_GLOBAL): every symbol the plugin's compiler emitted
	// as an external -dynlink reference is looked up on the host with
	// GetProcAddress; the slot table embedded in the plugin DLL is
	// then filled in with the host's addresses so subsequent calls
	// (including direct calls from the plugin's own runtime copies of
	// shared functions, which were rewritten to JMP-via-slot thunks
	// at link time) reach the host's implementation.
	if err := resolvePluginSlots(dll); err != nil {
		pluginsMu.Unlock()
		return nil, errors.New(`plugin.Open("` + name + `"): ` + err.Error())
	}

	// LoadDLL on windows does not register the plugin's moduledata in
	// the host runtime's firstmoduledata chain (there is no
	// dlopen(RTLD_GLOBAL) equivalent). Instead the plugin DLL exports
	// its local.pluginmoduledata symbol; we look it up and pass it to
	// runtime.pluginAddmoduledata host-side. lastmoduleinit() below
	// then sees a fresh moduledata at the tail of the chain.
	mdProc, err := dll.FindProc("go_pluginmoduledata")
	if err != nil {
		pluginsMu.Unlock()
		return nil, errors.New(`plugin.Open("` + name + `"): plugin DLL does not export go_pluginmoduledata: ` + err.Error())
	}
	pluginAddmoduledata(mdProc.Addr())

	displayName := name
	if ext := filepath.Ext(displayName); ext == ".dll" || ext == ".so" {
		displayName = displayName[:len(displayName)-len(ext)]
	}
	if plugins == nil {
		plugins = make(map[string]*Plugin)
	}

	pluginpath, syms, initTasks, errstr := lastmoduleinit()
	if errstr != "" {
		plugins[abs] = &Plugin{
			pluginpath: pluginpath,
			err:        errstr,
		}
		pluginsMu.Unlock()
		return nil, errors.New(`plugin.Open("` + name + `"): ` + errstr)
	}

	p := &Plugin{
		pluginpath: pluginpath,
		loaded:     make(chan struct{}),
	}
	plugins[abs] = p
	pluginsMu.Unlock()

	// Skip init tasks for packages whose `.inittask` symbol is also
	// exported from the host EXE: those packages are host-shared (their
	// init has already run during host startup), and re-running them in
	// the plugin would corrupt host runtime state (e.g. a duplicate
	// forcegchelper goroutine from runtime.init.7). For the remaining
	// plugin-only packages (the user's own code, plus any imports the
	// host did not pull in), doInit must still run.
	if hostMod, herr := getModuleHandleNull(); herr == nil {
		for _, t := range initTasks {
			if t == nil {
				continue
			}
			nfns := *(*uint32)(unsafe.Pointer(uintptr(unsafe.Pointer(t)) + 4))
			if nfns == 0 {
				continue
			}
			firstPC := *(*uintptr)(unsafe.Pointer(uintptr(unsafe.Pointer(t)) + 8))
			fn := runtime.FuncForPC(firstPC)
			if fn == nil {
				continue
			}
			name := fn.Name() // e.g. "fmt.init.0" or "runtime.init.7"
			i := strings.LastIndex(name, ".init")
			if i < 0 {
				continue
			}
			pkg := name[:i]
			if addr, err := syscall.GetProcAddress(hostMod, pkg+"..inittask"); err == nil && addr != 0 {
				*(*uint32)(unsafe.Pointer(t)) = 2 // mark as already done
			}
		}
	}
	doInit(initTasks)

	updatedSyms := map[string]any{}
	for symName, sym := range syms {
		isFunc := symName[0] == '.'
		if isFunc {
			delete(syms, symName)
			symName = symName[1:]
		}

		fullName := pluginpath + "." + symName
		proc, err := dll.FindProc(fullName)
		if err != nil {
			return nil, errors.New(`plugin.Open("` + displayName + `"): could not find symbol ` + symName + `: ` + err.Error())
		}
		valp := (*[2]unsafe.Pointer)(unsafe.Pointer(&sym))
		if isFunc {
			addr := proc.Addr()
			(*valp)[1] = unsafe.Pointer(&addr)
		} else {
			(*valp)[1] = unsafe.Pointer(proc.Addr())
		}
		updatedSyms[symName] = sym
	}
	p.syms = updatedSyms

	close(p.loaded)
	return p, nil
}

func lookup(p *Plugin, symName string) (Symbol, error) {
	if s := p.syms[symName]; s != nil {
		return s, nil
	}
	return nil, errors.New("plugin: symbol " + symName + " not found in plugin " + p.pluginpath)
}

var (
	pluginsMu sync.Mutex
	plugins   map[string]*Plugin
)

// lastmoduleinit is defined in package runtime.
func lastmoduleinit() (pluginpath string, syms map[string]any, inittasks []*initTask, errstr string)

// pluginAddmoduledata is defined in package runtime. It registers the
// plugin's local.pluginmoduledata with the host runtime's
// firstmoduledata chain.
//
//go:linkname pluginAddmoduledata runtime.pluginAddmoduledata
func pluginAddmoduledata(md uintptr)

// doInit is defined in package runtime.
//
//go:linkname doInit runtime.doInit
func doInit(t []*initTask)

type initTask struct {
	// fields defined in runtime.initTask. We only handle pointers to an initTask
	// in this package, so the contents are irrelevant.
}

// resolvePluginSlots is the windows analogue of dlopen(RTLD_GLOBAL):
// the plugin's link-time slot table (go_plugin_resolveslots, layout
// [N]uintptr) holds one entry per host-shared symbol the plugin's
// compiler emitted as an external -dynlink reference. For each slot
// described in the go_plugin_resolvemeta blob, we look the name up in
// the running host EXE's export table via GetProcAddress and write the
// resulting address into the slot. Subsequent calls â€” including direct
// calls from the plugin's own runtime copies of shared functions,
// whose bodies were rewritten to JMP-via-slot thunks at link time â€”
// then reach the host's real implementation.
//
// Meta layout (matches the linker in pe_plugin.go):
//
//	header [16]byte { magic "GORESV01" (8) | count uint32 | pad uint32 }
//	per entry { flags uint32 | nameLen uint32 | name [nameLen]byte | pad to 4 }
//
// flags bit 0 (TEXT) means the slot has no usable local fallback (the
// text body is a thunk); missing host export => load fails. flags=0
// means a data slot whose initial value is the plugin-local fallback;
// missing host export => slot stays at fallback.
func resolvePluginSlots(dll *syscall.DLL) error {
	slotsProc, err := dll.FindProc("go_plugin_resolveslots")
	if err != nil {
		// Plugin has no shared-symbol slots at all (e.g. trivial
		// main-only plugin with -dynlink off). Nothing to do.
		return nil
	}
	metaProc, err := dll.FindProc("go_plugin_resolvemeta")
	if err != nil {
		return errors.New("plugin DLL exports go_plugin_resolveslots but no go_plugin_resolvemeta")
	}

	metaBase := metaProc.Addr()
	magic := (*[8]byte)(unsafe.Pointer(metaBase))
	if string(magic[:]) != "GORESV01" {
		return errors.New("plugin resolve meta: bad magic")
	}
	count := *(*uint32)(unsafe.Pointer(metaBase + 8))
	if count == 0 {
		return nil
	}

	hostMod, err := getModuleHandleNull()
	if err != nil {
		return err
	}

	slotsBase := slotsProc.Addr()
	slotsSize := uintptr(count) * 8
	var oldProtect uint32
	if err := virtualProtect(slotsBase, slotsSize, 0x04 /* PAGE_READWRITE */, &oldProtect); err != nil {
		return err
	}
	defer virtualProtect(slotsBase, slotsSize, oldProtect, &oldProtect)

	off := uintptr(16) // header padded to 16 bytes
	for i := uint32(0); i < count; i++ {
		flags := *(*uint32)(unsafe.Pointer(metaBase + off))
		nameLen := *(*uint32)(unsafe.Pointer(metaBase + off + 4))
		nameBytes := unsafe.Slice((*byte)(unsafe.Pointer(metaBase+off+8)), int(nameLen))
		name := string(nameBytes)
		off += 8 + uintptr(nameLen)
		if pad := off & 3; pad != 0 {
			off += 4 - pad
		}
		addr, _ := syscall.GetProcAddress(hostMod, name)
		if addr == 0 {
			if flags&1 != 0 {
				return errors.New("host EXE does not export required plugin symbol: " + name)
			}
			continue
		}
		*(*uintptr)(unsafe.Pointer(slotsBase + uintptr(i)*8)) = addr
	}
	return nil
}

var (
	modKernel32        = syscall.NewLazyDLL("kernel32.dll")
	procVirtualProtect = modKernel32.NewProc("VirtualProtect")
)

func virtualProtect(addr, size uintptr, newProtect uint32, oldProtect *uint32) error {
	r1, _, e := syscall.SyscallN(procVirtualProtect.Addr(), addr, size, uintptr(newProtect), uintptr(unsafe.Pointer(oldProtect)))
	if r1 == 0 {
		return e
	}
	return nil
}

var procGetModuleHandleW = modKernel32.NewProc("GetModuleHandleW")

func getModuleHandleNull() (syscall.Handle, error) {
	r1, _, e := syscall.SyscallN(procGetModuleHandleW.Addr(), 0)
	if r1 == 0 {
		return 0, e
	}
	return syscall.Handle(r1), nil
}
