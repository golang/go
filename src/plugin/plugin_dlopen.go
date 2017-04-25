// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,cgo darwin,cgo

package plugin

/*
#cgo linux LDFLAGS: -ldl
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#include <stdio.h>

static uintptr_t pluginOpen(const char* path, char** err) {
	void* h = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
	if (h == NULL) {
		*err = (char*)dlerror();
	}
	return (uintptr_t)h;
}

static void* pluginLookup(uintptr_t h, const char* name, char** err) {
	void* r = dlsym((void*)h, name);
	if (r == NULL) {
		*err = (char*)dlerror();
	}
	return r;
}
*/
import "C"

import (
	"errors"
	"sync"
	"unsafe"
)

// avoid a dependency on strings
func lastIndexByte(s string, c byte) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == c {
			return i
		}
	}
	return -1
}

// pathToPrefix converts raw string to the prefix that will be used in the symbol
// table. If modifying, modify the version in internal/obj/sym.go as well.
func pathToPrefix(s string) string {
	slash := lastIndexByte(s, '/')
	// check for chars that need escaping
	n := 0
	for r := 0; r < len(s); r++ {
		if c := s[r]; c <= ' ' || (c == '.' && r > slash) || c == '%' || c == '"' || c >= 0x7F {
			n++
		}
	}

	// quick exit
	if n == 0 {
		return s
	}

	// escape
	const hex = "0123456789abcdef"
	p := make([]byte, 0, len(s)+2*n)
	for r := 0; r < len(s); r++ {
		if c := s[r]; c <= ' ' || (c == '.' && r > slash) || c == '%' || c == '"' || c >= 0x7F {
			p = append(p, '%', hex[c>>4], hex[c&0xF])
		} else {
			p = append(p, c)
		}
	}

	return string(p)
}

func open(name string) (*Plugin, error) {
	cPath := (*C.char)(C.malloc(C.PATH_MAX + 1))
	defer C.free(unsafe.Pointer(cPath))

	cRelName := C.CString(name)
	defer C.free(unsafe.Pointer(cRelName))
	if C.realpath(cRelName, cPath) == nil {
		return nil, errors.New("plugin.Open(" + name + "): realpath failed")
	}

	filepath := C.GoString(cPath)

	pluginsMu.Lock()
	if p := plugins[filepath]; p != nil {
		pluginsMu.Unlock()
		<-p.loaded
		return p, nil
	}
	var cErr *C.char
	h := C.pluginOpen(cPath, &cErr)
	if h == 0 {
		pluginsMu.Unlock()
		return nil, errors.New("plugin.Open: " + C.GoString(cErr))
	}
	// TODO(crawshaw): look for plugin note, confirm it is a Go plugin
	// and it was built with the correct toolchain.
	if len(name) > 3 && name[len(name)-3:] == ".so" {
		name = name[:len(name)-3]
	}

	pluginpath, syms, mismatchpkg := lastmoduleinit()
	if mismatchpkg != "" {
		pluginsMu.Unlock()
		return nil, errors.New("plugin.Open: plugin was built with a different version of package " + mismatchpkg)
	}
	if plugins == nil {
		plugins = make(map[string]*Plugin)
	}
	// This function can be called from the init function of a plugin.
	// Drop a placeholder in the map so subsequent opens can wait on it.
	p := &Plugin{
		pluginpath: pluginpath,
		loaded:     make(chan struct{}),
	}
	plugins[filepath] = p
	pluginsMu.Unlock()

	initStr := C.CString(pluginpath + ".init")
	initFuncPC := C.pluginLookup(h, initStr, &cErr)
	C.free(unsafe.Pointer(initStr))
	if initFuncPC != nil {
		initFuncP := &initFuncPC
		initFunc := *(*func())(unsafe.Pointer(&initFuncP))
		initFunc()
	}

	// Fill out the value of each plugin symbol.
	updatedSyms := map[string]interface{}{}
	for symName, sym := range syms {
		isFunc := symName[0] == '.'
		if isFunc {
			delete(syms, symName)
			symName = symName[1:]
		}
		cname := C.CString(pathToPrefix(pluginpath) + "." + symName)
		p := C.pluginLookup(h, cname, &cErr)
		C.free(unsafe.Pointer(cname))
		if p == nil {
			return nil, errors.New("plugin.Open: could not find symbol " + symName + ": " + C.GoString(cErr))
		}
		valp := (*[2]unsafe.Pointer)(unsafe.Pointer(&sym))
		if isFunc {
			(*valp)[1] = unsafe.Pointer(&p)
		} else {
			(*valp)[1] = p
		}
		// we can't add to syms during iteration as we'll end up processing
		// some symbols twice with the inability to tell if the symbol is a function
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

// lastmoduleinit is defined in package runtime
func lastmoduleinit() (pluginpath string, syms map[string]interface{}, mismatchpkg string)
