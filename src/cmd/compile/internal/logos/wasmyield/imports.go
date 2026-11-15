// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasmyield

import (
	"strings"
	"sync"

	"cmd/compile/internal/logos/facts"
	"cmd/internal/objabi"
)

var pureImports = func() map[string]bool {
	m := make(map[string]bool)
	for _, n := range []string{
		"args_get", "args_sizes_get", "clock_time_get", "environ_get",
		"environ_sizes_get", "fd_close", "fd_fdstat_get", "fd_fdstat_set_flags",
		"fd_fdstat_set_rights", "fd_filestat_get", "fd_filestat_set_size",
		"fd_pread", "fd_prestat_dir_name", "fd_prestat_get", "fd_pwrite",
		"fd_read", "fd_readdir", "fd_seek", "fd_sync", "fd_write",
		"path_create_directory", "path_filestat_get", "path_filestat_set_times",
		"path_link", "path_open", "path_readlink", "path_remove_directory",
		"path_rename", "path_symlink", "path_unlink_file", "proc_exit",
		"random_get", "sock_shutdown",
	} {
		m["wasi_snapshot_preview1."+n] = true
	}
	for _, n := range []string{
		"clearTimeoutEvent", "getRandomData", "nanotime1",
		"resetMemoryDataView", "scheduleTimeoutEvent", "walltime",
		"wasmExit", "wasmWrite",
	} {
		m["gojs.runtime."+n] = true
	}
	return m
}()

var pureAsmNames = map[string]bool{
	"runtime.memmove":                  true,
	"runtime.memclrNoHeapPointers":     true,
	"runtime.memequal":                 true,
	"runtime.memequal_varlen":          true,
	"runtime.cmpstring":                true,
	"internal/bytealg.Compare":         true,
	"internal/bytealg.IndexByte":       true,
	"internal/bytealg.IndexByteString": true,
}

func pureAsm(name string) bool {
	return pureAsmNames[name]
}

func importYields(module, name string) bool {
	return !pureImports[module+"."+name]
}

var (
	indexMu sync.Mutex
	indexes = map[string]map[string]bool{}
)

func importedYield(name string) bool {
	pkg, object, ok := splitName(name)
	if !ok {
		return true
	}
	yield, ok := factIndex(pkg)[object]
	return !ok || yield
}

func splitName(name string) (pkg, object string, ok bool) {
	dir, file, slashed := strings.CutLast(name, "/")
	if !slashed {
		dir, file = "", name
	}
	last, object, ok := strings.Cut(file, ".")
	if !ok {
		return "", "", false
	}
	prefix := last
	if slashed {
		prefix = dir + "/" + last
	}
	pkg, err := objabi.PrefixToPath(prefix)
	if err != nil {
		return "", "", false
	}
	return pkg, object, true
}

func factIndex(pkg string) map[string]bool {
	indexMu.Lock()
	defer indexMu.Unlock()
	if idx, ok := indexes[pkg]; ok {
		return idx
	}
	var idx map[string]bool
	if f, ok := facts.Lookup(pkg); ok {
		idx = make(map[string]bool, len(f.Fns))
		for _, fn := range f.Fns {
			if fn.HasMayYield {
				idx[fn.Object] = fn.MayYield
			}
		}
	}
	indexes[pkg] = idx
	return idx
}
