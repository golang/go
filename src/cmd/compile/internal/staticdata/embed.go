// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package staticdata

import (
	"path"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

const (
	embedUnknown = iota
	embedBytes
	embedString
	embedFiles
)

func embedFileList(v *ir.Name, kind int) []string {
	// Build list of files to store.
	have := make(map[string]bool)
	var list []string
	for _, e := range *v.Embed {
		for _, pattern := range e.Patterns {
			files, ok := base.Flag.Cfg.Embed.Patterns[pattern]
			if !ok {
				base.ErrorfAt(e.Pos, 0, "invalid go:embed: build system did not map pattern: %s", pattern)
			}
			for _, file := range files {
				if base.Flag.Cfg.Embed.Files[file] == "" {
					base.ErrorfAt(e.Pos, 0, "invalid go:embed: build system did not map file: %s", file)
					continue
				}
				if !have[file] {
					have[file] = true
					list = append(list, file)
				}
				if kind == embedFiles {
					for dir := path.Dir(file); dir != "." && !have[dir]; dir = path.Dir(dir) {
						have[dir] = true
						list = append(list, dir+"/")
					}
				}
			}
		}
	}
	sort.Slice(list, func { i, j -> embedFileLess(list[i], list[j]) })

	if kind == embedString || kind == embedBytes {
		if len(list) > 1 {
			base.ErrorfAt(v.Pos(), 0, "invalid go:embed: multiple files for type %v", v.Type())
			return nil
		}
	}

	return list
}

// embedKind determines the kind of embedding variable.
func embedKind(typ *types.Type) int {
	if typ.Sym() != nil && typ.Sym().Name == "FS" && typ.Sym().Pkg.Path == "embed" {
		return embedFiles
	}
	if typ.Kind() == types.TSTRING {
		return embedString
	}
	if typ.IsSlice() && typ.Elem().Kind() == types.TUINT8 {
		return embedBytes
	}
	return embedUnknown
}

func embedFileNameSplit(name string) (dir, elem string, isDir bool) {
	if name[len(name)-1] == '/' {
		isDir = true
		name = name[:len(name)-1]
	}
	i := len(name) - 1
	for i >= 0 && name[i] != '/' {
		i--
	}
	if i < 0 {
		return ".", name, isDir
	}
	return name[:i], name[i+1:], isDir
}

// embedFileLess implements the sort order for a list of embedded files.
// See the comment inside ../../../../embed/embed.go's Files struct for rationale.
func embedFileLess(x, y string) bool {
	xdir, xelem, _ := embedFileNameSplit(x)
	ydir, yelem, _ := embedFileNameSplit(y)
	return xdir < ydir || xdir == ydir && xelem < yelem
}

// WriteEmbed emits the init data for a //go:embed variable,
// which is either a string, a []byte, or an embed.FS.
func WriteEmbed(v *ir.Name) {
	// TODO(mdempsky): User errors should be reported by the frontend.

	commentPos := (*v.Embed)[0].Pos
	if base.Flag.Cfg.Embed.Patterns == nil {
		base.ErrorfAt(commentPos, 0, "invalid go:embed: build system did not supply embed configuration")
		return
	}
	kind := embedKind(v.Type())
	if kind == embedUnknown {
		base.ErrorfAt(v.Pos(), 0, "go:embed cannot apply to var of type %v", v.Type())
		return
	}

	files := embedFileList(v, kind)
	switch kind {
	case embedString, embedBytes:
		file := files[0]
		fsym, size, err := fileStringSym(v.Pos(), base.Flag.Cfg.Embed.Files[file], kind == embedString, nil)
		if err != nil {
			base.ErrorfAt(v.Pos(), 0, "embed %s: %v", file, err)
		}
		sym := v.Linksym()
		off := 0
		off = objw.SymPtr(sym, off, fsym, 0)       // data string
		off = objw.Uintptr(sym, off, uint64(size)) // len
		if kind == embedBytes {
			objw.Uintptr(sym, off, uint64(size)) // cap for slice
		}

	case embedFiles:
		slicedata := v.Sym().Pkg.Lookup(v.Sym().Name + `.files`).Linksym()
		off := 0
		// []files pointed at by Files
		off = objw.SymPtr(slicedata, off, slicedata, 3*types.PtrSize) // []file, pointing just past slice
		off = objw.Uintptr(slicedata, off, uint64(len(files)))
		off = objw.Uintptr(slicedata, off, uint64(len(files)))

		// embed/embed.go type file is:
		//	name string
		//	data string
		//	hash [16]byte
		// Emit one of these per file in the set.
		const hashSize = 16
		hash := make([]byte, hashSize)
		for _, file := range files {
			off = objw.SymPtr(slicedata, off, StringSym(v.Pos(), file), 0) // file string
			off = objw.Uintptr(slicedata, off, uint64(len(file)))
			if strings.HasSuffix(file, "/") {
				// entry for directory - no data
				off = objw.Uintptr(slicedata, off, 0)
				off = objw.Uintptr(slicedata, off, 0)
				off += hashSize
			} else {
				fsym, size, err := fileStringSym(v.Pos(), base.Flag.Cfg.Embed.Files[file], true, hash)
				if err != nil {
					base.ErrorfAt(v.Pos(), 0, "embed %s: %v", file, err)
				}
				off = objw.SymPtr(slicedata, off, fsym, 0) // data string
				off = objw.Uintptr(slicedata, off, uint64(size))
				off = int(slicedata.WriteBytes(base.Ctxt, int64(off), hash))
			}
		}
		objw.Global(slicedata, int32(off), obj.RODATA|obj.LOCAL)
		sym := v.Linksym()
		objw.SymPtr(sym, 0, slicedata, 0)
	}
}
