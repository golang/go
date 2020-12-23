// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/internal/obj"

	"path"
	"sort"
	"strconv"
	"strings"
)

const (
	embedUnknown = iota
	embedBytes
	embedString
	embedFiles
)

func varEmbed(p *noder, names []*ir.Name, typ ir.Ntype, exprs []ir.Node, embeds []PragmaEmbed) (newExprs []ir.Node) {
	haveEmbed := false
	for _, decl := range p.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok {
			// imports always come first
			break
		}
		path, _ := strconv.Unquote(imp.Path.Value)
		if path == "embed" {
			haveEmbed = true
			break
		}
	}

	pos := embeds[0].Pos
	if !haveEmbed {
		p.errorAt(pos, "invalid go:embed: missing import \"embed\"")
		return exprs
	}
	if base.Flag.Cfg.Embed.Patterns == nil {
		p.errorAt(pos, "invalid go:embed: build system did not supply embed configuration")
		return exprs
	}
	if len(names) > 1 {
		p.errorAt(pos, "go:embed cannot apply to multiple vars")
		return exprs
	}
	if len(exprs) > 0 {
		p.errorAt(pos, "go:embed cannot apply to var with initializer")
		return exprs
	}
	if typ == nil {
		// Should not happen, since len(exprs) == 0 now.
		p.errorAt(pos, "go:embed cannot apply to var without type")
		return exprs
	}
	if dclcontext != ir.PEXTERN {
		p.errorAt(pos, "go:embed cannot apply to var inside func")
		return exprs
	}

	v := names[0]
	Target.Embeds = append(Target.Embeds, v)
	v.Embed = new([]ir.Embed)
	for _, e := range embeds {
		*v.Embed = append(*v.Embed, ir.Embed{Pos: p.makeXPos(e.Pos), Patterns: e.Patterns})
	}
	return exprs
}

func embedFileList(v *ir.Name) []string {
	kind := embedKind(v.Type())
	if kind == embedUnknown {
		base.ErrorfAt(v.Pos(), "go:embed cannot apply to var of type %v", v.Type())
		return nil
	}

	// Build list of files to store.
	have := make(map[string]bool)
	var list []string
	for _, e := range *v.Embed {
		for _, pattern := range e.Patterns {
			files, ok := base.Flag.Cfg.Embed.Patterns[pattern]
			if !ok {
				base.ErrorfAt(e.Pos, "invalid go:embed: build system did not map pattern: %s", pattern)
			}
			for _, file := range files {
				if base.Flag.Cfg.Embed.Files[file] == "" {
					base.ErrorfAt(e.Pos, "invalid go:embed: build system did not map file: %s", file)
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
	sort.Slice(list, func(i, j int) bool {
		return embedFileLess(list[i], list[j])
	})

	if kind == embedString || kind == embedBytes {
		if len(list) > 1 {
			base.ErrorfAt(v.Pos(), "invalid go:embed: multiple files for type %v", v.Type())
			return nil
		}
	}

	return list
}

// embedKindApprox determines the kind of embedding variable, approximately.
// The match is approximate because we haven't done scope resolution yet and
// can't tell whether "string" and "byte" really mean "string" and "byte".
// The result must be confirmed later, after type checking, using embedKind.
func embedKindApprox(typ ir.Node) int {
	if typ.Sym() != nil && typ.Sym().Name == "FS" && (typ.Sym().Pkg.Path == "embed" || (typ.Sym().Pkg == types.LocalPkg && base.Ctxt.Pkgpath == "embed")) {
		return embedFiles
	}
	// These are not guaranteed to match only string and []byte -
	// maybe the local package has redefined one of those words.
	// But it's the best we can do now during the noder.
	// The stricter check happens later, in initEmbed calling embedKind.
	if typ.Sym() != nil && typ.Sym().Name == "string" && typ.Sym().Pkg == types.LocalPkg {
		return embedString
	}
	if typ, ok := typ.(*ir.SliceType); ok {
		if sym := typ.Elem.Sym(); sym != nil && sym.Name == "byte" && sym.Pkg == types.LocalPkg {
			return embedBytes
		}
	}
	return embedUnknown
}

// embedKind determines the kind of embedding variable.
func embedKind(typ *types.Type) int {
	if typ.Sym() != nil && typ.Sym().Name == "FS" && (typ.Sym().Pkg.Path == "embed" || (typ.Sym().Pkg == types.LocalPkg && base.Ctxt.Pkgpath == "embed")) {
		return embedFiles
	}
	if typ == types.Types[types.TSTRING] {
		return embedString
	}
	if typ.Sym() == nil && typ.IsSlice() && typ.Elem() == types.ByteType {
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

func dumpembeds() {
	for _, v := range Target.Embeds {
		initEmbed(v)
	}
}

// initEmbed emits the init data for a //go:embed variable,
// which is either a string, a []byte, or an embed.FS.
func initEmbed(v *ir.Name) {
	files := embedFileList(v)
	switch kind := embedKind(v.Type()); kind {
	case embedUnknown:
		base.ErrorfAt(v.Pos(), "go:embed cannot apply to var of type %v", v.Type())

	case embedString, embedBytes:
		file := files[0]
		fsym, size, err := fileStringSym(v.Pos(), base.Flag.Cfg.Embed.Files[file], kind == embedString, nil)
		if err != nil {
			base.ErrorfAt(v.Pos(), "embed %s: %v", file, err)
		}
		sym := v.Sym().Linksym()
		off := 0
		off = dsymptr(sym, off, fsym, 0)       // data string
		off = duintptr(sym, off, uint64(size)) // len
		if kind == embedBytes {
			duintptr(sym, off, uint64(size)) // cap for slice
		}

	case embedFiles:
		slicedata := base.Ctxt.Lookup(`"".` + v.Sym().Name + `.files`)
		off := 0
		// []files pointed at by Files
		off = dsymptr(slicedata, off, slicedata, 3*types.PtrSize) // []file, pointing just past slice
		off = duintptr(slicedata, off, uint64(len(files)))
		off = duintptr(slicedata, off, uint64(len(files)))

		// embed/embed.go type file is:
		//	name string
		//	data string
		//	hash [16]byte
		// Emit one of these per file in the set.
		const hashSize = 16
		hash := make([]byte, hashSize)
		for _, file := range files {
			off = dsymptr(slicedata, off, stringsym(v.Pos(), file), 0) // file string
			off = duintptr(slicedata, off, uint64(len(file)))
			if strings.HasSuffix(file, "/") {
				// entry for directory - no data
				off = duintptr(slicedata, off, 0)
				off = duintptr(slicedata, off, 0)
				off += hashSize
			} else {
				fsym, size, err := fileStringSym(v.Pos(), base.Flag.Cfg.Embed.Files[file], true, hash)
				if err != nil {
					base.ErrorfAt(v.Pos(), "embed %s: %v", file, err)
				}
				off = dsymptr(slicedata, off, fsym, 0) // data string
				off = duintptr(slicedata, off, uint64(size))
				off = int(slicedata.WriteBytes(base.Ctxt, int64(off), hash))
			}
		}
		ggloblsym(slicedata, int32(off), obj.RODATA|obj.LOCAL)
		sym := v.Sym().Linksym()
		dsymptr(sym, 0, slicedata, 0)
	}
}
