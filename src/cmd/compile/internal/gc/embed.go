// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"encoding/json"
	"io/ioutil"
	"log"
	"path"
	"sort"
	"strconv"
	"strings"
)

var embedlist []*Node

var embedCfg struct {
	Patterns map[string][]string
	Files    map[string]string
}

func readEmbedCfg(file string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-embedcfg: %v", err)
	}
	if err := json.Unmarshal(data, &embedCfg); err != nil {
		log.Fatalf("%s: %v", file, err)
	}
	if embedCfg.Patterns == nil {
		log.Fatalf("%s: invalid embedcfg: missing Patterns", file)
	}
	if embedCfg.Files == nil {
		log.Fatalf("%s: invalid embedcfg: missing Files", file)
	}
}

const (
	embedUnknown = iota
	embedBytes
	embedString
	embedFiles
)

func varEmbed(p *noder, names []*Node, typ *Node, exprs []*Node, embeds []PragmaEmbed) {
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
		p.yyerrorpos(pos, "invalid go:embed: missing import \"embed\"")
		return
	}
	if len(names) > 1 {
		p.yyerrorpos(pos, "go:embed cannot apply to multiple vars")
		return
	}
	if len(exprs) > 0 {
		p.yyerrorpos(pos, "go:embed cannot apply to var with initializer")
		return
	}
	if typ == nil {
		// Should not happen, since len(exprs) == 0 now.
		p.yyerrorpos(pos, "go:embed cannot apply to var without type")
		return
	}
	if dclcontext != PEXTERN {
		p.yyerrorpos(pos, "go:embed cannot apply to var inside func")
		return
	}

	var list []irEmbed
	for _, e := range embeds {
		list = append(list, irEmbed{Pos: p.makeXPos(e.Pos), Patterns: e.Patterns})
	}
	v := names[0]
	v.Name.Param.SetEmbedList(list)
	embedlist = append(embedlist, v)
}

func embedFileList(v *Node, kind int) []string {
	// Build list of files to store.
	have := make(map[string]bool)
	var list []string
	for _, e := range v.Name.Param.EmbedList() {
		for _, pattern := range e.Patterns {
			files, ok := embedCfg.Patterns[pattern]
			if !ok {
				yyerrorl(e.Pos, "invalid go:embed: build system did not map pattern: %s", pattern)
			}
			for _, file := range files {
				if embedCfg.Files[file] == "" {
					yyerrorl(e.Pos, "invalid go:embed: build system did not map file: %s", file)
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
			yyerrorl(v.Pos, "invalid go:embed: multiple files for type %v", v.Type)
			return nil
		}
	}

	return list
}

// embedKind determines the kind of embedding variable.
func embedKind(typ *types.Type) int {
	if typ.Sym != nil && typ.Sym.Name == "FS" && (typ.Sym.Pkg.Path == "embed" || (typ.Sym.Pkg == localpkg && myimportpath == "embed")) {
		return embedFiles
	}
	if typ.Etype == types.TSTRING {
		return embedString
	}
	if typ.Etype == types.TSLICE && typ.Elem().Etype == types.TUINT8 {
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
	for _, v := range embedlist {
		initEmbed(v)
	}
}

// initEmbed emits the init data for a //go:embed variable,
// which is either a string, a []byte, or an embed.FS.
func initEmbed(v *Node) {
	commentPos := v.Name.Param.EmbedList()[0].Pos
	if !langSupported(1, 16, localpkg) {
		lno := lineno
		lineno = commentPos
		yyerrorv("go1.16", "go:embed")
		lineno = lno
		return
	}
	if embedCfg.Patterns == nil {
		yyerrorl(commentPos, "invalid go:embed: build system did not supply embed configuration")
		return
	}
	kind := embedKind(v.Type)
	if kind == embedUnknown {
		yyerrorl(v.Pos, "go:embed cannot apply to var of type %v", v.Type)
		return
	}

	files := embedFileList(v, kind)
	switch kind {
	case embedString, embedBytes:
		file := files[0]
		fsym, size, err := fileStringSym(v.Pos, embedCfg.Files[file], kind == embedString, nil)
		if err != nil {
			yyerrorl(v.Pos, "embed %s: %v", file, err)
		}
		sym := v.Sym.Linksym()
		off := 0
		off = dsymptr(sym, off, fsym, 0)       // data string
		off = duintptr(sym, off, uint64(size)) // len
		if kind == embedBytes {
			duintptr(sym, off, uint64(size)) // cap for slice
		}

	case embedFiles:
		slicedata := Ctxt.Lookup(`"".` + v.Sym.Name + `.files`)
		off := 0
		// []files pointed at by Files
		off = dsymptr(slicedata, off, slicedata, 3*Widthptr) // []file, pointing just past slice
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
			off = dsymptr(slicedata, off, stringsym(v.Pos, file), 0) // file string
			off = duintptr(slicedata, off, uint64(len(file)))
			if strings.HasSuffix(file, "/") {
				// entry for directory - no data
				off = duintptr(slicedata, off, 0)
				off = duintptr(slicedata, off, 0)
				off += hashSize
			} else {
				fsym, size, err := fileStringSym(v.Pos, embedCfg.Files[file], true, hash)
				if err != nil {
					yyerrorl(v.Pos, "embed %s: %v", file, err)
				}
				off = dsymptr(slicedata, off, fsym, 0) // data string
				off = duintptr(slicedata, off, uint64(size))
				off = int(slicedata.WriteBytes(Ctxt, int64(off), hash))
			}
		}
		ggloblsym(slicedata, int32(off), obj.RODATA|obj.LOCAL)
		sym := v.Sym.Linksym()
		dsymptr(sym, 0, slicedata, 0)
	}
}
