// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package staticdata

import (
	"crypto/sha256"
	"fmt"
	"go/constant"
	"internal/buildcfg"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strconv"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// InitAddrOffset writes the static name symbol lsym to n, it does not modify n.
// It's the caller responsibility to make sure lsym is from ONAME/PEXTERN node.
func InitAddrOffset(n *ir.Name, noff int64, lsym *obj.LSym, off int64) {
	if n.Op() != ir.ONAME {
		base.Fatalf("InitAddr n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("InitAddr nil n sym")
	}
	s := n.Linksym()
	s.WriteAddr(base.Ctxt, noff, types.PtrSize, lsym, off)
}

// InitAddr is InitAddrOffset, with offset fixed to 0.
func InitAddr(n *ir.Name, noff int64, lsym *obj.LSym) {
	InitAddrOffset(n, noff, lsym, 0)
}

// InitSlice writes a static slice symbol {lsym, lencap, lencap} to n+noff, it does not modify n.
// It's the caller responsibility to make sure lsym is from ONAME node.
func InitSlice(n *ir.Name, noff int64, lsym *obj.LSym, lencap int64) {
	s := n.Linksym()
	s.WriteAddr(base.Ctxt, noff, types.PtrSize, lsym, 0)
	s.WriteInt(base.Ctxt, noff+types.SliceLenOffset, types.PtrSize, lencap)
	s.WriteInt(base.Ctxt, noff+types.SliceCapOffset, types.PtrSize, lencap)
}

func InitSliceBytes(nam *ir.Name, off int64, s string) {
	if nam.Op() != ir.ONAME {
		base.Fatalf("InitSliceBytes %v", nam)
	}
	InitSlice(nam, off, slicedata(nam.Pos(), s).Linksym(), int64(len(s)))
}

const (
	stringSymPrefix  = "go.string."
	stringSymPattern = ".gostring.%d.%x"
)

// StringSym returns a symbol containing the string s.
// The symbol contains the string data, not a string header.
func StringSym(pos src.XPos, s string) (data *obj.LSym) {
	var symname string
	if len(s) > 100 {
		// Huge strings are hashed to avoid long names in object files.
		// Indulge in some paranoia by writing the length of s, too,
		// as protection against length extension attacks.
		// Same pattern is known to fileStringSym below.
		h := sha256.New()
		io.WriteString(h, s)
		symname = fmt.Sprintf(stringSymPattern, len(s), h.Sum(nil))
	} else {
		// Small strings get named directly by their contents.
		symname = strconv.Quote(s)
	}

	symdata := base.Ctxt.Lookup(stringSymPrefix + symname)
	if !symdata.OnList() {
		off := dstringdata(symdata, 0, s, pos, "string")
		objw.Global(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)
		symdata.Set(obj.AttrContentAddressable, true)
	}

	return symdata
}

// maxFileSize is the maximum file size permitted by the linker
// (see issue #9862).
const maxFileSize = int64(2e9)

// fileStringSym returns a symbol for the contents and the size of file.
// If readonly is true, the symbol shares storage with any literal string
// or other file with the same content and is placed in a read-only section.
// If readonly is false, the symbol is a read-write copy separate from any other,
// for use as the backing store of a []byte.
// The content hash of file is copied into hash. (If hash is nil, nothing is copied.)
// The returned symbol contains the data itself, not a string header.
func fileStringSym(pos src.XPos, file string, readonly bool, hash []byte) (*obj.LSym, int64, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	if !info.Mode().IsRegular() {
		return nil, 0, fmt.Errorf("not a regular file")
	}
	size := info.Size()
	if size <= 1*1024 {
		data, err := ioutil.ReadAll(f)
		if err != nil {
			return nil, 0, err
		}
		if int64(len(data)) != size {
			return nil, 0, fmt.Errorf("file changed between reads")
		}
		var sym *obj.LSym
		if readonly {
			sym = StringSym(pos, string(data))
		} else {
			sym = slicedata(pos, string(data)).Linksym()
		}
		if len(hash) > 0 {
			sum := sha256.Sum256(data)
			copy(hash, sum[:])
		}
		return sym, size, nil
	}
	if size > maxFileSize {
		// ggloblsym takes an int32,
		// and probably the rest of the toolchain
		// can't handle such big symbols either.
		// See golang.org/issue/9862.
		return nil, 0, fmt.Errorf("file too large (%d bytes > %d bytes)", size, maxFileSize)
	}

	// File is too big to read and keep in memory.
	// Compute hash if needed for read-only content hashing or if the caller wants it.
	var sum []byte
	if readonly || len(hash) > 0 {
		h := sha256.New()
		n, err := io.Copy(h, f)
		if err != nil {
			return nil, 0, err
		}
		if n != size {
			return nil, 0, fmt.Errorf("file changed between reads")
		}
		sum = h.Sum(nil)
		copy(hash, sum)
	}

	var symdata *obj.LSym
	if readonly {
		symname := fmt.Sprintf(stringSymPattern, size, sum)
		symdata = base.Ctxt.Lookup(stringSymPrefix + symname)
		if !symdata.OnList() {
			info := symdata.NewFileInfo()
			info.Name = file
			info.Size = size
			objw.Global(symdata, int32(size), obj.DUPOK|obj.RODATA|obj.LOCAL)
			// Note: AttrContentAddressable cannot be set here,
			// because the content-addressable-handling code
			// does not know about file symbols.
		}
	} else {
		// Emit a zero-length data symbol
		// and then fix up length and content to use file.
		symdata = slicedata(pos, "").Linksym()
		symdata.Size = size
		symdata.Type = objabi.SNOPTRDATA
		info := symdata.NewFileInfo()
		info.Name = file
		info.Size = size
	}

	return symdata, size, nil
}

var slicedataGen int

func slicedata(pos src.XPos, s string) *ir.Name {
	slicedataGen++
	symname := fmt.Sprintf(".gobytes.%d", slicedataGen)
	sym := types.LocalPkg.Lookup(symname)
	symnode := typecheck.NewName(sym)
	sym.Def = symnode

	lsym := symnode.Linksym()
	off := dstringdata(lsym, 0, s, pos, "slice")
	objw.Global(lsym, int32(off), obj.NOPTR|obj.LOCAL)

	return symnode
}

func dstringdata(s *obj.LSym, off int, t string, pos src.XPos, what string) int {
	// Objects that are too large will cause the data section to overflow right away,
	// causing a cryptic error message by the linker. Check for oversize objects here
	// and provide a useful error message instead.
	if int64(len(t)) > 2e9 {
		base.ErrorfAt(pos, "%v with length %v is too big", what, len(t))
		return 0
	}

	s.WriteString(base.Ctxt, int64(off), len(t), t)
	return off + len(t)
}

var (
	funcsymsmu sync.Mutex // protects funcsyms and associated package lookups (see func funcsym)
	funcsyms   []*ir.Name // functions that need function value symbols
)

// FuncLinksym returns n·f, the function value symbol for n.
func FuncLinksym(n *ir.Name) *obj.LSym {
	if n.Op() != ir.ONAME || n.Class != ir.PFUNC {
		base.Fatalf("expected func name: %v", n)
	}
	s := n.Sym()

	// funcsymsmu here serves to protect not just mutations of funcsyms (below),
	// but also the package lookup of the func sym name,
	// since this function gets called concurrently from the backend.
	// There are no other concurrent package lookups in the backend,
	// except for the types package, which is protected separately.
	// Reusing funcsymsmu to also cover this package lookup
	// avoids a general, broader, expensive package lookup mutex.
	// Note NeedFuncSym also does package look-up of func sym names,
	// but that it is only called serially, from the front end.
	funcsymsmu.Lock()
	sf, existed := s.Pkg.LookupOK(ir.FuncSymName(s))
	// Don't export s·f when compiling for dynamic linking.
	// When dynamically linking, the necessary function
	// symbols will be created explicitly with NeedFuncSym.
	// See the NeedFuncSym comment for details.
	if !base.Ctxt.Flag_dynlink && !existed {
		funcsyms = append(funcsyms, n)
	}
	funcsymsmu.Unlock()

	return sf.Linksym()
}

func GlobalLinksym(n *ir.Name) *obj.LSym {
	if n.Op() != ir.ONAME || n.Class != ir.PEXTERN {
		base.Fatalf("expected global variable: %v", n)
	}
	return n.Linksym()
}

// NeedFuncSym ensures that fn·f is exported, if needed.
// It is only used with -dynlink.
// When not compiling for dynamic linking,
// the funcsyms are created as needed by
// the packages that use them.
// Normally we emit the fn·f stubs as DUPOK syms,
// but DUPOK doesn't work across shared library boundaries.
// So instead, when dynamic linking, we only create
// the fn·f stubs in fn's package.
func NeedFuncSym(fn *ir.Func) {
	if base.Ctxt.InParallel {
		// The append below probably just needs to lock
		// funcsymsmu, like in FuncSym.
		base.Fatalf("NeedFuncSym must be called in serial")
	}
	if fn.ABI != obj.ABIInternal && buildcfg.Experiment.RegabiWrappers {
		// Function values must always reference ABIInternal
		// entry points, so it doesn't make sense to create a
		// funcsym for other ABIs.
		//
		// (If we're not using ABI wrappers, it doesn't matter.)
		base.Fatalf("expected ABIInternal: %v has %v", fn.Nname, fn.ABI)
	}
	if ir.IsBlank(fn.Nname) {
		// Blank functions aren't unique, so we can't make a
		// funcsym for them.
		base.Fatalf("NeedFuncSym called for _")
	}
	if !base.Ctxt.Flag_dynlink {
		return
	}
	s := fn.Nname.Sym()
	if base.Flag.CompilingRuntime && (s.Name == "getg" || s.Name == "getclosureptr" || s.Name == "getcallerpc" || s.Name == "getcallersp") ||
		(base.Ctxt.Pkgpath == "internal/abi" && (s.Name == "FuncPCABI0" || s.Name == "FuncPCABIInternal")) {
		// runtime.getg(), getclosureptr(), getcallerpc(), getcallersp(),
		// and internal/abi.FuncPCABIxxx() are not real functions and so
		// do not get funcsyms.
		return
	}
	funcsyms = append(funcsyms, fn.Nname)
}

func WriteFuncSyms() {
	sort.Slice(funcsyms, func(i, j int) bool {
		return funcsyms[i].Linksym().Name < funcsyms[j].Linksym().Name
	})
	for _, nam := range funcsyms {
		s := nam.Sym()
		sf := s.Pkg.Lookup(ir.FuncSymName(s)).Linksym()
		// Function values must always reference ABIInternal
		// entry points.
		target := s.Linksym()
		if target.ABI() != obj.ABIInternal {
			base.Fatalf("expected ABIInternal: %v has %v", target, target.ABI())
		}
		objw.SymPtr(sf, 0, target, 0)
		objw.Global(sf, int32(types.PtrSize), obj.DUPOK|obj.RODATA)
	}
}

// InitConst writes the static literal c to n.
// Neither n nor c is modified.
func InitConst(n *ir.Name, noff int64, c ir.Node, wid int) {
	if n.Op() != ir.ONAME {
		base.Fatalf("InitConst n op %v", n.Op())
	}
	if n.Sym() == nil {
		base.Fatalf("InitConst nil n sym")
	}
	if c.Op() == ir.ONIL {
		return
	}
	if c.Op() != ir.OLITERAL {
		base.Fatalf("InitConst c op %v", c.Op())
	}
	s := n.Linksym()
	switch u := c.Val(); u.Kind() {
	case constant.Bool:
		i := int64(obj.Bool2int(constant.BoolVal(u)))
		s.WriteInt(base.Ctxt, noff, wid, i)

	case constant.Int:
		s.WriteInt(base.Ctxt, noff, wid, ir.IntVal(c.Type(), u))

	case constant.Float:
		f, _ := constant.Float64Val(u)
		switch c.Type().Kind() {
		case types.TFLOAT32:
			s.WriteFloat32(base.Ctxt, noff, float32(f))
		case types.TFLOAT64:
			s.WriteFloat64(base.Ctxt, noff, f)
		}

	case constant.Complex:
		re, _ := constant.Float64Val(constant.Real(u))
		im, _ := constant.Float64Val(constant.Imag(u))
		switch c.Type().Kind() {
		case types.TCOMPLEX64:
			s.WriteFloat32(base.Ctxt, noff, float32(re))
			s.WriteFloat32(base.Ctxt, noff+4, float32(im))
		case types.TCOMPLEX128:
			s.WriteFloat64(base.Ctxt, noff, re)
			s.WriteFloat64(base.Ctxt, noff+8, im)
		}

	case constant.String:
		i := constant.StringVal(u)
		symdata := StringSym(n.Pos(), i)
		s.WriteAddr(base.Ctxt, noff, types.PtrSize, symdata, 0)
		s.WriteInt(base.Ctxt, noff+int64(types.PtrSize), types.PtrSize, int64(len(i)))

	default:
		base.Fatalf("InitConst unhandled OLITERAL %v", c)
	}
}
