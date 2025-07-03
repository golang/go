// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/buildcfg"
	"internal/types/errors"
	"regexp"
	"sort"

	"cmd/compile/internal/base"
	"cmd/compile/internal/rangefunc"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

var versionErrorRx = regexp.MustCompile(`requires go[0-9]+\.[0-9]+ or later`)

// checkFiles configures and runs the types2 checker on the given
// parsed source files and then returns the result.
// The map result value indicates which closures are generated from the bodies of range function loops.
func checkFiles(m posMap, noders []*noder) (*types2.Package, *types2.Info, map[*syntax.FuncLit]bool) {
	if base.SyntaxErrors() != 0 {
		base.ErrorExit()
	}

	// setup and syntax error reporting
	files := make([]*syntax.File, len(noders))
	// fileBaseMap maps all file pos bases back to *syntax.File
	// for checking Go version mismatched.
	fileBaseMap := make(map[*syntax.PosBase]*syntax.File)
	for i, p := range noders {
		files[i] = p.file
		// The file.Pos() is the position of the package clause.
		// If there's a //line directive before that, file.Pos().Base()
		// refers to that directive, not the file itself.
		// Make sure to consistently map back to file base, here and
		// when we look for a file in the conf.Error handler below,
		// otherwise the file may not be found (was go.dev/issue/67141).
		fileBaseMap[p.file.Pos().FileBase()] = p.file
	}

	// typechecking
	ctxt := types2.NewContext()
	importer := gcimports{
		ctxt:     ctxt,
		packages: make(map[string]*types2.Package),
	}
	conf := types2.Config{
		Context:            ctxt,
		GoVersion:          base.Flag.Lang,
		IgnoreBranchErrors: true, // parser already checked via syntax.CheckBranches mode
		Importer:           &importer,
		Sizes:              types2.SizesFor("gc", buildcfg.GOARCH),
		EnableAlias:        true,
	}
	if base.Flag.ErrorURL {
		conf.ErrorURL = " [go.dev/e/%s]"
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Implicits:          make(map[syntax.Node]types2.Object),
		Scopes:             make(map[syntax.Node]*types2.Scope),
		Instances:          make(map[*syntax.Name]types2.Instance),
		FileVersions:       make(map[*syntax.PosBase]string),
		// expand as needed
	}
	conf.Error = func(err error) {
		terr := err.(types2.Error)
		msg := terr.Msg
		if versionErrorRx.MatchString(msg) {
			fileBase := terr.Pos.FileBase()
			fileVersion := info.FileVersions[fileBase]
			file := fileBaseMap[fileBase]
			if file == nil {
				// This should never happen, but be careful and don't crash.
			} else if file.GoVersion == fileVersion {
				// If we have a version error caused by //go:build, report it.
				msg = fmt.Sprintf("%s (file declares //go:build %s)", msg, fileVersion)
			} else {
				// Otherwise, hint at the -lang setting.
				msg = fmt.Sprintf("%s (-lang was set to %s; check go.mod)", msg, base.Flag.Lang)
			}
		}
		base.ErrorfAt(m.makeXPos(terr.Pos), terr.Code, "%s", msg)
	}

	pkg, err := conf.Check(base.Ctxt.Pkgpath, files, info)
	base.ExitIfErrors()
	if err != nil {
		base.FatalfAt(src.NoXPos, "conf.Check error: %v", err)
	}

	// Check for anonymous interface cycles (#56103).
	// TODO(gri) move this code into the type checkers (types2 and go/types)
	var f cycleFinder
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			if n, ok := n.(*syntax.InterfaceType); ok {
				if f.hasCycle(types2.Unalias(n.GetTypeInfo().Type).(*types2.Interface)) {
					base.ErrorfAt(m.makeXPos(n.Pos()), errors.InvalidTypeCycle, "invalid recursive type: anonymous interface refers to itself (see https://go.dev/issue/56103)")

					for typ := range f.cyclic {
						f.cyclic[typ] = false // suppress duplicate errors
					}
				}
				return false
			}
			return true
		})
	}
	base.ExitIfErrors()

	// Implementation restriction: we don't allow not-in-heap types to
	// be used as type arguments (#54765).
	{
		type nihTarg struct {
			pos src.XPos
			typ types2.Type
		}
		var nihTargs []nihTarg

		for name, inst := range info.Instances {
			for i := 0; i < inst.TypeArgs.Len(); i++ {
				if targ := inst.TypeArgs.At(i); isNotInHeap(targ) {
					nihTargs = append(nihTargs, nihTarg{m.makeXPos(name.Pos()), targ})
				}
			}
		}
		sort.Slice(nihTargs, func(i, j int) bool {
			ti, tj := nihTargs[i], nihTargs[j]
			return ti.pos.Before(tj.pos)
		})
		for _, targ := range nihTargs {
			base.ErrorfAt(targ.pos, 0, "cannot use incomplete (or unallocatable) type as a type argument: %v", targ.typ)
		}
	}
	base.ExitIfErrors()

	// Implementation restriction: we don't allow not-in-heap types to
	// be used as map keys/values, or channel.
	{
		for _, file := range files {
			syntax.Inspect(file, func(n syntax.Node) bool {
				if n, ok := n.(*syntax.TypeDecl); ok {
					switch n := n.Type.(type) {
					case *syntax.MapType:
						typ := n.GetTypeInfo().Type.Underlying().(*types2.Map)
						if isNotInHeap(typ.Key()) {
							base.ErrorfAt(m.makeXPos(n.Pos()), 0, "incomplete (or unallocatable) map key not allowed")
						}
						if isNotInHeap(typ.Elem()) {
							base.ErrorfAt(m.makeXPos(n.Pos()), 0, "incomplete (or unallocatable) map value not allowed")
						}
					case *syntax.ChanType:
						typ := n.GetTypeInfo().Type.Underlying().(*types2.Chan)
						if isNotInHeap(typ.Elem()) {
							base.ErrorfAt(m.makeXPos(n.Pos()), 0, "chan of incomplete (or unallocatable) type not allowed")
						}
					}
				}
				return true
			})
		}
	}
	base.ExitIfErrors()

	// Rewrite range over function to explicit function calls
	// with the loop bodies converted into new implicit closures.
	// We do this now, before serialization to unified IR, so that if the
	// implicit closures are inlined, we will have the unified IR form.
	// If we do the rewrite in the back end, like between typecheck and walk,
	// then the new implicit closure will not have a unified IR inline body,
	// and bodyReaderFor will fail.
	rangeInfo := rangefunc.Rewrite(pkg, info, files)

	return pkg, info, rangeInfo
}

// A cycleFinder detects anonymous interface cycles (go.dev/issue/56103).
type cycleFinder struct {
	cyclic map[*types2.Interface]bool
}

// hasCycle reports whether typ is part of an anonymous interface cycle.
func (f *cycleFinder) hasCycle(typ *types2.Interface) bool {
	// We use Method instead of ExplicitMethod to implicitly expand any
	// embedded interfaces. Then we just need to walk any anonymous
	// types, keeping track of *types2.Interface types we visit along
	// the way.
	for i := 0; i < typ.NumMethods(); i++ {
		if f.visit(typ.Method(i).Type()) {
			return true
		}
	}
	return false
}

// visit recursively walks typ0 to check any referenced interface types.
func (f *cycleFinder) visit(typ0 types2.Type) bool {
	for { // loop for tail recursion
		switch typ := types2.Unalias(typ0).(type) {
		default:
			base.Fatalf("unexpected type: %T", typ)

		case *types2.Basic, *types2.Named, *types2.TypeParam:
			return false // named types cannot be part of an anonymous cycle
		case *types2.Pointer:
			typ0 = typ.Elem()
		case *types2.Array:
			typ0 = typ.Elem()
		case *types2.Chan:
			typ0 = typ.Elem()
		case *types2.Map:
			if f.visit(typ.Key()) {
				return true
			}
			typ0 = typ.Elem()
		case *types2.Slice:
			typ0 = typ.Elem()

		case *types2.Struct:
			for i := 0; i < typ.NumFields(); i++ {
				if f.visit(typ.Field(i).Type()) {
					return true
				}
			}
			return false

		case *types2.Interface:
			// The empty interface (e.g., "any") cannot be part of a cycle.
			if typ.NumExplicitMethods() == 0 && typ.NumEmbeddeds() == 0 {
				return false
			}

			// As an optimization, we wait to allocate cyclic here, after
			// we've found at least one other (non-empty) anonymous
			// interface. This means when a cycle is present, we need to
			// make an extra recursive call to actually detect it. But for
			// most packages, it allows skipping the map allocation
			// entirely.
			if x, ok := f.cyclic[typ]; ok {
				return x
			}
			if f.cyclic == nil {
				f.cyclic = make(map[*types2.Interface]bool)
			}
			f.cyclic[typ] = true
			if f.hasCycle(typ) {
				return true
			}
			f.cyclic[typ] = false
			return false

		case *types2.Signature:
			return f.visit(typ.Params()) || f.visit(typ.Results())
		case *types2.Tuple:
			for i := 0; i < typ.Len(); i++ {
				if f.visit(typ.At(i).Type()) {
					return true
				}
			}
			return false
		}
	}
}
