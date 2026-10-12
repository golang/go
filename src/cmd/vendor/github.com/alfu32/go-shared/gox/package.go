package gox

import (
	"context"
	"fmt"
	gxparser "github.com/alfu32/go-shared/gox/internal/parser"
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strconv"
)

// PackageResult contains emitted files and their complete import dependencies.
// Keys in Files are original filenames. Callers retain control of filesystem IO.
type PackageResult struct {
	Files   map[string]Result
	Imports []string
}

// TransformPackage transforms selected source files deterministically. File
// selection (including build constraints) belongs to the project adapter.
func TransformPackage(ctx context.Context, files map[string][]byte, opts Options) (PackageResult, error) {
	out := PackageResult{Files: map[string]Result{}}
	names := make([]string, 0, len(files))
	for n := range files {
		names = append(names, n)
	}
	sort.Strings(names)
	bindings, e := PackageHookBindings(files)
	if e != nil {
		return out, e
	}
	imports := map[string]bool{}
	var pkg string
	for _, n := range names {
		if e := ctx.Err(); e != nil {
			return out, e
		}
		fileOpts := opts
		for _, names := range bindings {
			fileOpts.HookBindings = append(fileOpts.HookBindings, names...)
		}
		if opts.Identity != "" {
			fileOpts.Identity = opts.Identity + "/" + n
		}
		r, e := TransformFile(n, files[n], fileOpts)
		if e != nil {
			return out, e
		}
		f, e := parser.ParseFile(token.NewFileSet(), n, r.Source, parser.ImportsOnly)
		if e != nil {
			return out, e
		}
		if pkg != "" && pkg != f.Name.Name {
			return out, fmt.Errorf("mixed package names %s and %s", pkg, f.Name.Name)
		}
		pkg = f.Name.Name
		for _, i := range f.Imports {
			path, _ := strconv.Unquote(i.Path.Value)
			imports[path] = true
		}
		out.Files[n] = r
	}
	for n := range imports {
		out.Imports = append(out.Imports, n)
	}
	sort.Strings(out.Imports)
	return out, nil
}

// PackageHookBindings resolves package-level shadowing before individual files
// are lowered. Keys are package names, separating external test packages.
func PackageHookBindings(files map[string][]byte) (map[string][]string, error) {
	out := map[string][]string{}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, filename := range names {
		src := files[filename]
		f, e := gxparser.ParseFileWithBindings(token.NewFileSet(), filename, src, gxparser.GoX|gxparser.BareJSX|gxparser.SkipObjectResolution, filename, []string{"useState", "useRef", "useContext"})
		if e != nil {
			return nil, e
		}
		add := func(n string) {
			switch n {
			case "useState", "useRef", "useContext":
				out[f.Name.Name] = append(out[f.Name.Name], n)
			}
		}
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil {
					add(d.Name.Name)
				}
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.ValueSpec:
						for _, n := range s.Names {
							add(n.Name)
						}
					case *ast.TypeSpec:
						add(s.Name.Name)
					}
				}
			}
		}
	}
	for k := range out {
		sort.Strings(out[k])
	}
	return out, nil
}
