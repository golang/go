package test

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/internal/objabi"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

func validatePanicOnPatterns(pkgs []*load.Package) {
	patterns := parseCommaList(testPanicOn)
	if len(patterns) == 0 {
		return
	}

	rootPrefixes := make(map[string]bool, len(pkgs))
	pkgByPrefix := make(map[string]*load.Package)

	var walk func(*load.Package)
	walk = func(p *load.Package) {
		if p == nil || p.ImportPath == "" {
			return
		}
		prefix := objabi.PathToPrefix(p.ImportPath)
		if _, ok := pkgByPrefix[prefix]; ok {
			return
		}
		pkgByPrefix[prefix] = p
		for _, imp := range p.Internal.Imports {
			walk(imp)
		}
	}
	for _, p := range pkgs {
		if p != nil && p.ImportPath != "" {
			rootPrefixes[objabi.PathToPrefix(p.ImportPath)] = true
		}
		walk(p)
	}

	funcCache := make(map[string][]panicOnFunc)

	for _, pattern := range patterns {
		if !strings.Contains(pattern, ".") {
			base.Fatalf("invalid -panic-on pattern %q: expected a fully qualified symbol like %q", pattern, "example.com/pkg.Func")
		}

		prefix, pkg := resolvePanicOnPackagePrefix(pattern, pkgByPrefix)
		if pkg == nil {
			base.Fatalf("invalid -panic-on pattern %q: package not found in the current build", pattern)
		}

		funcs, ok := funcCache[prefix]
		if !ok {
			var err error
			funcs, err = listPanicOnFuncs(pkg, prefix, rootPrefixes[prefix])
			if err != nil {
				base.Fatalf("failed to validate -panic-on=%q: %v", pattern, err)
			}
			funcCache[prefix] = funcs
		}

		matched := false
		matchPrefix := pattern
		prefixMatch := strings.HasSuffix(pattern, "*")
		if prefixMatch {
			matchPrefix = strings.TrimSuffix(pattern, "*")
		}
		for _, fn := range funcs {
			if prefixMatch {
				if !strings.HasPrefix(fn.sym, matchPrefix) {
					continue
				}
			} else if fn.sym != pattern {
				continue
			}

			matched = true
			if fn.params == "" {
				_, _ = os.Stderr.WriteString("detected function: " + fn.sym + "()\n")
			} else {
				_, _ = os.Stderr.WriteString("detected function: " + fn.sym + "(" + fn.params + ")\n")
			}
		}

		if !matched {
			base.Fatalf("invalid -panic-on pattern %q: no matching function or method found", pattern)
		}
	}
}

type panicOnFunc struct {
	sym    string
	params string
}

func resolvePanicOnPackagePrefix(pattern string, pkgByPrefix map[string]*load.Package) (string, *load.Package) {
	candidate := pattern
	for {
		i := strings.LastIndex(candidate, ".")
		if i < 0 {
			return "", nil
		}
		candidate = candidate[:i]
		if p, ok := pkgByPrefix[candidate]; ok {
			return candidate, p
		}
	}
}

func listPanicOnFuncs(pkg *load.Package, pkgPrefix string, includeTests bool) ([]panicOnFunc, error) {
	files := append([]string{}, pkg.GoFiles...)
	files = append(files, pkg.CgoFiles...)
	if includeTests {
		files = append(files, pkg.TestGoFiles...)
	}

	if pkg.Dir == "" {
		return nil, fmt.Errorf("package %q has no directory", pkg.ImportPath)
	}

	fset := token.NewFileSet()
	out := make([]panicOnFunc, 0, 32)

	for _, file := range files {
		path := file
		if !filepath.IsAbs(path) {
			path = filepath.Join(pkg.Dir, path)
		}
		f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Name == nil {
				continue
			}

			name := fd.Name.Name
			if fd.Recv != nil && len(fd.Recv.List) > 0 {
				recv := exprString(fset, fd.Recv.List[0].Type)
				name = "(" + recv + ")." + name
			}

			out = append(out, panicOnFunc{
				sym:    pkgPrefix + "." + name,
				params: fieldListString(fset, fd.Type.Params),
			})
		}
	}

	return out, nil
}

func exprString(fset *token.FileSet, x ast.Expr) string {
	var buf bytes.Buffer
	_ = printer.Fprint(&buf, fset, x)
	return buf.String()
}

func fieldListString(fset *token.FileSet, fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	parts := make([]string, 0, len(fl.List))
	for _, f := range fl.List {
		var buf bytes.Buffer
		_ = printer.Fprint(&buf, fset, f)
		parts = append(parts, buf.String())
	}
	return strings.Join(parts, ", ")
}

func parseCommaList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
