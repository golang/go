// The pkglemma package is a demonstration and test of the package lemma
// mechanism.
//
// The output of the pkglemma analysis is a set of key/values pairs
// gathered from the analyzed package and its imported dependencies.
// Each key/value pair comes from a top-level constant declaration
// whose name starts with "_".  For example:
//
//      package p
//
// 	const _greeting  = "hello"
// 	const _audience  = "world"
//
// the pkglemma analysis output for package p would be:
//
//   {"greeting": "hello", "audience": "world"}.
//
// In addition, the analysis reports a finding at each import
// showing which key/value pairs it contributes.
package pkglemma

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var Analysis = &analysis.Analysis{
	Name:       "pkglemma",
	Doc:        "gather name/value pairs from constant declarations",
	Run:        run,
	LemmaTypes: []reflect.Type{reflect.TypeOf(new(note))},
	OutputType: reflect.TypeOf(map[string]string{}),
}

// A note is a package-level lemma that records
// key/value pairs accumulated from constant
// declarations in this package and its dependencies.
type note struct {
	M map[string]string
}

func (*note) IsLemma() {}

func run(unit *analysis.Unit) error {
	m := make(map[string]string)

	// At each import, print the lemma from the imported
	// package and accumulate its information into m.
	doImport := func(spec *ast.ImportSpec) {
		pkg := unit.Info.Defs[spec.Name].(*types.PkgName).Imported()
		var lemma note
		if unit.PackageLemma(pkg, &lemma) {
			var lines []string
			for k, v := range lemma.M {
				m[k] = v
				lines = append(lines, fmt.Sprintf("%s=%s", k, v))
			}
			sort.Strings(lines)
			unit.Findingf(spec.Pos(), "%s", strings.Join(lines, " "))
		}
	}

	// At each "const _name = value", add a fact into m.
	doConst := func(spec *ast.ValueSpec) {
		if len(spec.Names) == len(spec.Values) {
			for i := range spec.Names {
				name := spec.Names[i].Name
				if strings.HasPrefix(name, "_") {
					m[name[1:]] = unit.Info.Types[spec.Values[i]].Value.String()
				}
			}
		}
	}

	for _, f := range unit.Syntax {
		for _, decl := range f.Decls {
			if decl, ok := decl.(*ast.GenDecl); ok {
				for _, spec := range decl.Specs {
					switch decl.Tok {
					case token.IMPORT:
						doImport(spec.(*ast.ImportSpec))
					case token.CONST:
						doConst(spec.(*ast.ValueSpec))
					}
				}
			}
		}
	}

	unit.Output = m

	unit.SetPackageLemma(&note{m})

	return nil
}
