// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	"reflect"
	"strings"
	"sync"
	"text/template"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/snippet"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

// Postfix snippets are artificial methods that allow the user to
// compose common operations in an "argument oriented" fashion. For
// example, instead of "sort.Slice(someSlice, ...)" a user can expand
// "someSlice.sort!".

// postfixTmpl represents a postfix snippet completion candidate.
type postfixTmpl struct {
	// label is the completion candidate's label presented to the user.
	label string

	// details is passed along to the client as the candidate's details.
	details string

	// body is the template text. See postfixTmplArgs for details on the
	// facilities available to the template.
	body string

	tmpl *template.Template
}

// postfixTmplArgs are the template execution arguments available to
// the postfix snippet templates.
type postfixTmplArgs struct {
	// StmtOK is true if it is valid to replace the selector with a
	// statement. For example:
	//
	//    func foo() {
	//      bar.sort! // statement okay
	//
	//      someMethod(bar.sort!) // statement not okay
	//    }
	StmtOK bool

	// X is the textual SelectorExpr.X. For example, when completing
	// "foo.bar.print!", "X" is "foo.bar".
	X string

	// Obj is the types.Object of SelectorExpr.X, if any.
	Obj types.Object

	// Type is the type of "foo.bar" in "foo.bar.print!".
	Type types.Type

	scope          *types.Scope
	snip           snippet.Builder
	importIfNeeded func(pkgPath string, scope *types.Scope) (name string, edits []protocol.TextEdit, err error)
	edits          []protocol.TextEdit
	qf             types.Qualifier
	varNames       map[string]bool
}

var postfixTmpls = []postfixTmpl{{
	label:   "sort",
	details: "sort.Slice()",
	body: `{{if and (eq .Kind "slice") .StmtOK -}}
{{.Import "sort"}}.Slice({{.X}}, func({{.VarName nil "i"}}, {{.VarName nil "j"}} int) bool {
	{{.Cursor}}
})
{{- end}}`,
}, {
	label:   "last",
	details: "s[len(s)-1]",
	body: `{{if and (eq .Kind "slice") .Obj -}}
{{.X}}[len({{.X}})-1]
{{- end}}`,
}, {
	label:   "reverse",
	details: "reverse slice",
	body: `{{if and (eq .Kind "slice") .StmtOK -}}
{{$i := .VarName nil "i"}}{{$j := .VarName nil "j" -}}
for {{$i}}, {{$j}} := 0, len({{.X}})-1; {{$i}} < {{$j}}; {{$i}}, {{$j}} = {{$i}}+1, {{$j}}-1 {
	{{.X}}[{{$i}}], {{.X}}[{{$j}}] = {{.X}}[{{$j}}], {{.X}}[{{$i}}]
}
{{end}}`,
}, {
	label:   "range",
	details: "range over slice",
	body: `{{if and (eq .Kind "slice") .StmtOK -}}
for {{.VarName nil "i"}}, {{.VarName .ElemType "v"}} := range {{.X}} {
	{{.Cursor}}
}
{{- end}}`,
}, {
	label:   "append",
	details: "append and re-assign slice",
	body: `{{if and (eq .Kind "slice") .StmtOK .Obj -}}
{{.X}} = append({{.X}}, {{.Cursor}})
{{- end}}`,
}, {
	label:   "append",
	details: "append to slice",
	body: `{{if and (eq .Kind "slice") (not .StmtOK) -}}
append({{.X}}, {{.Cursor}})
{{- end}}`,
}, {
	label:   "copy",
	details: "duplicate slice",
	body: `{{if and (eq .Kind "slice") .StmtOK .Obj -}}
{{$v := (.VarName nil (printf "%sCopy" .X))}}{{$v}} := make([]{{.TypeName .ElemType}}, len({{.X}}))
copy({{$v}}, {{.X}})
{{end}}`,
}, {
	label:   "range",
	details: "range over map",
	body: `{{if and (eq .Kind "map") .StmtOK -}}
for {{.VarName .KeyType "k"}}, {{.VarName .ElemType "v"}} := range {{.X}} {
	{{.Cursor}}
}
{{- end}}`,
}, {
	label:   "clear",
	details: "clear map contents",
	body: `{{if and (eq .Kind "map") .StmtOK -}}
{{$k := (.VarName .KeyType "k")}}for {{$k}} := range {{.X}} {
	delete({{.X}}, {{$k}})
}
{{end}}`,
}, {
	label:   "keys",
	details: "create slice of keys",
	body: `{{if and (eq .Kind "map") .StmtOK -}}
{{$keysVar := (.VarName nil "keys")}}{{$keysVar}} := make([]{{.TypeName .KeyType}}, 0, len({{.X}}))
{{$k := (.VarName .KeyType "k")}}for {{$k}} := range {{.X}} {
	{{$keysVar}} = append({{$keysVar}}, {{$k}})
}
{{end}}`,
}, {
	label:   "range",
	details: "range over channel",
	body: `{{if and (eq .Kind "chan") .StmtOK -}}
for {{.VarName .ElemType "e"}} := range {{.X}} {
	{{.Cursor}}
}
{{- end}}`,
}, {
	label:   "var",
	details: "assign to variables",
	body: `{{if and (eq .Kind "tuple") .StmtOK -}}
{{$a := .}}{{range $i, $v := .Tuple}}{{if $i}}, {{end}}{{$a.VarName $v.Type $v.Name}}{{end}} := {{.X}}
{{- end}}`,
}, {
	label:   "var",
	details: "assign to variable",
	body: `{{if and (ne .Kind "tuple") .StmtOK -}}
{{.VarName .Type ""}} := {{.X}}
{{- end}}`,
}, {
	label:   "print",
	details: "print to stdout",
	body: `{{if and (ne .Kind "tuple") .StmtOK -}}
{{.Import "fmt"}}.Printf("{{.EscapeQuotes .X}}: %v\n", {{.X}})
{{- end}}`,
}, {
	label:   "print",
	details: "print to stdout",
	body: `{{if and (eq .Kind "tuple") .StmtOK -}}
{{.Import "fmt"}}.Println({{.X}})
{{- end}}`,
}, {
	label:   "split",
	details: "split string",
	body: `{{if (eq (.TypeName .Type) "string") -}}
{{.Import "strings"}}.Split({{.X}}, "{{.Cursor}}")
{{- end}}`,
}, {
	label:   "join",
	details: "join string slice",
	body: `{{if and (eq .Kind "slice") (eq (.TypeName .ElemType) "string") -}}
{{.Import "strings"}}.Join({{.X}}, "{{.Cursor}}")
{{- end}}`,
}}

// Cursor indicates where the client's cursor should end up after the
// snippet is done.
func (a *postfixTmplArgs) Cursor() string {
	a.snip.WriteFinalTabstop()
	return ""
}

// Import makes sure the package corresponding to path is imported,
// returning the identifier to use to refer to the package.
func (a *postfixTmplArgs) Import(path string) (string, error) {
	name, edits, err := a.importIfNeeded(path, a.scope)
	if err != nil {
		return "", fmt.Errorf("couldn't import %q: %w", path, err)
	}
	a.edits = append(a.edits, edits...)
	return name, nil
}

func (a *postfixTmplArgs) EscapeQuotes(v string) string {
	return strings.ReplaceAll(v, `"`, `\\"`)
}

// ElemType returns the Elem() type of xType, if applicable.
func (a *postfixTmplArgs) ElemType() types.Type {
	if e, _ := a.Type.(interface{ Elem() types.Type }); e != nil {
		return e.Elem()
	}
	return nil
}

// Kind returns the underlying kind of type, e.g. "slice", "struct",
// etc.
func (a *postfixTmplArgs) Kind() string {
	t := reflect.TypeOf(a.Type.Underlying())
	return strings.ToLower(strings.TrimPrefix(t.String(), "*types."))
}

// KeyType returns the type of X's key. KeyType panics if X is not a
// map.
func (a *postfixTmplArgs) KeyType() types.Type {
	return a.Type.Underlying().(*types.Map).Key()
}

// Tuple returns the tuple result vars if X is a call expression.
func (a *postfixTmplArgs) Tuple() []*types.Var {
	tuple, _ := a.Type.(*types.Tuple)
	if tuple == nil {
		return nil
	}

	typs := make([]*types.Var, 0, tuple.Len())
	for i := 0; i < tuple.Len(); i++ {
		typs = append(typs, tuple.At(i))
	}
	return typs
}

// TypeName returns the textual representation of type t.
func (a *postfixTmplArgs) TypeName(t types.Type) (string, error) {
	if t == nil || t == types.Typ[types.Invalid] {
		return "", fmt.Errorf("invalid type: %v", t)
	}
	return types.TypeString(t, a.qf), nil
}

// VarName returns a suitable variable name for the type t. If t
// implements the error interface, "err" is used. If t is not a named
// type then nonNamedDefault is used. Otherwise a name is made by
// abbreviating the type name. If the resultant name is already in
// scope, an integer is appended to make a unique name.
func (a *postfixTmplArgs) VarName(t types.Type, nonNamedDefault string) string {
	if t == nil {
		t = types.Typ[types.Invalid]
	}

	var name string
	if types.Implements(t, errorIntf) {
		name = "err"
	} else if _, isNamed := source.Deref(t).(*types.Named); !isNamed {
		name = nonNamedDefault
	}

	if name == "" {
		name = types.TypeString(t, func(p *types.Package) string {
			return ""
		})
		name = abbreviateTypeName(name)
	}

	if dot := strings.LastIndex(name, "."); dot > -1 {
		name = name[dot+1:]
	}

	uniqueName := name
	for i := 2; ; i++ {
		if s, _ := a.scope.LookupParent(uniqueName, token.NoPos); s == nil && !a.varNames[uniqueName] {
			break
		}
		uniqueName = fmt.Sprintf("%s%d", name, i)
	}

	a.varNames[uniqueName] = true

	return uniqueName
}

func (c *completer) addPostfixSnippetCandidates(ctx context.Context, sel *ast.SelectorExpr) {
	if !c.opts.postfix {
		return
	}

	initPostfixRules()

	if sel == nil || sel.Sel == nil {
		return
	}

	selType := c.pkg.GetTypesInfo().TypeOf(sel.X)
	if selType == nil {
		return
	}

	// Skip empty tuples since there is no value to operate on.
	if tuple, ok := selType.Underlying().(*types.Tuple); ok && tuple == nil {
		return
	}

	tokFile := c.snapshot.FileSet().File(c.pos)

	// Only replace sel with a statement if sel is already a statement.
	var stmtOK bool
	for i, n := range c.path {
		if n == sel && i < len(c.path)-1 {
			switch p := c.path[i+1].(type) {
			case *ast.ExprStmt:
				stmtOK = true
			case *ast.AssignStmt:
				// In cases like:
				//
				//   foo.<>
				//   bar = 123
				//
				// detect that "foo." makes up the entire statement since the
				// apparent selector spans lines.
				stmtOK = tokFile.Line(c.pos) < tokFile.Line(p.TokPos)
			}
			break
		}
	}

	scope := c.pkg.GetTypes().Scope().Innermost(c.pos)
	if scope == nil {
		return
	}

	// afterDot is the position after selector dot, e.g. "|" in
	// "foo.|print".
	afterDot := sel.Sel.Pos()

	// We must detect dangling selectors such as:
	//
	//    foo.<>
	//    bar
	//
	// and adjust afterDot so that we don't mistakenly delete the
	// newline thinking "bar" is part of our selector.
	if startLine := tokFile.Line(sel.Pos()); startLine != tokFile.Line(afterDot) {
		if tokFile.Line(c.pos) != startLine {
			return
		}
		afterDot = c.pos
	}

	for _, rule := range postfixTmpls {
		// When completing foo.print<>, "print" is naturally overwritten,
		// but we need to also remove "foo." so the snippet has a clean
		// slate.
		edits, err := c.editText(sel.Pos(), afterDot, "")
		if err != nil {
			event.Error(ctx, "error calculating postfix edits", err)
			return
		}

		tmplArgs := postfixTmplArgs{
			X:              source.FormatNode(c.snapshot.FileSet(), sel.X),
			StmtOK:         stmtOK,
			Obj:            exprObj(c.pkg.GetTypesInfo(), sel.X),
			Type:           selType,
			qf:             c.qf,
			importIfNeeded: c.importIfNeeded,
			scope:          scope,
			varNames:       make(map[string]bool),
		}

		// Feed the template straight into the snippet builder. This
		// allows templates to build snippets as they are executed.
		err = rule.tmpl.Execute(&tmplArgs.snip, &tmplArgs)
		if err != nil {
			event.Error(ctx, "error executing postfix template", err)
			continue
		}

		if strings.TrimSpace(tmplArgs.snip.String()) == "" {
			continue
		}

		score := c.matcher.Score(rule.label)
		if score <= 0 {
			continue
		}

		c.items = append(c.items, CompletionItem{
			Label:               rule.label + "!",
			Detail:              rule.details,
			Score:               float64(score) * 0.01,
			Kind:                protocol.SnippetCompletion,
			snippet:             &tmplArgs.snip,
			AdditionalTextEdits: append(edits, tmplArgs.edits...),
		})
	}
}

var postfixRulesOnce sync.Once

func initPostfixRules() {
	postfixRulesOnce.Do(func() {
		var idx int
		for _, rule := range postfixTmpls {
			var err error
			rule.tmpl, err = template.New("postfix_snippet").Parse(rule.body)
			if err != nil {
				log.Panicf("error parsing postfix snippet template: %v", err)
			}
			postfixTmpls[idx] = rule
			idx++
		}
		postfixTmpls = postfixTmpls[:idx]
	})
}

// importIfNeeded returns the package identifier and any necessary
// edits to import package pkgPath.
func (c *completer) importIfNeeded(pkgPath string, scope *types.Scope) (string, []protocol.TextEdit, error) {
	defaultName := imports.ImportPathToAssumedName(pkgPath)

	// Check if file already imports pkgPath.
	for _, s := range c.file.Imports {
		if source.ImportPath(s) == pkgPath {
			if s.Name == nil {
				return defaultName, nil, nil
			}
			if s.Name.Name != "_" {
				return s.Name.Name, nil, nil
			}
		}
	}

	// Give up if the package's name is already in use by another object.
	if _, obj := scope.LookupParent(defaultName, token.NoPos); obj != nil {
		return "", nil, fmt.Errorf("import name %q of %q already in use", defaultName, pkgPath)
	}

	edits, err := c.importEdits(&importInfo{
		importPath: pkgPath,
	})
	if err != nil {
		return "", nil, err
	}

	return defaultName, edits, nil
}
