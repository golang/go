package parser

import (
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
)

const runtimePath = "github.com/alfu32/go-shared/gox/goxview"

func ident(s string) *ast.Ident                       { return ast.NewIdent(s) }
func str(s string) ast.Expr                           { return &ast.BasicLit{Kind: token.STRING, Value: strconv.Quote(s)} }
func sel(x ast.Expr, s string) ast.Expr               { return &ast.SelectorExpr{X: x, Sel: ident(s)} }
func call(f ast.Expr, args ...ast.Expr) *ast.CallExpr { return &ast.CallExpr{Fun: f, Args: args} }
func expr(e ast.Expr) ast.Stmt                        { return &ast.ExprStmt{X: e} }
func ret(e ast.Expr) ast.Stmt                         { return &ast.ReturnStmt{Results: []ast.Expr{e}} }
func gxField(name string, t ast.Expr) *ast.Field {
	f := &ast.Field{Type: t}
	if name != "" {
		f.Names = []*ast.Ident{ident(name)}
	}
	return f
}
func function(result ast.Expr, body ...ast.Stmt) *ast.FuncLit {
	ft := &ast.FuncType{Params: &ast.FieldList{}}
	if result != nil {
		ft.Results = &ast.FieldList{List: []*ast.Field{gxField("", result)}}
	}
	return &ast.FuncLit{Type: ft, Body: &ast.BlockStmt{List: body}}
}
func variable(name string, t ast.Expr, v ast.Expr) ast.Stmt {
	spec := &ast.ValueSpec{Names: []*ast.Ident{ident(name)}, Type: t}
	if v != nil {
		spec.Values = []ast.Expr{v}
	}
	return &ast.DeclStmt{Decl: &ast.GenDecl{Tok: token.VAR, Specs: []ast.Spec{spec}}}
}
func assign(name string, v ast.Expr) ast.Stmt {
	return &ast.AssignStmt{Lhs: []ast.Expr{ident(name)}, Tok: token.ASSIGN, Rhs: []ast.Expr{v}}
}
func (p *parser) fresh(label string) string {
	for {
		p.serial++
		digest := sha256.Sum256([]byte(p.identity))
		s := fmt.Sprintf("gox_%x_%s_%d", digest[:8], label, p.serial)
		if !strings.Contains(string(p.source), s) {
			return s
		}
	}
}
func (p *parser) dep(path string) string {
	if n, ok := p.extraImports[path]; ok {
		return n
	}

	n := p.fresh("pkg")
	p.extraImports[path] = n
	return n
}
func (p *parser) runtime(name string) ast.Expr { return sel(ident(p.dep(runtimePath)), name) }
func (p *parser) finishGoX(f *ast.File) {
	if len(p.extraImports) == 0 {
		return
	}
	uniqueAST(reflect.ValueOf(f), map[uintptr]bool{})
	resolveFile(f, p.file, nil)
	for _, m := range p.markupUses {
		if n, ok := m.definition.(*ast.Ident); ok && n.Obj != nil {
			for _, d := range p.componentDefs {
				if n.Obj == d.name.Obj {
					provided := map[string]bool{}
					for _, a := range m.attrs {
						provided[a.name] = true
					}
					for name := range d.required {
						if !provided[name] {
							p.error(n.Pos(), "missing component property "+name)
						}
					}
				}
			}
		}
	}
	for _, c := range p.componentBodies {
		p.rewriteHooks(c.body, c.instance)
	}
	paths := []string{}
	for path := range p.extraImports {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		i := &ast.ImportSpec{Name: ident(p.extraImports[path]), Path: str(path).(*ast.BasicLit)}
		f.Imports = append(f.Imports, i)
		// Insert after existing imports so cgo preambles stay attached to import C.
		at := 0
		pos := f.Name.End()
		for j, d := range f.Decls {
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.IMPORT {
				at = j + 1
				pos = g.End()
			}
		}
		decl := &ast.GenDecl{TokPos: pos, Tok: token.IMPORT, Specs: []ast.Spec{i}}
		f.Decls = append(f.Decls, nil)
		copy(f.Decls[at+1:], f.Decls[at:])
		f.Decls[at] = decl
	}
	f.Decls = append(f.Decls, p.extraDecls...)
}
func (p *parser) enterExtension() func() {
	p.extensionDepth++
	if p.extensionDepth > 128 {
		panic(bailout{pos: p.pos, msg: "GoX nesting exceeds 128"})
	}
	return func() { p.extensionDepth-- }
}
func (p *parser) parseFormat() ast.Expr {
	defer p.enterExtension()()
	quote := rune(p.lit[0])
	start := p.pos
	name := p.fresh("text")
	body := []ast.Stmt{variable(name, ident("string"), nil)}
	holes := 0
	literal := ""
	for {
		part, boundary := p.scanner.FormatPart(quote)
		literal += part
		if part != "" {
			body = append(body, &ast.AssignStmt{Lhs: []ast.Expr{ident(name)}, Tok: token.ADD_ASSIGN, Rhs: []ast.Expr{str(part)}})
		}
		if boundary != '{' {
			break
		}
		holes++
		p.scanner.Semi(false)
		p.next()
		x := p.parseRhs()
		if p.tok == token.SEMICOLON {
			p.next()
		}
		if p.tok != token.RBRACE {
			p.errorExpected(p.pos, "} ending interpolation")
			break
		}
		slot := p.fresh("value")
		body = append(body, variable(slot, ident("any"), x))
		body = append(body, &ast.AssignStmt{Lhs: []ast.Expr{ident(name)}, Tok: token.ADD_ASSIGN, Rhs: []ast.Expr{call(ident(p.formatter()), ident(slot))}})
	}
	p.scanner.Semi(true)
	p.next()
	if holes == 0 {
		return &ast.BasicLit{ValuePos: start, Kind: token.STRING, Value: strconv.Quote(literal)}
	}
	body = append(body, ret(ident(name)))
	return call(function(ident("string"), body...))
}
func (p *parser) parseDo() ast.Expr {
	defer p.enterExtension()()
	p.next()
	t := p.parseType()
	body := p.parseBody()
	f := function(t)
	f.Body = body
	return call(f)
}
func (p *parser) parseLocalFunction() ast.Stmt {
	d := p.parseFuncDecl()
	if d.Recv != nil || d.Type.TypeParams != nil || d.Body == nil {
		p.error(d.Pos(), "local functions require a body and cannot have receivers or type parameters")
	}
	return &ast.BlockStmt{List: []ast.Stmt{variable(d.Name.Name, d.Type, nil), assign(d.Name.Name, &ast.FuncLit{Type: d.Type, Body: d.Body})}}
}
func (p *parser) rawExpect(c rune) {
	if p.scanner.Char() != c {
		p.error(p.scanner.Position(), fmt.Sprintf("expected %q in markup", c))
		panic(bailout{})
	}
	p.scanner.Advance()
}
func (p *parser) hole() ast.Expr {
	p.rawExpect('{')
	p.scanner.Semi(false)
	p.next()
	x := p.parseRhs()
	if p.tok == token.SEMICOLON {
		p.next()
	}
	if p.tok != token.RBRACE {
		p.errorExpected(p.pos, "} ending markup expression")
		panic(bailout{})
	}
	return x
}

type markup struct {
	tag        string
	definition ast.Expr
	attrs      []markupAttr
	children   []markupChild
}
type markupAttr struct {
	name  string
	value ast.Expr
}
type markupChild struct {
	node  *markup
	value ast.Expr
}

func (p *parser) readMarkup(openConsumed bool) *markup {
	defer p.enterExtension()()
	if !openConsumed {
		p.rawExpect('<')
	}
	m := &markup{tag: p.scanner.Name()}
	if m.tag == "" {
		p.error(p.scanner.Position(), "missing markup tag")
		panic(bailout{})
	}
	if strings.Contains(m.tag, ".") || m.tag[0] >= 'A' && m.tag[0] <= 'Z' {
		parts := strings.Split(m.tag, ".")
		m.definition = ident(parts[0])
		for _, s := range parts[1:] {
			m.definition = sel(m.definition, s)
		}
	}
	seen := map[string]bool{}
	for {
		p.scanner.Space()
		c := p.scanner.Char()
		if c == '/' {
			p.scanner.Advance()
			p.rawExpect('>')
			return m
		}
		if c == '>' {
			p.scanner.Advance()
			break
		}
		if c < 0 {
			p.error(p.scanner.Position(), "unterminated markup tag")
			panic(bailout{})
		}
		n := p.scanner.Name()
		if n == "" {
			p.error(p.scanner.Position(), "expected attribute")
			panic(bailout{})
		}
		if seen[n] {
			p.error(p.scanner.Position(), "duplicate attribute "+n)
		}
		seen[n] = true
		p.scanner.Space()
		p.rawExpect('=')
		p.scanner.Space()
		var v ast.Expr
		if p.scanner.Char() == '{' {
			v = p.hole()
		} else if p.scanner.Char() == '"' || p.scanner.Char() == '`' {
			v = &ast.BasicLit{Kind: token.STRING, Value: p.scanner.AttributeString()}
		} else {
			p.error(p.scanner.Position(), "attribute requires string or expression")
			panic(bailout{})
		}
		m.attrs = append(m.attrs, markupAttr{n, v})
	}
	for {
		switch p.scanner.Char() {
		case -1:
			p.error(p.scanner.Position(), "unterminated markup element")
			panic(bailout{})
		case '{':
			m.children = append(m.children, markupChild{value: p.hole()})
		case '<':
			p.scanner.Advance()
			if p.scanner.Char() == '/' {
				p.scanner.Advance()
				end := p.scanner.Name()
				if end != m.tag {
					p.error(p.scanner.Position(), "mismatched closing tag: "+end)
				}
				p.scanner.Space()
				p.rawExpect('>')
				return m
			}
			m.children = append(m.children, markupChild{node: p.readMarkup(true)})
		default:
			var b strings.Builder
			for c := p.scanner.Char(); c >= 0 && c != '{' && c != '<'; c = p.scanner.Char() {
				b.WriteRune(c)
				p.scanner.Advance()
			}
			m.children = append(m.children, markupChild{value: str(strings.ReplaceAll(b.String(), "\r\n", "\n"))})
		}
	}
}

// Markup evaluation and construction are separate. The inner template has
// explicit typed parameters; only the outer evaluator sees source bindings.
type templateBuilder struct {
	eval, build []ast.Stmt
	params      []*ast.Field
	args        []ast.Expr
}

func (p *parser) templateSlot(t *templateBuilder, typ ast.Expr, value ast.Expr) ast.Expr {
	name := p.fresh("slot")
	t.eval = append(t.eval, variable(name, typ, value))
	param := p.fresh("parameter")
	t.params = append(t.params, gxField(param, typ))
	t.args = append(t.args, ident(name))
	return ident(param)
}
func (p *parser) parseMarkup(wrapped bool) ast.Expr {
	defer p.enterExtension()()
	if wrapped {
		p.scanner.Space()
	}
	m := p.readMarkup(!wrapped)
	if wrapped {
		p.scanner.Space()
		p.rawExpect('"')
	}
	p.scanner.Semi(true)
	p.next()
	t := &templateBuilder{}
	node := p.lowerMarkup(m, t)
	template := function(p.runtime("Node"), append(t.build, ret(node))...)
	template.Type.Params.List = t.params
	t.eval = append(t.eval, ret(call(template, t.args...)))
	return call(function(p.runtime("Node"), t.eval...))
}
func (p *parser) lowerMarkup(m *markup, t *templateBuilder) ast.Expr {
	var def, templateDef ast.Expr
	if m.definition != nil {
		n := p.fresh("definition")
		t.eval = append(t.eval, variable(n, nil, m.definition))
		def = ident(n)
		templateDef = p.templateSlot(t, &ast.StarExpr{X: p.runtime("Definition")}, sel(def, "GoXDefinition"))
		p.markupUses = append(p.markupUses, m)
	}
	if def != nil {
		names := []string{}
		for _, a := range m.attrs {
			if a.name != "key" {
				names = append(names, a.name)
			}
		}
		sort.Strings(names)
		fields := []*ast.Field{}
		for _, n := range names {
			fields = append(fields, gxField("P_"+n, ident("bool")))
		}
		shape := &ast.CompositeLit{Type: &ast.StructType{Fields: &ast.FieldList{List: fields}}}
		t.eval = append(t.eval, expr(call(sel(def, "GoXValidate"), shape)))
	}
	attrs := []ast.Expr{}
	for _, a := range m.attrs {
		if def != nil && a.name != "key" {
			attrs = append(attrs, p.templateSlot(t, p.runtime("Attr"), call(sel(def, "GoXProp_"+a.name), a.value)))
		} else {
			v := p.templateSlot(t, ident("any"), a.value)
			attrs = append(attrs, &ast.CompositeLit{Type: p.runtime("Attr"), Elts: []ast.Expr{&ast.KeyValueExpr{Key: ident("Name"), Value: str(a.name)}, &ast.KeyValueExpr{Key: ident("Value"), Value: v}}})
		}
	}
	cs := p.fresh("children")
	t.build = append(t.build, variable(cs, &ast.ArrayType{Elt: p.runtime("Node")}, nil))
	for _, c := range m.children {
		var v ast.Expr
		if c.node != nil {
			v = p.lowerMarkup(c.node, t)
		} else {
			v = p.templateSlot(t, ident("any"), c.value)
		}
		app := call(ident("append"), ident(cs), call(p.runtime("Child"), v))
		app.Ellipsis = 1
		t.build = append(t.build, assign(cs, app))
	}
	if def != nil && len(m.children) > 0 {
		// Field presence enforces children support even for imported components.
		t.eval = append(t.eval, &ast.AssignStmt{Lhs: []ast.Expr{ident("_")}, Tok: token.ASSIGN, Rhs: []ast.Expr{sel(def, "GoXChildren")}})
	}
	as := &ast.CompositeLit{Type: &ast.ArrayType{Elt: p.runtime("Attr")}, Elts: attrs}
	if templateDef != nil {
		return call(p.runtime("Component"), templateDef, as, ident(cs))
	}
	return call(p.runtime("Element"), str(m.tag), as, ident(cs))
}
func (p *parser) parseComponent() ast.Decl {
	pos := p.pos
	p.next()
	name := p.parseIdent()
	params := p.parseParameters(false)
	for _, f := range params.List {
		if len(f.Names) == 0 {
			p.error(f.Pos(), "component parameters require names")
		}
		if _, ok := f.Type.(*ast.Ellipsis); ok {
			p.error(f.Pos(), "component parameters cannot be variadic")
		}
		for _, n := range f.Names {
			if n.Name == "key" || n.Name == "_" {
				p.error(n.Pos(), "reserved component parameter name "+n.Name)
			}
		}
	}
	old := p.instance
	p.instance = p.fresh("instance")
	instance := p.instance
	body := p.parseBody()
	p.expectSemi()
	p.instance = old
	attrs := p.fresh("attrs")
	children := p.fresh("children")
	prefix := []ast.Stmt{}
	for _, f := range params.List {
		for _, n := range f.Names {
			var v ast.Expr
			if n.Name == "children" {
				v = ident(children)
			} else {
				v = call(&ast.IndexExpr{X: p.runtime("Prop"), Index: f.Type}, ident(attrs), str(n.Name))
			}
			prefix = append(prefix, variable(n.Name, f.Type, v), &ast.AssignStmt{Lhs: []ast.Expr{ident("_")}, Tok: token.ASSIGN, Rhs: []ast.Expr{ident(n.Name)}})
		}
	}
	body.List = append(prefix, body.List...)
	p.componentBodies = append(p.componentBodies, componentBody{body, instance})
	render := function(p.runtime("Node"))
	render.Type.Params.List = []*ast.Field{gxField(instance, &ast.StarExpr{X: p.runtime("Instance")}), gxField(attrs, &ast.ArrayType{Elt: p.runtime("Attr")}), gxField(children, &ast.ArrayType{Elt: p.runtime("Node")})}
	render.Body = body
	id := fmt.Sprintf("%s:%s:%d", p.identity, name.Name, p.file.Offset(pos))
	core := call(p.runtime("Define"), str(id), render)
	fields := []*ast.Field{gxField("GoXDefinition", &ast.StarExpr{X: p.runtime("Definition")})}
	values := []ast.Expr{&ast.KeyValueExpr{Key: ident("GoXDefinition"), Value: core}}
	required := map[string]bool{}
	for _, f := range params.List {
		for _, n := range f.Names {
			if n.Name == "children" {
				fields = append(fields, gxField("GoXChildren", ident("bool")))
				values = append(values, &ast.KeyValueExpr{Key: ident("GoXChildren"), Value: ident("true")})
				continue
			}
			required[n.Name] = true
			arg := p.fresh("property")
			value := &ast.CompositeLit{Type: p.runtime("Attr"), Elts: []ast.Expr{&ast.KeyValueExpr{Key: ident("Name"), Value: str(n.Name)}, &ast.KeyValueExpr{Key: ident("Value"), Value: ident(arg)}}}
			fn := function(p.runtime("Attr"), ret(value))
			fn.Type.Params.List = []*ast.Field{gxField(arg, f.Type)}
			fields = append(fields, gxField("GoXProp_"+n.Name, fn.Type))
			values = append(values, &ast.KeyValueExpr{Key: ident("GoXProp_" + n.Name), Value: fn})
		}
	}
	names := []string{}
	for n := range required {
		names = append(names, n)
	}
	sort.Strings(names)
	shapeFields := []*ast.Field{}
	for _, n := range names {
		shapeFields = append(shapeFields, gxField("P_"+n, ident("bool")))
	}
	validate := function(nil)
	validate.Type.Params.List = []*ast.Field{gxField("", &ast.StructType{Fields: &ast.FieldList{List: shapeFields}})}
	fields = append(fields, gxField("GoXValidate", validate.Type))
	values = append(values, &ast.KeyValueExpr{Key: ident("GoXValidate"), Value: validate})
	value := &ast.CompositeLit{Type: &ast.StructType{Fields: &ast.FieldList{List: fields}}, Elts: values}
	p.componentDefs = append(p.componentDefs, componentDef{name, required})
	return &ast.GenDecl{Tok: token.VAR, Specs: []ast.Spec{&ast.ValueSpec{Names: []*ast.Ident{name}, Values: []ast.Expr{value}}}}

}
func (p *parser) rewriteHooks(body *ast.BlockStmt, instance string) {
	allowed := map[*ast.CallExpr]bool{}
	conditional := false
	for _, s := range body.List {
		switch s.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt, *ast.ReturnStmt, *ast.BranchStmt, *ast.LabeledStmt:
			conditional = true
		}
		if conditional {
			continue
		}
		if a, ok := s.(*ast.AssignStmt); ok {
			for _, x := range a.Rhs {
				if c, ok := x.(*ast.CallExpr); ok {
					allowed[c] = true
				}
			}
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		if b, ok := n.(*ast.BlockStmt); ok && b != body {
			for _, child := range p.componentBodies {
				if child.body == b {
					return false
				}
			}
		}
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		var name *ast.Ident
		switch f := c.Fun.(type) {
		case *ast.Ident:
			name = f
		case *ast.IndexExpr:
			name, _ = f.X.(*ast.Ident)
		}
		if name == nil || name.Obj != nil || slices.Contains(p.hookBindings, name.Name) {
			return true
		}
		mapped := map[string]string{"useState": "UseState", "useRef": "UseRef", "useContext": "UseContext"}[name.Name]
		if mapped == "" {
			return true
		}
		if !allowed[c] {
			p.error(c.Pos(), "hooks must be direct top-level component assignments")
			return false
		}
		p.hooks++
		args := []ast.Expr{ident(instance)}
		if mapped != "UseContext" {
			args = append(args, str(fmt.Sprint("hook", p.hooks)))
		}
		c.Args = append(args, c.Args...)
		if f, ok := c.Fun.(*ast.IndexExpr); ok {
			f.X = p.runtime(mapped)
		} else {
			c.Fun = p.runtime(mapped)
		}
		return true
	})
}

type componentDef struct {
	name     *ast.Ident
	required map[string]bool
}

type componentBody struct {
	body     *ast.BlockStmt
	instance string
}

func (p *parser) formatter() string {
	if p.formatterName != "" {
		return p.formatterName
	}
	name := p.fresh("format")
	p.formatterName = name
	v := p.fresh("operand")
	fn := function(ident("string"), ret(call(sel(ident(p.dep("fmt")), "Sprint"), ident(v))))
	fn.Type.Params.List = []*ast.Field{gxField(v, ident("any"))}
	p.extraDecls = append(p.extraDecls, &ast.FuncDecl{Name: ident(name), Type: fn.Type, Body: fn.Body})
	return name
}

func uniqueAST(v reflect.Value, seen map[uintptr]bool) {
	if !v.IsValid() {
		return
	}
	switch v.Kind() {
	case reflect.Interface:
		if !v.IsNil() {
			x := reflect.New(v.Elem().Type()).Elem()
			x.Set(v.Elem())
			uniqueAST(x, seen)
			v.Set(x)
		}
	case reflect.Pointer:
		if v.IsNil() {
			return
		}
		ptr := v.Pointer()
		if seen[ptr] && v.CanSet() {
			x := reflect.New(v.Type().Elem())
			x.Elem().Set(v.Elem())
			v.Set(x)
		}
		seen[v.Pointer()] = true
		uniqueAST(v.Elem(), seen)
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Field(i).CanSet() {
				uniqueAST(v.Field(i), seen)
			}
		}
	case reflect.Slice:
		if !v.IsNil() {
			x := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
			reflect.Copy(x, v)
			v.Set(x)
		}
		for i := 0; i < v.Len(); i++ {
			uniqueAST(v.Index(i), seen)
		}
	}
}
