// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strings"
	"testing"
)

func TestPrefixNames(t *testing.T) {
	src := "package main\n\nimport (\n\"comptime\"\n\"context\"\n\"encoding/json\"\n\"fmt\"\n\"iter\"\n\"logos/arena\"\n)\n\nfunc probeParse(probeInput string) (int, error) {\nreturn len(probeInput), nil\n}\n\nfunc probeGet(probeInput string) (int, error) {\nprobeVal := probeParse(probeInput)?\nreturn probeVal, nil\n}\n\nfunc probeFirstOr(probeOpt int?, probeDef int) int {\nreturn probeOpt ?? probeDef\n}\n\ntype ProbeA struct {\nProbeX int\n}\n\nfunc (probeRecv ProbeA) ProbeM() string {\nreturn fmt.Sprint(probeRecv.ProbeX)\n}\n\ntype ProbeB struct {\nProbeY string\n}\n\nfunc (probeRecv ProbeB) ProbeM() string {\nreturn probeRecv.ProbeY\n}\n\ntype ProbeU interface {\nProbeA | ProbeB\nProbeM() string\n}\n\nfunc probeWho(probeU ProbeU) string {\nreturn probeU.ProbeM()\n}\n\nfunc probePair() (int, error) {\nreturn 3, nil\n}\n\nconst probeEight = map[int]string{1: \"a\", 2: \"b\", 3: \"c\", 4: \"d\", 5: \"e\", 6: \"f\", 7: \"g\", 8: \"h\"}\n\nconst probeNine = map[int]string{1: \"a\", 2: \"b\", 3: \"c\", 4: \"d\", 5: \"e\", 6: \"f\", 7: \"g\", 8: \"h\", 9: \"i\"}\n\ntype ProbePair struct {\nProbeX int\nProbeY int\n}\n\nfunc probeNames[T any]() []string {\nvar probeOut []string\nconst for _, probeF := range comptime.Fields[T]() {\nprobeOut = append(probeOut, probeF.Name)\n}\nreturn probeOut\n}\n\ntype ProbeJS struct {\nProbeA int64 `json:\"probeA\"`\nProbeB string `json:\"probeB,omitempty\"`\n}\n\ntype ProbeBig struct {\nProbeF1 int64 `json:\"probeF1\"`\nProbeF2 string `json:\"probeF2\"`\nProbeF3 int64 `json:\"probeF3\"`\nProbeF4 string `json:\"probeF4\"`\nProbeF5 int64 `json:\"probeF5\"`\nProbeF6 string `json:\"probeF6\"`\nProbeF7 int64 `json:\"probeF7\"`\nProbeF8 string `json:\"probeF8\"`\nProbeF9 string `json:\"probeF9\"`\n}\n\ntype ProbeReq struct {\nProbeR *int64 `json:\"probeR\"`\n}\n\ntype probeNode struct {\nprobeV int\n}\n\nfunc probeLocals() []probeNode {\nprobeRegion := arena.Open()\nreturn arena.Append[probeNode](probeRegion, nil, probeNode{3})\n}\n\nvar probeCh = make(chan int, 8)\n\nfunc probeTask(probeN int) {\nprobeCh <- probeN + 41\n}\n\nfunc probeRun(probeCtx context.Context, probeIn int) error {\ngo group (probeCtx) {\ngo probeTask(probeIn)\n}\nreturn nil\n}\n\nfunc probeCount(probeN int) iter.Seq[int] {\nfor probeIdx := range probeN {\nyield probeIdx\n}\n}\n\ntype probeCloser struct {\nprobeErr error\n}\n\nfunc (probeC *probeCloser) Close() error {\nreturn probeC.probeErr\n}\n\nfunc probeReturnNil() (err error) {\nprobeC := &probeCloser{}\ndefer probeC.Close()?\nreturn nil\n}\n\ntype ProbeVec struct {\nProbeX int\nProbeY int\n}\n\ntype ProbeCarrier interface {\nProbePos ProbeVec\n}\n\ntype ProbeBot struct : ProbeCarrier {\nProbePos ProbeVec\n}\n\nfunc main() {\nfmt.Println(probeGet(\"hi\"))\nvar probeOpt int?\nfmt.Println(probeFirstOr(probeOpt, 7))\nfmt.Println(probeWho(ProbeA{1}))\nprobeT := probePair()\nprobeA, probeB := probeT\nfmt.Println(probeA, probeB)\nfmt.Println(probeEight[3] ?? \"none\")\nfmt.Println(probeNine[9] ?? \"none\")\nfmt.Println(probeNames[ProbePair]())\nprobeJ, _ := json.Marshal(ProbeJS{1, \"x\"})\nfmt.Println(string(probeJ))\nvar probeJU ProbeJS\n_ = json.Unmarshal(probeJ, &probeJU)\nprobeBin, _ := probeJU.MarshalBinary()\nvar probeBinBack ProbeJS\n_ = probeBinBack.UnmarshalBinary(probeBin)\nvar probeBig ProbeBig\nprobeBigJ, _ := json.Marshal(probeBig)\nfmt.Println(string(probeBigJ))\nvar probeBigBack ProbeBig\n_ = json.Unmarshal(probeBigJ, &probeBigBack)\nprobeBigBin, _ := probeBig.MarshalBinary()\nvar probeBigBinBack ProbeBig\n_ = probeBigBinBack.UnmarshalBinary(probeBigBin)\nprobeRVal := int64(7)\nvar probeReq ProbeReq = ProbeReq{ProbeR: &probeRVal}\nprobeReqJ, _ := json.Marshal(probeReq)\nfmt.Println(string(probeReqJ))\nvar probeReqBack ProbeReq = ProbeReq{ProbeR: &probeRVal}\n_ = json.Unmarshal(probeReqJ, &probeReqBack)\nfmt.Println(probeLocals())\n_ = probeRun(context.Background(), 1)\nfor probeV := range probeCount(3) {\nfmt.Println(probeV)\n}\nfmt.Println(probeReturnNil())\nvar probeBot ProbeBot\nprobeBot.ProbePos = ProbeVec{1, 2}\nfmt.Println(probeBot.ProbePos)\n}\n"
	out := lowerSnippetSrc(t, "prefixprobe.lgo", src)
	prefixCheckOutput(t, src, out)
	srcInterp := "package main\nimport \"fmt\"\nfunc main() {\nprobeName := \"world\"\nprobeN := 3\nprobeS := $\"hello {probeName} n={probeN}\"\nfmt.Println(probeS)\n}\n"
	outInterp := lowerSnippetSrc(t, "interp.lgo", srcInterp)
	prefixCheckOutput(t, srcInterp, outInterp)
	srcConstSwitch := "package main\nimport \"fmt\"\ntype ProbeSwitchScalar interface { ~float32 | float64 }\ntype ProbeKilos float32\nfunc probeSwitchKind[T ProbeSwitchScalar](probeV T) string {\nconst switch T {\ncase float32:\nreturn \"exact-f32\"\ncase ~float32:\nreturn \"tilde-f32\"\ncase float64:\nreturn \"exact-f64\"\n}\nreturn \"other\"\n}\nfunc main() {\nfmt.Println(probeSwitchKind(float64(3.75)))\nfmt.Println(probeSwitchKind(float32(7.5)))\nfmt.Println(probeSwitchKind(ProbeKilos(1.5)))\n}\n"
	outConstSwitch := lowerSnippetSrc(t, "constswitch.lgo", srcConstSwitch)
	prefixCheckOutput(t, srcConstSwitch, outConstSwitch)
	srcOptChain := "package main\nimport \"fmt\"\ntype ProbeUser struct { ProbeName string; ProbeAge int }\nfunc (probeRecv ProbeUser) Greet() string { return \"hi \" + probeRecv.ProbeName }\ntype ProbeTeam struct { Lead *ProbeUser? }\nfunc probeGreet(probeU *ProbeUser?) string { return probeU?.ProbeName ?? \"anon\" }\nfunc probeLeadGreet(probeT *ProbeTeam?) string { return probeT?.Lead?.ProbeName ?? \"anon\" }\nfunc main() {\nvar probeULocal *ProbeUser?\nfmt.Println(probeGreet(probeULocal))\nprobeULocal = &ProbeUser{ProbeName: \"ann\"}\nfmt.Println(probeGreet(probeULocal))\nfmt.Println(probeULocal?.ProbeAge ?? -1)\nvar probeTLocal *ProbeTeam? = &ProbeTeam{Lead: &ProbeUser{ProbeName: \"ann\"}}\nfmt.Println(probeLeadGreet(probeTLocal))\n}\n"
	outOptChain := lowerSnippetSrc(t, "optchain.lgo", srcOptChain)
	prefixCheckOutput(t, srcOptChain, outOptChain)
	srcChains := "package main\nimport (\"fmt\"; \"errors\")\nfunc probePos(probeX int) (bool, error) { if probeX < 0 { return false, errors.New(\"neg\") }; return probeX > 1, nil }\nfunc probeParse(probeX int) (int, error) { if probeX < 0 { return 0, errors.New(\"neg\") }; return 2*probeX, nil }\nfunc probeCountPos(probeItems []int) (int, error) { return probeItems.count(probeX => probePos(probeX)?), nil }\nfunc probeMapTry(probeItems []int) ([]int, error) { return probeItems.map(probeX => probeParse(probeX)?), nil }\nfunc probeFilterTry(probeItems []int) ([]int, error) { return probeItems.filter(probeX => probePos(probeX)?), nil }\nfunc main() {\nprobeItems := []int{3,1,4}\nfmt.Println(probeItems.take(2))\nfmt.Println(probeItems.drop(1))\nfmt.Println(probeItems.map(probeX => probeX*2))\nfmt.Println(probeItems.filter(probeX => probeX > 1))\nfmt.Println(probeCountPos(probeItems))\nfmt.Println(probeMapTry(probeItems))\nfmt.Println(probeFilterTry(probeItems))\n}\n"
	outChains := lowerSnippetSrc(t, "chains.lgo", srcChains)
	prefixCheckOutput(t, srcChains, outChains)
}

func TestPrefixNamesEjectOptTemp(t *testing.T) {
	src := "package main\nimport \"fmt\"\ntype optUser struct { optName string; optAge int }\nfunc (optRecv optUser) optGreet(optWord string) string { return optWord + optRecv.optName }\nfunc (optRecv *optUser) optLog() { fmt.Println(\"log \" + optRecv.optName) }\ntype optTeam struct { optLead *optUser? }\nfunc optField(optU *optUser?) string { return optU?.optName ?? \"anon\" }\nfunc optCall(optU *optUser?) string { return optU?.optGreet(\"hi \") ?? \"anon\" }\nfunc optVoid(optU *optUser?) { optU?.optLog() }\nfunc optChain(optT *optTeam?) string { return optT?.optLead?.optName ?? \"anon\" }\nfunc optIndex(optM map[string]*optUser) string { return optM[\"a\"]?.optName ?? \"anon\" }\nfunc main() {\nfmt.Println(optField(nil))\nfmt.Println(optCall(&optUser{optName: \"ann\", optAge: 7}))\noptVoid(&optUser{optName: \"ann\"})\nfmt.Println(optChain(&optTeam{optLead: &optUser{optName: \"ann\"}}))\nfmt.Println(optIndex(map[string]*optUser{\"a\": {optName: \"ann\"}}))\n}\n"
	out := lowerSnippetSrc(t, "opttemp.lgo", src)
	prefixCheckOutput(t, src, out)
	saved := base.LogosEject
	base.LogosEject = t.TempDir()
	t.Cleanup(func() { base.LogosEject = saved })
	ejectOut := lowerSnippetSrc(t, "opttempeject.lgo", src)
	prefixCheckOutput(t, src, ejectOut)
}

func prefixSnippetWords(src string) map[string]bool {
	m := map[string]bool{}
	re := regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)
	for _, w := range re.FindAllString(src, -1) {
		m[w] = true
	}
	return m
}

func prefixCheckOutput(t *testing.T, src, out string) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "prefix.go", out, 0)
	if err != nil {
		t.Fatal(err)
	}
	imports := prefixImports(f)
	words := prefixSnippetWords(src)
	var bad []string
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		bad = append(bad, prefixCheckFunc(fd, imports, words)...)
	}
	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, sp := range gd.Specs {
			bad = append(bad, prefixCheckSpec(sp, imports, words)...)
		}
	}
	if len(bad) > 0 {
		t.Fatalf("%d compiler-made names lack prefix: %s", len(bad), strings.Join(bad, ", "))
	}
	prefixCheckUses(t, src, out)
}

func prefixDeclared(f *ast.File) map[string]bool {
	declared := map[string]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.FuncDecl:
			declared[v.Name.Name] = true
			prefixAddFieldNames(declared, v.Recv)
			prefixAddFieldNames(declared, v.Type.Params)
			prefixAddFieldNames(declared, v.Type.Results)
			prefixAddFieldNames(declared, v.Type.TypeParams)
		case *ast.Field:
			for _, nm := range v.Names {
				declared[nm.Name] = true
			}
		case *ast.TypeSpec:
			declared[v.Name.Name] = true
		case *ast.ValueSpec:
			for _, nm := range v.Names {
				declared[nm.Name] = true
			}
		case *ast.AssignStmt:
			if v.Tok == token.DEFINE {
				for _, e := range v.Lhs {
					if id, ok := e.(*ast.Ident); ok {
						declared[id.Name] = true
					}
				}
			}
		case *ast.RangeStmt:
			prefixAddExprName(declared, v.Key)
			prefixAddExprName(declared, v.Value)
		case *ast.LabeledStmt:
			declared[v.Label.Name] = true
		case *ast.ImportSpec:
			if v.Name != nil {
				declared[v.Name.Name] = true
			}
		}
		return true
	})
	return declared
}

func prefixAddFieldNames(declared map[string]bool, list *ast.FieldList) {
	if list == nil {
		return
	}
	for _, fld := range list.List {
		for _, nm := range fld.Names {
			declared[nm.Name] = true
		}
	}
}

func prefixAddExprName(declared map[string]bool, e ast.Expr) {
	if id, ok := e.(*ast.Ident); ok {
		declared[id.Name] = true
	}
}

func prefixUseSkipped(stack []ast.Node, id *ast.Ident) bool {
	if len(stack) == 0 {
		return false
	}
	switch p := stack[len(stack)-1].(type) {
	case *ast.SelectorExpr:
		return p.Sel == id
	case *ast.KeyValueExpr:
		if k, ok := p.Key.(*ast.Ident); ok && k == id {
			return true
		}
	}
	return false
}

func prefixCheckUses(t *testing.T, src, out string) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "prefixuse.go", out, 0)
	if err != nil {
		t.Fatal(err)
	}
	imports := prefixImports(f)
	words := prefixSnippetWords(src)
	declared := prefixDeclared(f)
	var bad []string
	var stack []ast.Node
	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if id, ok := n.(*ast.Ident); ok {
			if !prefixAllowed(id.Name, imports) && !words[id.Name] && !declared[id.Name] && !prefixUseSkipped(stack, id) {
				bad = append(bad, id.Name)
			}
		}
		stack = append(stack, n)
		return true
	})
	if len(bad) > 0 {
		t.Fatalf("%d undeclared names in eject output: %s", len(bad), strings.Join(bad, ", "))
	}
}

func prefixImports(f *ast.File) map[string]bool {
	m := map[string]bool{}
	for _, im := range f.Imports {
		if im.Name != nil {
			if im.Name.Name == "_" || im.Name.Name == "." {
				continue
			}
			m[im.Name.Name] = true
			continue
		}
		s := strings.Trim(im.Path.Value, "\"")
		_, after, found := strings.Cut(s, "/")
		if found {
			s = after
			if i := strings.LastIndex(s, "/"); i >= 0 {
				s = s[i+1:]
			}
		}
		if s != "" {
			m[s] = true
		}
	}
	return m
}

func prefixIsGen(name string) bool {
	return strings.HasPrefix(name, "_logos_") || strings.Contains(name, "_logosct")
}

func prefixIsProp(name string) bool {
	return strings.HasPrefix(name, "Logos_Get") || strings.HasPrefix(name, "Logos_Set")
}

func prefixAllowed(name string, imports map[string]bool) bool {
	if name == "_" {
		return true
	}
	if imports[name] {
		return true
	}
	switch name {
	case "true", "false", "iota", "nil", "bool", "byte", "complex64", "complex128", "error", "float32", "float64", "int", "int8", "int16", "int32", "int64", "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "any", "comparable", "append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len", "make", "max", "min", "new", "panic", "print", "println", "real", "recover", "MarshalJSON", "UnmarshalJSON", "MarshalBinary", "AppendBinary", "UnmarshalBinary":
		return true
	}
	if prefixIsGen(name) {
		return true
	}
	if prefixIsProp(name) {
		return true
	}
	return false
}

func prefixCheckFunc(fd *ast.FuncDecl, imports map[string]bool, words map[string]bool) []string {
	if strings.Contains(fd.Name.Name, "_logosct") {
		return nil
	}
	if prefixIsProp(fd.Name.Name) {
		return nil
	}
	var bad []string
	if !prefixAllowed(fd.Name.Name, imports) && !words[fd.Name.Name] {
		bad = append(bad, fd.Name.Name)
	}
	if fd.Recv != nil {
		for _, fl := range fd.Recv.List {
			for _, nm := range fl.Names {
				if !prefixAllowed(nm.Name, imports) && !words[nm.Name] {
					bad = append(bad, nm.Name)
				}
			}
		}
	}
	if fd.Type.Params != nil {
		for _, fl := range fd.Type.Params.List {
			for _, nm := range fl.Names {
				if !prefixAllowed(nm.Name, imports) && !words[nm.Name] {
					bad = append(bad, nm.Name)
				}
			}
		}
	}
	if fd.Type.Results != nil {
		for _, fl := range fd.Type.Results.List {
			for _, nm := range fl.Names {
				if !prefixAllowed(nm.Name, imports) && !words[nm.Name] {
					bad = append(bad, nm.Name)
				}
			}
		}
	}
	if fd.Body == nil {
		return bad
	}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.AssignStmt:
			if v.Tok.String() == ":=" {
				for _, e := range v.Lhs {
					id, ok := e.(*ast.Ident)
					if ok && !prefixAllowed(id.Name, imports) && !words[id.Name] {
						bad = append(bad, id.Name)
					}
				}
			}
		case *ast.ValueSpec:
			for _, nm := range v.Names {
				if !prefixAllowed(nm.Name, imports) && !words[nm.Name] {
					bad = append(bad, nm.Name)
				}
			}
		case *ast.RangeStmt:
			if id, ok := v.Key.(*ast.Ident); ok && id.Name != "_" && !prefixAllowed(id.Name, imports) && !words[id.Name] {
				bad = append(bad, id.Name)
			}
			if id, ok := v.Value.(*ast.Ident); ok && id.Name != "_" && !prefixAllowed(id.Name, imports) && !words[id.Name] {
				bad = append(bad, id.Name)
			}
		case *ast.LabeledStmt:
			if !prefixAllowed(v.Label.Name, imports) && !words[v.Label.Name] {
				bad = append(bad, v.Label.Name)
			}
		}
		return true
	})
	return bad
}

func prefixCheckGenFields(ts *ast.TypeSpec) []string {
	st, ok := ts.Type.(*ast.StructType)
	if !ok {
		return nil
	}
	var bad []string
	for _, fl := range st.Fields.List {
		for _, nm := range fl.Names {
			if !prefixIsGen(nm.Name) && !prefixIsProp(nm.Name) {
				bad = append(bad, nm.Name)
			}
		}
	}
	return bad
}

func prefixCheckSpec(sp ast.Spec, imports, words map[string]bool) []string {
	var names []*ast.Ident
	var bad []string
	switch s := sp.(type) {
	case *ast.ValueSpec:
		names = s.Names
	case *ast.TypeSpec:
		names = []*ast.Ident{s.Name}
		if prefixIsGen(s.Name.Name) {
			bad = prefixCheckGenFields(s)
		}
	}
	for _, nm := range names {
		if !prefixAllowed(nm.Name, imports) && !words[nm.Name] {
			bad = append(bad, nm.Name)
		}
	}
	return bad
}
