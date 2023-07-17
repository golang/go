// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"bytes"
	"fmt"
	"log"
	"sort"
	"strings"
)

var (
	// tsclient.go has 3 sections
	cdecls = make(sortedMap[string])
	ccases = make(sortedMap[string])
	cfuncs = make(sortedMap[string])
	// tsserver.go has 3 sections
	sdecls = make(sortedMap[string])
	scases = make(sortedMap[string])
	sfuncs = make(sortedMap[string])
	// tsprotocol.go has 2 sections
	types  = make(sortedMap[string])
	consts = make(sortedMap[string])
	// tsjson has 1 section
	jsons = make(sortedMap[string])
)

func generateOutput(model Model) {
	for _, r := range model.Requests {
		genDecl(r.Method, r.Params, r.Result, r.Direction)
		genCase(r.Method, r.Params, r.Result, r.Direction)
		genFunc(r.Method, r.Params, r.Result, r.Direction, false)
	}
	for _, n := range model.Notifications {
		if n.Method == "$/cancelRequest" {
			continue // handled internally by jsonrpc2
		}
		genDecl(n.Method, n.Params, nil, n.Direction)
		genCase(n.Method, n.Params, nil, n.Direction)
		genFunc(n.Method, n.Params, nil, n.Direction, true)
	}
	genStructs(model)
	genAliases(model)
	genGenTypes() // generate the unnamed types
	genConsts(model)
	genMarshal()
}

func genDecl(method string, param, result *Type, dir string) {
	fname := methodName(method)
	p := ""
	if notNil(param) {
		p = ", *" + goplsName(param)
	}
	ret := "error"
	if notNil(result) {
		tp := goplsName(result)
		if !hasNilValue(tp) {
			tp = "*" + tp
		}
		ret = fmt.Sprintf("(%s, error)", tp)
	}
	// special gopls compatibility case (PJW: still needed?)
	switch method {
	case "workspace/configuration":
		// was And_Param_workspace_configuration, but the type substitution doesn't work,
		// as ParamConfiguration is embedded in And_Param_workspace_configuration
		p = ", *ParamConfiguration"
		ret = "([]LSPAny, error)"
	}
	msg := fmt.Sprintf("\t%s(context.Context%s) %s // %s\n", fname, p, ret, method)
	switch dir {
	case "clientToServer":
		sdecls[method] = msg
	case "serverToClient":
		cdecls[method] = msg
	case "both":
		sdecls[method] = msg
		cdecls[method] = msg
	default:
		log.Fatalf("impossible direction %q", dir)
	}
}

func genCase(method string, param, result *Type, dir string) {
	out := new(bytes.Buffer)
	fmt.Fprintf(out, "\tcase %q:\n", method)
	var p string
	fname := methodName(method)
	if notNil(param) {
		nm := goplsName(param)
		if method == "workspace/configuration" { // gopls compatibility
			// was And_Param_workspace_configuration, which contains ParamConfiguration
			// so renaming the type leads to circular definitions
			nm = "ParamConfiguration" // gopls compatibility
		}
		fmt.Fprintf(out, "\t\tvar params %s\n", nm)
		fmt.Fprintf(out, "\t\tif err := json.Unmarshal(r.Params(), &params); err != nil {\n")
		fmt.Fprintf(out, "\t\t\treturn true, sendParseError(ctx, reply, err)\n\t\t}\n")
		p = ", &params"
	}
	if notNil(result) {
		fmt.Fprintf(out, "\t\tresp, err := %%s.%s(ctx%s)\n", fname, p)
		out.WriteString("\t\tif err != nil {\n")
		out.WriteString("\t\t\treturn true, reply(ctx, nil, err)\n")
		out.WriteString("\t\t}\n")
		out.WriteString("\t\treturn true, reply(ctx, resp, nil)\n")
	} else {
		fmt.Fprintf(out, "\t\terr := %%s.%s(ctx%s)\n", fname, p)
		out.WriteString("\t\treturn true, reply(ctx, nil, err)\n")
	}
	msg := out.String()
	switch dir {
	case "clientToServer":
		scases[method] = fmt.Sprintf(msg, "server")
	case "serverToClient":
		ccases[method] = fmt.Sprintf(msg, "client")
	case "both":
		scases[method] = fmt.Sprintf(msg, "server")
		ccases[method] = fmt.Sprintf(msg, "client")
	default:
		log.Fatalf("impossible direction %q", dir)
	}
}

func genFunc(method string, param, result *Type, dir string, isnotify bool) {
	out := new(bytes.Buffer)
	var p, r string
	var goResult string
	if notNil(param) {
		p = ", params *" + goplsName(param)
	}
	if notNil(result) {
		goResult = goplsName(result)
		if !hasNilValue(goResult) {
			goResult = "*" + goResult
		}
		r = fmt.Sprintf("(%s, error)", goResult)
	} else {
		r = "error"
	}
	// special gopls compatibility case
	switch method {
	case "workspace/configuration":
		// was And_Param_workspace_configuration, but the type substitution doesn't work,
		// as ParamConfiguration is embedded in And_Param_workspace_configuration
		p = ", params *ParamConfiguration"
		r = "([]LSPAny, error)"
		goResult = "[]LSPAny"
	}
	fname := methodName(method)
	fmt.Fprintf(out, "func (s *%%sDispatcher) %s(ctx context.Context%s) %s {\n",
		fname, p, r)

	if !notNil(result) {
		if isnotify {
			if notNil(param) {
				fmt.Fprintf(out, "\treturn s.sender.Notify(ctx, %q, params)\n", method)
			} else {
				fmt.Fprintf(out, "\treturn s.sender.Notify(ctx, %q, nil)\n", method)
			}
		} else {
			if notNil(param) {
				fmt.Fprintf(out, "\treturn s.sender.Call(ctx, %q, params, nil)\n", method)
			} else {
				fmt.Fprintf(out, "\treturn s.sender.Call(ctx, %q, nil, nil)\n", method)
			}
		}
	} else {
		fmt.Fprintf(out, "\tvar result %s\n", goResult)
		if isnotify {
			if notNil(param) {
				fmt.Fprintf(out, "\ts.sender.Notify(ctx, %q, params)\n", method)
			} else {
				fmt.Fprintf(out, "\t\tif err := s.sender.Notify(ctx, %q, nil); err != nil {\n", method)
			}
		} else {
			if notNil(param) {
				fmt.Fprintf(out, "\t\tif err := s.sender.Call(ctx, %q, params, &result); err != nil {\n", method)
			} else {
				fmt.Fprintf(out, "\t\tif err := s.sender.Call(ctx, %q, nil, &result); err != nil {\n", method)
			}
		}
		fmt.Fprintf(out, "\t\treturn nil, err\n\t}\n\treturn result, nil\n")
	}
	out.WriteString("}\n")
	msg := out.String()
	switch dir {
	case "clientToServer":
		sfuncs[method] = fmt.Sprintf(msg, "server")
	case "serverToClient":
		cfuncs[method] = fmt.Sprintf(msg, "client")
	case "both":
		sfuncs[method] = fmt.Sprintf(msg, "server")
		cfuncs[method] = fmt.Sprintf(msg, "client")
	default:
		log.Fatalf("impossible direction %q", dir)
	}
}

func genStructs(model Model) {
	structures := make(map[string]*Structure) // for expanding Extends
	for _, s := range model.Structures {
		structures[s.Name] = s
	}
	for _, s := range model.Structures {
		out := new(bytes.Buffer)
		generateDoc(out, s.Documentation)
		nm := goName(s.Name)
		if nm == "string" { // an unacceptable strut name
			// a weird case, and needed only so the generated code contains the old gopls code
			nm = "DocumentDiagnosticParams"
		}
		fmt.Fprintf(out, "type %s struct {%s\n", nm, linex(s.Line))
		// for gpls compatibilitye, embed most extensions, but expand the rest some day
		props := append([]NameType{}, s.Properties...)
		if s.Name == "SymbolInformation" { // but expand this one
			for _, ex := range s.Extends {
				fmt.Fprintf(out, "\t// extends %s\n", ex.Name)
				props = append(props, structures[ex.Name].Properties...)
			}
			genProps(out, props, nm)
		} else {
			genProps(out, props, nm)
			for _, ex := range s.Extends {
				fmt.Fprintf(out, "\t%s\n", goName(ex.Name))
			}
		}
		for _, ex := range s.Mixins {
			fmt.Fprintf(out, "\t%s\n", goName(ex.Name))
		}
		out.WriteString("}\n")
		types[nm] = out.String()
	}
	// base types
	types["DocumentURI"] = "type DocumentURI string\n"
	types["URI"] = "type URI = string\n"

	types["LSPAny"] = "type LSPAny = interface{}\n"
	// A special case, the only previously existing Or type
	types["DocumentDiagnosticReport"] = "type DocumentDiagnosticReport = Or_DocumentDiagnosticReport // (alias) line 13909\n"

}

func genProps(out *bytes.Buffer, props []NameType, name string) {
	for _, p := range props {
		tp := goplsName(p.Type)
		if newNm, ok := renameProp[prop{name, p.Name}]; ok {
			usedRenameProp[prop{name, p.Name}] = true
			if tp == newNm {
				log.Printf("renameProp useless {%q, %q} for %s", name, p.Name, tp)
			}
			tp = newNm
		}
		// it's a pointer if it is optional, or for gopls compatibility
		opt, star := propStar(name, p, tp)
		json := fmt.Sprintf(" `json:\"%s%s\"`", p.Name, opt)
		generateDoc(out, p.Documentation)
		fmt.Fprintf(out, "\t%s %s%s %s\n", goName(p.Name), star, tp, json)
	}
}

func genAliases(model Model) {
	for _, ta := range model.TypeAliases {
		out := new(bytes.Buffer)
		generateDoc(out, ta.Documentation)
		nm := goName(ta.Name)
		if nm != ta.Name {
			continue // renamed the type, e.g., "DocumentDiagnosticReport", an or-type to "string"
		}
		tp := goplsName(ta.Type)
		fmt.Fprintf(out, "type %s = %s // (alias) line %d\n", nm, tp, ta.Line)
		types[nm] = out.String()
	}
}

func genGenTypes() {
	for _, nt := range genTypes {
		out := new(bytes.Buffer)
		nm := goplsName(nt.typ)
		switch nt.kind {
		case "literal":
			fmt.Fprintf(out, "// created for Literal (%s)\n", nt.name)
			fmt.Fprintf(out, "type %s struct {%s\n", nm, linex(nt.line+1))
			genProps(out, nt.properties, nt.name) // systematic name, not gopls name; is this a good choice?
		case "or":
			if !strings.HasPrefix(nm, "Or") {
				// It was replaced by a narrower type defined elsewhere
				continue
			}
			names := []string{}
			for _, t := range nt.items {
				if notNil(t) {
					names = append(names, goplsName(t))
				}
			}
			sort.Strings(names)
			fmt.Fprintf(out, "// created for Or %v\n", names)
			fmt.Fprintf(out, "type %s struct {%s\n", nm, linex(nt.line+1))
			fmt.Fprintf(out, "\tValue interface{} `json:\"value\"`\n")
		case "and":
			fmt.Fprintf(out, "// created for And\n")
			fmt.Fprintf(out, "type %s struct {%s\n", nm, linex(nt.line+1))
			for _, x := range nt.items {
				nm := goplsName(x)
				fmt.Fprintf(out, "\t%s\n", nm)
			}
		case "tuple": // there's only this one
			nt.name = "UIntCommaUInt"
			fmt.Fprintf(out, "//created for Tuple\ntype %s struct {%s\n", nm, linex(nt.line+1))
			fmt.Fprintf(out, "\tFld0 uint32 `json:\"fld0\"`\n")
			fmt.Fprintf(out, "\tFld1 uint32 `json:\"fld1\"`\n")
		default:
			log.Fatalf("%s not handled", nt.kind)
		}
		out.WriteString("}\n")
		types[nm] = out.String()
	}
}
func genConsts(model Model) {
	for _, e := range model.Enumerations {
		out := new(bytes.Buffer)
		generateDoc(out, e.Documentation)
		tp := goplsName(e.Type)
		nm := goName(e.Name)
		fmt.Fprintf(out, "type %s %s%s\n", nm, tp, linex(e.Line))
		types[nm] = out.String()
		vals := new(bytes.Buffer)
		generateDoc(vals, e.Documentation)
		for _, v := range e.Values {
			generateDoc(vals, v.Documentation)
			nm := goName(v.Name)
			more, ok := disambiguate[e.Name]
			if ok {
				usedDisambiguate[e.Name] = true
				nm = more.prefix + nm + more.suffix
				nm = goName(nm) // stringType
			}
			var val string
			switch v := v.Value.(type) {
			case string:
				val = fmt.Sprintf("%q", v)
			case float64:
				val = fmt.Sprintf("%d", int(v))
			default:
				log.Fatalf("impossible type %T", v)
			}
			fmt.Fprintf(vals, "\t%s %s = %s%s\n", nm, e.Name, val, linex(v.Line))
		}
		consts[nm] = vals.String()
	}
}
func genMarshal() {
	for _, nt := range genTypes {
		nm := goplsName(nt.typ)
		if !strings.HasPrefix(nm, "Or") {
			continue
		}
		names := []string{}
		for _, t := range nt.items {
			if notNil(t) {
				names = append(names, goplsName(t))
			}
		}
		sort.Strings(names)
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "// from line %d\n", nt.line)
		fmt.Fprintf(&buf, "func (t %s) MarshalJSON() ([]byte, error) {\n", nm)
		buf.WriteString("\tswitch x := t.Value.(type){\n")
		for _, nmx := range names {
			fmt.Fprintf(&buf, "\tcase %s:\n", nmx)
			fmt.Fprintf(&buf, "\t\treturn json.Marshal(x)\n")
		}
		buf.WriteString("\tcase nil:\n\t\treturn []byte(\"null\"), nil\n\t}\n")
		fmt.Fprintf(&buf, "\treturn nil, fmt.Errorf(\"type %%T not one of %v\", t)\n", names)
		buf.WriteString("}\n\n")

		fmt.Fprintf(&buf, "func (t *%s) UnmarshalJSON(x []byte) error {\n", nm)
		buf.WriteString("\tif string(x) == \"null\" {\n\t\tt.Value = nil\n\t\t\treturn nil\n\t}\n")
		for i, nmx := range names {
			fmt.Fprintf(&buf, "\tvar h%d %s\n", i, nmx)
			fmt.Fprintf(&buf, "\tif err := json.Unmarshal(x, &h%d); err == nil {\n\t\tt.Value = h%d\n\t\t\treturn nil\n\t\t}\n", i, i)
		}
		fmt.Fprintf(&buf, "return &UnmarshalError{\"unmarshal failed to match one of %v\"}", names)
		buf.WriteString("}\n\n")
		jsons[nm] = buf.String()
	}
}

func linex(n int) string {
	if *lineNumbers {
		return fmt.Sprintf(" // line %d", n)
	}
	return ""
}

func goplsName(t *Type) string {
	nm := typeNames[t]
	// translate systematic name to gopls name
	if newNm, ok := goplsType[nm]; ok {
		usedGoplsType[nm] = true
		nm = newNm
	}
	return nm
}

func notNil(t *Type) bool { // shutdwon is the special case that needs this
	return t != nil && (t.Kind != "base" || t.Name != "null")
}

func hasNilValue(t string) bool {
	// this may be unreliable, and need a supplementary table
	if strings.HasPrefix(t, "[]") || strings.HasPrefix(t, "*") {
		return true
	}
	if t == "interface{}" || t == "any" {
		return true
	}
	// that's all the cases that occur currently
	return false
}
