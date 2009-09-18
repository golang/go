// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Annotate Crefs in Prog with C types by parsing gcc debug output.

package main

import (
	"debug/dwarf";
	"debug/elf";
	"debug/macho";
	"fmt";
	"os";
	"strconv";
	"strings";
)

func (p *Prog) loadDebugInfo() {
	// Construct a slice of unique names from p.Crefs.
	m := make(map[string]int);
	for _, c := range p.Crefs {
		m[c.Name] = -1;
	}
	names := make([]string, 0, len(m));
	for name, _ := range m {
		i := len(names);
		names = names[0:i+1];
		names[i] = name;
		m[name] = i;
	}

	// Coerce gcc into telling us whether each name is
	// a type, a value, or undeclared.  We compile a function
	// containing the line:
	//	name;
	// If name is a type, gcc will print:
	//	x.c:2: warning: useless type name in empty declaration
	// If name is a value, gcc will print
	//	x.c:2: warning: statement with no effect
	// If name is undeclared, gcc will print
	//	x.c:2: error: 'name' undeclared (first use in this function)
	// A line number directive causes the line number to
	// correspond to the index in the names array.
	var b strings.Buffer;
	b.WriteString(p.Preamble);
	b.WriteString("void f(void) {\n");
	b.WriteString("#line 0 \"cgo-test\"\n");
	for _, n := range names {
		b.WriteString(n);
		b.WriteString(";\n");
	}
	b.WriteString("}\n");

	kind := make(map[string]string);
	_, stderr := gccDebug(b.Bytes());
	if stderr == "" {
		fatal("gcc produced no output");
	}
	for _, line := range strings.Split(stderr, "\n", 0) {
		if len(line) < 9 || line[0:9] != "cgo-test:" {
			continue;
		}
		line = line[9:len(line)];
		colon := strings.Index(line, ":");
		if colon < 0 {
			continue;
		}
		i, err := strconv.Atoi(line[0:colon]);
		if err != nil {
			continue;
		}
		what := "";
		switch {
		default:
			continue;
		case strings.Index(line, "warning: useless type name in empty declaration") >= 0:
			what = "type";
		case strings.Index(line, "warning: statement with no effect") >= 0:
			what = "value";
		case strings.Index(line, "undeclared") >= 0:
			what = "error";
		}
		if old, ok := kind[names[i]]; ok && old != what {
			error(noPos, "inconsistent gcc output about C.%s", names[i]);
		}
		kind[names[i]] = what;
	}
	for _, n := range names {
		if _, ok := kind[n]; !ok {
			error(noPos, "could not determine kind of name for C.%s", n);
		}
	}

	// Extract the types from the DWARF section of an object
	// from a well-formed C program.  Gcc only generates DWARF info
	// for symbols in the object file, so it is not enough to print the
	// preamble and hope the symbols we care about will be there.
	// Instead, emit
	//	typeof(names[i]) *__cgo__i;
	// for each entry in names and then dereference the type we
	// learn for __cgo__i.
	b.Reset();
	b.WriteString(p.Preamble);
	for i, n := range names {
		fmt.Fprintf(&b, "typeof(%s) *__cgo__%d;\n", n, i);
	}
	d, stderr := gccDebug(b.Bytes());
	if d == nil {
		fatal("gcc failed:\n%s\non input:\n%s", stderr, b.Bytes());
	}

	// Scan DWARF info for  top-level TagVariable entries with AttrName __cgo__i.
	types := make([]dwarf.Type, len(names));
	r := d.Reader();
	for {
		e, err := r.Next();
		if err != nil {
			fatal("reading DWARF entry: %s", err);
		}
		if e == nil {
			break;
		}
		if e.Tag != dwarf.TagVariable {
			goto Continue;
		}
		name, _ := e.Val(dwarf.AttrName).(string);
		typOff, _ := e.Val(dwarf.AttrType).(dwarf.Offset);
		if name == "" || typOff == 0 {
			fatal("malformed DWARF TagVariable entry");
		}
		if !strings.HasPrefix(name, "__cgo__") {
			goto Continue;
		}
		typ, err := d.Type(typOff);
		if err != nil {
			fatal("loading DWARF type: %s", err);
		}
		t, ok := typ.(*dwarf.PtrType);
		if !ok || t == nil {
			fatal("internal error: %s has non-pointer type", name);
		}
		i, err := strconv.Atoi(name[7:len(name)]);
		if err != nil {
			fatal("malformed __cgo__ name: %s", name);
		}
		types[i] = t.Type;

	Continue:
		if e.Tag != dwarf.TagCompileUnit {
			r.SkipChildren();
		}
	}

	// Apply types to Crefs.
	for _, c := range p.Crefs {
		i := m[c.Name];
		c.TypeName = kind[c.Name] == "type";
		c.DebugType = types[i];
	}
}

// gccDebug runs gcc -gdwarf-2 over the C program stdin and
// returns the corresponding DWARF data and any messages
// printed to standard error.
func gccDebug(stdin []byte) (*dwarf.Data, string) {
	machine := "-m32";
	if os.Getenv("GOARCH") == "amd64" {
		machine = "-m64";
	}

	tmp := "_cgo_.o";
	_, stderr, ok := run(stdin, []string{
		"gcc",
		machine,
		"-Wall",	// many warnings
		"-Werror",	// warnings are errors
		"-o"+tmp, 	// write object to tmp
		"-gdwarf-2", 	// generate DWARF v2 debugging symbols
		"-c",	// do not link
		"-xc", 	// input language is C
		"-",	// read input from standard input
	});
	if !ok {
		return nil, string(stderr);
	}

	// Try to parse f as ELF and Mach-O and hope one works.
	var f interface{DWARF() (*dwarf.Data, os.Error)};
	var err os.Error;
	if f, err = elf.Open(tmp); err != nil {
		if f, err = macho.Open(tmp); err != nil {
			fatal("cannot parse gcc output %s as ELF or Mach-O object", tmp);
		}
	}

	d, err := f.DWARF();
	if err != nil {
		fatal("cannot load DWARF debug information from %s: %s", tmp, err);
	}
	return d, "";
}

