// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package gobuild

import (
	"bufio";
	"exec";
	"fmt";
	"io";
	"go/ast";
	"go/parser";
	"os";
	"path";
	"sort";
	"strconv";
	"strings";
)

const (
	ShowErrors = 1<<iota;
	ForceDisplay;
)

var (
	theChar string;
	goarch string;
	goos string;
	bin = make(map[string] string);
)

var theChars = map[string] string {
	"amd64": "6",
	"386": "8",
	"arm": "5"
}

const ObjDir = "_obj"

func fatal(format string, args ...) {
	fmt.Fprintf(os.Stderr, "gobuild: %s\n", fmt.Sprintf(format, args));
	os.Exit(1);
}

func init() {
	goarch = os.Getenv("GOARCH");
	goos = os.Getenv("GOOS");

	var ok bool;
	theChar, ok = theChars[goarch];
	if !ok {
		fatal("unknown $GOARCH: %s", goarch);
	}

	var binaries = []string{
		theChar + "g",
		theChar + "c",
		theChar + "a",
		"gopack",
	};

	for i, v := range binaries {
		var s string;
		var err os.Error;
		if s, err = exec.LookPath(v); err != nil {
			fatal("cannot find binary %s", v);
		}
		bin[v] = s;
	}
}

func PushString(vp *[]string, p string) {
	v := *vp;
	n := len(v);
	if n >= cap(v) {
		m := 2*n + 10;
		a := make([]string, n, m);
		for i := range v {
			a[i] = v[i];
		}
		v = a;
	}
	v = v[0:n+1];
	v[n] = p;
	*vp = v;
}

func run(argv []string, flag int) (ok bool) {
	argv0 := bin[argv[0]];
	null, err := os.Open("/dev/null", os.O_RDWR, 0);
	if err != nil {
		fatal("open /dev/null: %s", err);
	}
	defer null.Close();
	r, w, err := os.Pipe();
	if err != nil {
		fatal("pipe: %s", err);
	}
	pid, err := os.ForkExec(argv0, argv, os.Environ(), "", []*os.File{null, w, w});
	defer r.Close();
	w.Close();
	if err != nil {
		fmt.Fprintln(os.Stderr, err);
		return false;
	}

	// Read the first line of output, if any.  Discard the rest.
	// If there is output and ShowErrors is set, show it,
	// preceded by a shell command line.
	// If ForceDisplay is set, we show the command even
	// if there's no output; this gets set if we're just trying
	// to keep the user informed.
	b := bufio.NewReader(r);
	line, err := b.ReadLineString('\n', true);
	if flag & ShowErrors != 0 && line != "" || flag & ForceDisplay != 0 {
		fmt.Fprint(os.Stderr, "$ ");
		for i, s := range argv {
			fmt.Fprint(os.Stderr, s, " ");
		}
		fmt.Fprint(os.Stderr, "\n");
		fmt.Fprint(os.Stderr, "  ", line);
		io.Copy(r, null);	// don't let process block on pipe
	}
	waitmsg, err := os.Wait(pid, 0);
	if err != nil {
		fmt.Fprintln(os.Stderr, err);
		return false;
	}
	return waitmsg.Exited() && waitmsg.ExitStatus() == 0;
}

func Build(cmd []string, file string, flag int) (ok bool) {
	var argv []string;
	for i, c := range cmd {
		PushString(&argv, c);
	}
	PushString(&argv, file);
	return run(argv, flag);
}

func Archive(pkg string, files []string) {
	argv := []string{ "gopack", "grc", pkg };
	for i, file := range files {
		PushString(&argv, file);
	}
	if !run(argv, ShowErrors) {
		fatal("archive failed");
	}
}

func Compiler(file string) []string {
	switch {
	case strings.HasSuffix(file, ".go"):
		return []string{ theChar + "g", "-I", ObjDir };
	case strings.HasSuffix(file, ".c"):
		return []string{ theChar + "c", "-FVw" };
	case strings.HasSuffix(file, ".s"):
		return []string{ theChar + "a" };
	}
	fatal("don't know how to compile %s", file);
	return nil;
}

func Object(file, suffix string) string {
	ext := path.Ext(file);
	return file[0:len(file)-len(ext)] + "." + suffix;
}

// Dollarstring returns s with literal goarch/goos values
// replaced by $lGOARCHr where l and r are the specified delimeters.
func dollarString(s, l, r string) string {
	out := "";
	j := 0;	// index of last byte in s copied to out.
	for i := 0; i < len(s); {
		switch {
		case i+len(goarch) <= len(s) && s[i:i+len(goarch)] == goarch:
			out += s[j:i];
			out += "$" + l + "GOARCH" + r;
			i += len(goarch);
			j = i;
		case i+len(goos) <= len(s) && s[i:i+len(goos)] == goos:
			out += s[j:i];
			out += "$" + l + "GOOS" + r;
			i += len(goos);
			j = i;
		default:
			i++;
		}
	}
	out += s[j:len(s)];
	return out;
}

// dollarString wrappers.
// Print ShellString(s) or MakeString(s) depending on
// the context in which the result will be interpreted.
type ShellString string;
func (s ShellString) String() string {
	return dollarString(string(s), "{", "}");
}

type MakeString string;
func (s MakeString) String() string {
	return dollarString(string(s), "(", ")");
}

// TODO(rsc): Should this be in the AST library?
func LitString(p []*ast.StringLit) (string, os.Error) {
	s := "";
	for i, lit := range p {
		t, err := strconv.Unquote(string(lit.Value));
		if err != nil {
			return "", err;
		}
		s += t;
	}
	return s, nil;
}

func PackageImports(file string) (pkg string, imports []string, err1 os.Error) {
	prog, err := parser.ParseFile(file, nil, parser.ImportsOnly);
	if err != nil {
		return "", nil, err;
	}

	// Normally one must consult the types of decl and spec,
	// but we told the parser to return imports only,
	// so assume it did.
	var imp []string;
	for _, decl := range prog.Decls {
		for _, spec := range decl.(*ast.GenDecl).Specs {
			str, err := LitString(spec.(*ast.ImportSpec).Path);
			if err != nil {
				return "", nil, os.NewError("invalid import specifier");	// better than os.EINVAL
			}
			PushString(&imp, str);
		}
	}

	// TODO(rsc): should be prog.Package.Value
	return prog.Name.Value, imp, nil;
}

func SourceFiles(dir string) ([]string, os.Error) {
	f, err := os.Open(dir, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	names, err1 := f.Readdirnames(-1);
	f.Close();
	out := make([]string, 0, len(names));
	for i, name := range names {
		if strings.HasSuffix(name, ".go")
		|| strings.HasSuffix(name, ".c")
		|| strings.HasSuffix(name, ".s") {
			n := len(out);
			out = out[0:n+1];
			out[n] = name;
		}
	}
	sort.SortStrings(out);
	return out, nil;
}

// TODO(rsc): Implement these for real as
// os.MkdirAll and os.RemoveAll and then
// make these wrappers that call fatal on error.

func MkdirAll(name string) {
	p, err := exec.Run("/bin/mkdir", []string{"mkdir", "-p", name}, os.Environ(), exec.DevNull, exec.PassThrough, exec.PassThrough);
	if err != nil {
		fatal("run /bin/mkdir: %v", err);
	}
	w, err1 := p.Wait(0);
	if err1 != nil {
		fatal("wait /bin/mkdir: %v", err);
	}
	if !w.Exited() || w.ExitStatus() != 0 {
		fatal("/bin/mkdir: %v", w);
	}
}

func RemoveAll(name string) {
	p, err := exec.Run("/bin/rm", []string{"rm", "-rf", name}, os.Environ(), exec.DevNull, exec.PassThrough, exec.PassThrough);
	if err != nil {
		fatal("run /bin/rm: %v", err);
	}
	w, err := p.Wait(0);
	if err != nil {
		fatal("wait /bin/rm: %v", err);
	}
	if !w.Exited() || w.ExitStatus() != 0 {
		fatal("/bin/rm: %v", w);
	}

}

