package main

import (
	"fmt"
	gox "github.com/alfu32/go-shared/gox"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() { os.Exit(run(os.Args[1:])) }
func run(args []string) int {
	if len(args) == 0 || args[0] == "help" || args[0] == "--help" {
		fmt.Println("usage: gox emit FILE [--output FILE]\n       gox check|build|run|test|list|vet [GO OPTIONS] [PACKAGES] [-- PROGRAM ARGS]")
		return 0
	}
	if args[0] == "emit" {
		if len(args) < 2 {
			return fail("emit requires a file", 2)
		}
		src, e := os.ReadFile(args[1])
		if e != nil {
			return fail(e, 1)
		}
		r, e := gox.TransformFile(args[1], src, gox.Options{BareJSX: true})
		if e != nil {
			return fail(e, 1)
		}
		if len(args) == 2 {
			os.Stdout.Write(r.Source)
			return 0
		}
		if len(args) != 4 || args[2] != "--output" {
			return fail("expected --output FILE", 2)
		}
		a, _ := filepath.Abs(args[1])
		b, _ := filepath.Abs(args[3])
		infoA, _ := os.Stat(a)
		infoB, _ := os.Stat(b)
		if a == b || infoA != nil && infoB != nil && os.SameFile(infoA, infoB) {
			return fail("output would overwrite source", 2)
		}
		if e = writeOutput(b, r.Source); e != nil {
			return fail(e, 1)
		}
		return 0
	}
	cmd := args[0]
	switch cmd {
	case "check":
		cmd = "vet"
	case "build", "run", "test", "list", "vet":
	default:
		return fail("unknown command "+cmd, 2)
	}
	for j := 1; j < len(args); j++ {
		a := args[j]
		if a == "--" || a == "-args" || cmd == "run" && !strings.HasPrefix(a, "-") {
			break
		}
		if a == "-C" || strings.HasPrefix(a, "-C=") {
			dir := strings.TrimPrefix(a, "-C=")
			end := j + 1
			if a == "-C" {
				if end >= len(args) {
					return fail("-C requires a directory", 2)
				}
				dir = args[end]
				end++
			}
			if e := os.Chdir(dir); e != nil {
				return fail(e, 1)
			}
			args = append(args[:j], args[end:]...)
			j--
		}
		if a == "-overlay" || strings.HasPrefix(a, "-overlay=") {
			return fail("standalone GoX manages -overlay; use the native GoX toolchain for editor overlays", 2)
		}
	}
	cwd, e := os.Getwd()
	if e != nil {
		return fail(e, 1)
	}
	root := cwd
	for d := cwd; ; d = filepath.Dir(d) {
		if _, e := os.Stat(filepath.Join(d, "go.mod")); e == nil {
			root = d
			break
		}
		if filepath.Dir(d) == d {
			break
		}
	}
	ctx := build.Default
	for j, a := range args {
		if strings.HasPrefix(a, "-tags=") {
			ctx.BuildTags = strings.FieldsFunc(strings.TrimPrefix(a, "-tags="), func(r rune) bool { return r == ',' || r == ' ' })
		}
		if a == "-tags" && j+1 < len(args) {
			ctx.BuildTags = strings.Split(args[j+1], ",")
		}
	}
	o, e := prepareWorkspace(root, gox.ProjectOptions{Options: gox.Options{BareJSX: true}, Context: ctx, ExcludeTests: cmd == "build" || cmd == "run"})
	if e != nil {
		return fail(e, 1)
	}
	dir, e := os.MkdirTemp("", "gox-overlay-")
	if e != nil {
		return fail(e, 1)
	}
	defer os.RemoveAll(dir)
	overlay := filepath.Join(dir, "overlay.json")
	if e = o.Write(overlay); e != nil {
		return fail(e, 1)
	}
	forward := []string{cmd, "-overlay=" + overlay}
	forward = append(forward, forwardArgs(cmd, args[1:])...)

	tool, prefix := goTool()
	forward = append(prefix, forward...)
	c := exec.Command(tool, forward...)
	c.Env = append(os.Environ(), "GOX_STANDALONE_OVERLAY=1")
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	if e = c.Run(); e != nil {
		if x, ok := e.(*exec.ExitError); ok {
			return x.ExitCode()
		}
		return fail(e, 1)
	}
	return 0
}
func fail(e any, status int) int { fmt.Fprintln(os.Stderr, "gox:", e); return status }

// Run takes either a list of source files or one package followed by program
// arguments. A -- separator belongs to GoX only; later separators are preserved.
func forwardArgs(command string, args []string) []string {
	out := make([]string, 0, len(args))
	program := false
	files := false
	flagValue := false
	for _, a := range args {
		if program {
			out = append(out, a)
			continue
		}
		if flagValue {
			out = append(out, a)
			flagValue = false
			continue
		}
		if a == "--" {
			program = true
			continue
		}
		if strings.HasPrefix(a, "-") {
			out = append(out, a)
			switch a {
			case "-tags", "-o", "-C", "-overlay", "-modfile", "-mod", "-exec", "-gcflags", "-ldflags", "-asmflags", "-toolexec", "-p", "-pkgdir", "-coverpkg":
				flagValue = true
			}
			continue
		}
		source := strings.HasSuffix(a, ".go") || strings.HasSuffix(a, ".gox")
		if command == "run" {
			if files && !source {
				program = true
			}
			if !files && !source {
				out = append(out, a)
				program = true
				continue
			}
			if source {
				files = true
			}
		}
		if !program && strings.HasSuffix(a, ".gox") {
			a = gox.GeneratedName(a)
		}
		out = append(out, a)
	}
	return out
}

func writeOutput(path string, content []byte) error {
	f, e := os.CreateTemp(filepath.Dir(path), ".gox-output-")
	if e != nil {
		return e
	}
	defer os.Remove(f.Name())
	if _, e = f.Write(content); e != nil {
		f.Close()
		return e
	}
	if e = f.Chmod(0644); e != nil {
		f.Close()
		return e
	}
	if e = f.Close(); e != nil {
		return e
	}
	return os.Rename(f.Name(), path)
}
