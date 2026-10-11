// Package goxcmd bridges the standalone GoX front end into Go package discovery.
package goxcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"context"
	"fmt"
	gox "github.com/alfu32/go-shared/gox"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var CmdGoX = &base.Command{UsageLine: "go gox [emit file.gox]", Short: "translate GoX source to Go", Long: "GoX supports F formatting, X markup, bare JSX, local functions and components.", Run: run}

func run(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) != 2 || args[0] != "emit" {
		base.Fatalf("usage: go gox emit file.gox")
	}
	src, e := fsys.ReadFile(args[1])
	if e != nil {
		base.Fatal(e)
	}
	r, e := gox.TransformFile(args[1], src, gox.Options{BareJSX: true})
	if e != nil {
		base.Fatal(e)
	}
	os.Stdout.Write(r.Source)
}
func init() { fsys.SourceExpander = expand; base.AddChdirFlag(&CmdGoX.Flag) }

type renamed struct {
	fs.DirEntry
	name string
}

func (r renamed) Name() string { return r.name }
func expand(dir string, entries []fs.DirEntry) ([]fs.DirEntry, error) {
	if os.Getenv("GOX_STANDALONE_OVERLAY") == "1" {
		return entries, nil
	}
	excludeTests := cfg.CmdName == "build" || cfg.CmdName == "run" || cfg.CmdName == "install"
	hasGoX := false
	for _, d := range entries {
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".gox") {
			hasGoX = true
			break
		}
	}
	if !hasGoX {
		return entries, nil
	}
	files := map[string][]byte{}
	for _, d := range entries {
		if excludeTests && (strings.HasSuffix(d.Name(), "_test.go") || strings.HasSuffix(d.Name(), "_test.gox")) {
			continue
		}
		if d.IsDir() || (!strings.HasSuffix(d.Name(), ".go") && !strings.HasSuffix(d.Name(), ".gox")) {
			continue
		}
		path := filepath.Join(dir, d.Name())
		src, e := fsys.ReadFile(path)
		if e != nil {
			return nil, e
		}
		selected, e := gox.Selected(path, src, cfg.BuildContext)
		if e != nil {
			return nil, e
		}
		if selected {
			files[path] = src
		}
	}
	bindings, e := gox.PackageHookBindings(files)
	if e != nil {
		return nil, e
	}
	names := map[string]bool{}
	for _, d := range entries {
		names[d.Name()] = true
	}
	out := append([]fs.DirEntry(nil), entries...)
	for _, d := range entries {
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".gox") || excludeTests && strings.HasSuffix(d.Name(), "_test.gox") {
			continue
		}
		source := filepath.Join(dir, d.Name())
		src, e := fsys.ReadFile(source)
		if e != nil {
			return nil, e
		}
		selected, e := gox.Selected(source, src, cfg.BuildContext)
		if e != nil {
			return nil, e
		}
		if !selected {
			continue
		}
		name := gox.GeneratedName(d.Name())
		if names[name] {
			return nil, fmt.Errorf("GoX output conflicts with existing file: %s", filepath.Join(dir, name))
		}
		header, e := parser.ParseFile(token.NewFileSet(), source, src, parser.PackageClauseOnly)
		if e != nil {
			return nil, e
		}
		hookBindings := append([]string{}, bindings[header.Name.Name]...)
		path, e := gox.TransformCached(source, src, gox.ProjectOptions{Options: gox.Options{BareJSX: true, HookBindings: hookBindings}, Context: cfg.BuildContext})
		if e != nil {
			return nil, e
		}
		info, e := os.Stat(path)
		if e != nil {
			return nil, e
		}
		fsys.RegisterGenerated(filepath.Join(dir, name), path)
		out = append(out, renamed{fs.FileInfoToDirEntry(info), name})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name() < out[j].Name() })
	return out, nil
}

// PrepareArgs translates only source-file arguments, never arguments passed to
// a program by go run. Package patterns use normal discovery instead.
func PrepareArgs(command string, args []string) ([]string, error) {
	switch command {
	case "run", "build", "test", "list", "vet":
	default:
		return args, nil
	}
	out := append([]string(nil), args...)
	skipValue := false
	for j, a := range args {
		if skipValue {
			skipValue = false
			continue
		}
		if a == "-args" {
			break
		}
		if strings.HasPrefix(a, "-") {
			if !strings.Contains(a, "=") {
				switch a {
				case "-run", "-bench", "-timeout", "-count", "-fuzz", "-fuzztime", "-fuzzminimizetime", "-parallel", "-cpu", "-coverprofile", "-outputdir", "-shuffle", "-list", "-exec", "-vet", "-tags", "-o", "-overlay", "-modfile", "-mod", "-gcflags", "-ldflags", "-asmflags", "-toolexec", "-p", "-pkgdir", "-coverpkg":
					skipValue = true
				}
			}
			continue
		}
		if command == "run" && !strings.HasSuffix(a, ".go") && !strings.HasSuffix(a, ".gox") {
			break
		}
		if strings.HasSuffix(a, ".gox") {
			if e := fsys.Init(); e != nil {
				return nil, e
			}
			if _, e := fsys.ReadDir(filepath.Dir(a)); e != nil {
				return nil, e
			}
			out[j] = gox.GeneratedName(a)
		}
	}
	return out, nil
}
