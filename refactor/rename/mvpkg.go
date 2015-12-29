// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// licence that can be found in the LICENSE file.

// This file contains the implementation of the 'gomvpkg' command
// whose main function is in golang.org/x/tools/cmd/gomvpkg.

package rename

// TODO(matloob):
// - think about what happens if the package is moving across version control systems.
// - think about windows, which uses "\" as its directory separator.
// - dot imports are not supported. Make sure it's clearly documented.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"text/template"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/refactor/importgraph"
)

// Move, given a package path and a destination package path, will try
// to move the given package to the new path. The Move function will
// first check for any conflicts preventing the move, such as a
// package already existing at the destination package path. If the
// move can proceed, it builds an import graph to find all imports of
// the packages whose paths need to be renamed. This includes uses of
// the subpackages of the package to be moved as those packages will
// also need to be moved. It then renames all imports to point to the
// new paths, and then moves the packages to their new paths.
func Move(ctxt *build.Context, from, to, moveTmpl string) error {
	srcDir, err := srcDir(ctxt, from)
	if err != nil {
		return err
	}

	// This should be the only place in the program that constructs
	// file paths.
	// TODO(matloob): test on Microsoft Windows.
	fromDir := buildutil.JoinPath(ctxt, srcDir, filepath.FromSlash(from))
	toDir := buildutil.JoinPath(ctxt, srcDir, filepath.FromSlash(to))
	toParent := filepath.Dir(toDir)
	if !buildutil.IsDir(ctxt, toParent) {
		return fmt.Errorf("parent directory does not exist for path %s", toDir)
	}

	// Build the import graph and figure out which packages to update.
	fwd, rev, errors := importgraph.Build(ctxt)
	if len(errors) > 0 {
		// With a large GOPATH tree, errors are inevitable.
		// Report them but proceed.
		fmt.Fprintf(os.Stderr, "While scanning Go workspace:\n")
		for path, err := range errors {
			fmt.Fprintf(os.Stderr, "Package %q: %s.\n", path, err)
		}
	}

	// Determine the affected packages---the set of packages whose import
	// statements need updating.
	affectedPackages := map[string]bool{from: true}
	destinations := map[string]string{} // maps old dir to new dir
	for pkg := range subpackages(ctxt, srcDir, from) {
		for r := range rev[pkg] {
			affectedPackages[r] = true
		}
		destinations[pkg] = strings.Replace(pkg,
			// Ensure directories have a trailing "/".
			filepath.Join(from, ""), filepath.Join(to, ""), 1)
	}

	// Load all the affected packages.
	iprog, err := loadProgram(ctxt, affectedPackages)
	if err != nil {
		return err
	}

	// Prepare the move command, if one was supplied.
	var cmd string
	if moveTmpl != "" {
		if cmd, err = moveCmd(moveTmpl, fromDir, toDir); err != nil {
			return err
		}
	}

	m := mover{
		ctxt:             ctxt,
		fwd:              fwd,
		rev:              rev,
		iprog:            iprog,
		from:             from,
		to:               to,
		fromDir:          fromDir,
		toDir:            toDir,
		affectedPackages: affectedPackages,
		destinations:     destinations,
		cmd:              cmd,
	}

	if err := m.checkValid(); err != nil {
		return err
	}

	m.move()

	return nil
}

// srcDir returns the absolute path of the srcdir containing pkg.
func srcDir(ctxt *build.Context, pkg string) (string, error) {
	for _, srcDir := range ctxt.SrcDirs() {
		path := buildutil.JoinPath(ctxt, srcDir, pkg)
		if buildutil.IsDir(ctxt, path) {
			return srcDir, nil
		}
	}
	return "", fmt.Errorf("src dir not found for package: %s", pkg)
}

// subpackages returns the set of packages in the given srcDir whose
// import paths start with dir.
func subpackages(ctxt *build.Context, srcDir string, dir string) map[string]bool {
	subs := map[string]bool{dir: true}

	// Find all packages under srcDir whose import paths start with dir.
	buildutil.ForEachPackage(ctxt, func(pkg string, err error) {
		if err != nil {
			log.Fatalf("unexpected error in ForEachPackage: %v", err)
		}

		if !strings.HasPrefix(pkg, path.Join(dir, "")) {
			return
		}

		p, err := ctxt.Import(pkg, "", build.FindOnly)
		if err != nil {
			log.Fatalf("unexpected: package %s can not be located by build context: %s", pkg, err)
		}
		if p.SrcRoot == "" {
			log.Fatalf("unexpected: could not determine srcDir for package %s: %s", pkg, err)
		}
		if p.SrcRoot != srcDir {
			return
		}

		subs[pkg] = true
	})

	return subs
}

type mover struct {
	// iprog contains all packages whose contents need to be updated
	// with new package names or import paths.
	iprog *loader.Program
	ctxt  *build.Context
	// fwd and rev are the forward and reverse import graphs
	fwd, rev importgraph.Graph
	// from and to are the source and destination import
	// paths. fromDir and toDir are the source and destination
	// absolute paths that package source files will be moved between.
	from, to, fromDir, toDir string
	// affectedPackages is the set of all packages whose contents need
	// to be updated to reflect new package names or import paths.
	affectedPackages map[string]bool
	// destinations maps each subpackage to be moved to its
	// destination path.
	destinations map[string]string
	// cmd, if not empty, will be executed to move fromDir to toDir.
	cmd string
}

func (m *mover) checkValid() error {
	const prefix = "invalid move destination"

	match, err := regexp.MatchString("^[_\\pL][_\\pL\\p{Nd}]*$", path.Base(m.to))
	if err != nil {
		panic("regexp.MatchString failed")
	}
	if !match {
		return fmt.Errorf("%s: %s; gomvpkg does not support move destinations "+
			"whose base names are not valid go identifiers", prefix, m.to)
	}

	if buildutil.FileExists(m.ctxt, m.toDir) {
		return fmt.Errorf("%s: %s conflicts with file %s", prefix, m.to, m.toDir)
	}
	if buildutil.IsDir(m.ctxt, m.toDir) {
		return fmt.Errorf("%s: %s conflicts with directory %s", prefix, m.to, m.toDir)
	}

	for _, toSubPkg := range m.destinations {
		if _, err := m.ctxt.Import(toSubPkg, "", build.FindOnly); err == nil {
			return fmt.Errorf("%s: %s; package or subpackage %s already exists",
				prefix, m.to, toSubPkg)
		}
	}

	return nil
}

// moveCmd produces the version control move command used to move fromDir to toDir by
// executing the given template.
func moveCmd(moveTmpl, fromDir, toDir string) (string, error) {
	tmpl, err := template.New("movecmd").Parse(moveTmpl)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	err = tmpl.Execute(&buf, struct {
		Src string
		Dst string
	}{fromDir, toDir})
	return buf.String(), err
}

func (m *mover) move() error {
	filesToUpdate := make(map[*ast.File]bool)

	// Change the moved package's "package" declaration to its new base name.
	pkg, ok := m.iprog.Imported[m.from]
	if !ok {
		log.Fatalf("unexpected: package %s is not in import map", m.from)
	}
	newName := filepath.Base(m.to)
	for _, f := range pkg.Files {
		// Update all import comments.
		for _, cg := range f.Comments {
			c := cg.List[0]
			if c.Slash >= f.Name.End() &&
				sameLine(m.iprog.Fset, c.Slash, f.Name.End()) &&
				(f.Decls == nil || c.Slash < f.Decls[0].Pos()) {
				if strings.HasPrefix(c.Text, `// import "`) {
					c.Text = `// import "` + m.to + `"`
					break
				}
				if strings.HasPrefix(c.Text, `/* import "`) {
					c.Text = `/* import "` + m.to + `" */`
					break
				}
			}
		}
		f.Name.Name = newName // change package decl
		filesToUpdate[f] = true
	}

	// Look through the external test packages (m.iprog.Created contains the external test packages).
	for _, info := range m.iprog.Created {
		// Change the "package" declaration of the external test package.
		if info.Pkg.Path() == m.from+"_test" {
			for _, f := range info.Files {
				f.Name.Name = newName + "_test" // change package decl
				filesToUpdate[f] = true
			}
		}

		// Mark all the loaded external test packages, which import the "from" package,
		// as affected packages and update the imports.
		for _, imp := range info.Pkg.Imports() {
			if imp.Path() == m.from {
				m.affectedPackages[info.Pkg.Path()] = true
				m.iprog.Imported[info.Pkg.Path()] = info
				if err := importName(m.iprog, info, m.from, path.Base(m.from), newName); err != nil {
					return err
				}
			}
		}
	}

	// Update imports of that package to use the new import name.
	// None of the subpackages will change their name---only the from package
	// itself will.
	for p := range m.rev[m.from] {
		if err := importName(m.iprog, m.iprog.Imported[p], m.from, path.Base(m.from), newName); err != nil {
			return err
		}
	}

	// Update import paths for all imports by affected packages.
	for ap := range m.affectedPackages {
		info, ok := m.iprog.Imported[ap]
		if !ok {
			log.Fatalf("unexpected: package %s is not in import map", ap)
		}
		for _, f := range info.Files {
			for _, imp := range f.Imports {
				importPath, _ := strconv.Unquote(imp.Path.Value)
				if newPath, ok := m.destinations[importPath]; ok {
					imp.Path.Value = strconv.Quote(newPath)

					oldName := path.Base(importPath)
					if imp.Name != nil {
						oldName = imp.Name.Name
					}

					newName := path.Base(newPath)
					if imp.Name == nil && oldName != newName {
						imp.Name = ast.NewIdent(oldName)
					} else if imp.Name == nil || imp.Name.Name == newName {
						imp.Name = nil
					}
					filesToUpdate[f] = true
				}
			}
		}
	}

	for f := range filesToUpdate {
		var buf bytes.Buffer
		if err := format.Node(&buf, m.iprog.Fset, f); err != nil {
			log.Printf("failed to pretty-print syntax tree: %v", err)
			continue
		}
		tokenFile := m.iprog.Fset.File(f.Pos())
		writeFile(tokenFile.Name(), buf.Bytes())
	}

	// Move the directories.
	// If either the fromDir or toDir are contained under version control it is
	// the user's responsibility to provide a custom move command that updates
	// version control to reflect the move.
	// TODO(matloob): If the parent directory of toDir does not exist, create it.
	//      For now, it's required that it does exist.

	if m.cmd != "" {
		// TODO(matloob): Verify that the windows and plan9 cases are correct.
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "windows":
			cmd = exec.Command("cmd", "/c", m.cmd)
		case "plan9":
			cmd = exec.Command("rc", "-c", m.cmd)
		default:
			cmd = exec.Command("sh", "-c", m.cmd)
		}
		cmd.Stderr = os.Stderr
		cmd.Stdout = os.Stdout
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("version control system's move command failed: %v", err)
		}

		return nil
	}

	return moveDirectory(m.fromDir, m.toDir)
}

// sameLine reports whether two positions in the same file are on the same line.
func sameLine(fset *token.FileSet, x, y token.Pos) bool {
	return fset.Position(x).Line == fset.Position(y).Line
}

var moveDirectory = func(from, to string) error {
	return os.Rename(from, to)
}
