// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
	"cmd/internal/par"
	"cmd/internal/pathcache"
	"errors"
	"fmt"
	"internal/lazyregexp"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// A Shell runs shell commands and performs shell-like file system operations.
//
// Shell tracks context related to running commands, and form a tree much like
// context.Context.
type Shell struct {
	action       *Action // nil for the root shell
	*shellShared         // per-Builder state shared across Shells
}

// shellShared is Shell state shared across all Shells derived from a single
// root shell (generally a single Builder).
type shellShared struct {
	workDir string // $WORK, immutable

	printLock sync.Mutex
	printer   load.Printer
	scriptDir string // current directory in printed script

	mkdirCache par.Cache[string, error] // a cache of created directories
}

// NewShell returns a new Shell.
//
// Shell will internally serialize calls to the printer.
// If printer is nil, it uses load.DefaultPrinter.
func NewShell(workDir string, printer load.Printer) *Shell {
	if printer == nil {
		printer = load.DefaultPrinter()
	}
	shared := &shellShared{
		workDir: workDir,
		printer: printer,
	}
	return &Shell{shellShared: shared}
}

func (sh *Shell) pkg() *load.Package {
	if sh.action == nil {
		return nil
	}
	return sh.action.Package
}

// Printf emits a to this Shell's output stream, formatting it like fmt.Printf.
// It is safe to call concurrently.
func (sh *Shell) Printf(format string, a ...any) {
	sh.printLock.Lock()
	defer sh.printLock.Unlock()
	sh.printer.Printf(sh.pkg(), format, a...)
}

func (sh *Shell) printfLocked(format string, a ...any) {
	sh.printer.Printf(sh.pkg(), format, a...)
}

// Errorf reports an error on sh's package and sets the process exit status to 1.
func (sh *Shell) Errorf(format string, a ...any) {
	sh.printLock.Lock()
	defer sh.printLock.Unlock()
	sh.printer.Errorf(sh.pkg(), format, a...)
}

// WithAction returns a Shell identical to sh, but bound to Action a.
func (sh *Shell) WithAction(a *Action) *Shell {
	sh2 := *sh
	sh2.action = a
	return &sh2
}

// Shell returns a shell for running commands on behalf of Action a.
func (b *Builder) Shell(a *Action) *Shell {
	if a == nil {
		// The root shell has a nil Action. The point of this method is to
		// create a Shell bound to an Action, so disallow nil Actions here.
		panic("nil Action")
	}
	if a.sh == nil {
		a.sh = b.backgroundSh.WithAction(a)
	}
	return a.sh
}

// BackgroundShell returns a Builder-wide Shell that's not bound to any Action.
// Try not to use this unless there's really no sensible Action available.
func (b *Builder) BackgroundShell() *Shell {
	return b.backgroundSh
}

// moveOrCopyFile is like 'mv src dst' or 'cp src dst'.
func (sh *Shell) moveOrCopyFile(dst, src string, perm fs.FileMode, force bool) error {
	if cfg.BuildN {
		sh.ShowCmd("", "mv %s %s", src, dst)
		return nil
	}

	// If we can update the mode and rename to the dst, do it.
	// Otherwise fall back to standard copy.

	// If the source is in the build cache, we need to copy it.
	dir, _, _ := cache.DefaultDir()
	if strings.HasPrefix(src, dir) {
		return sh.CopyFile(dst, src, perm, force)
	}

	// On Windows, always copy the file, so that we respect the NTFS
	// permissions of the parent folder. https://golang.org/issue/22343.
	// What matters here is not cfg.Goos (the system we are building
	// for) but runtime.GOOS (the system we are building on).
	if runtime.GOOS == "windows" {
		return sh.CopyFile(dst, src, perm, force)
	}

	// If the destination directory has the group sticky bit set,
	// we have to copy the file to retain the correct permissions.
	// https://golang.org/issue/18878
	if fi, err := os.Stat(filepath.Dir(dst)); err == nil {
		if fi.IsDir() && (fi.Mode()&fs.ModeSetgid) != 0 {
			return sh.CopyFile(dst, src, perm, force)
		}
	}

	// The perm argument is meant to be adjusted according to umask,
	// but we don't know what the umask is.
	// Create a dummy file to find out.
	// This avoids build tags and works even on systems like Plan 9
	// where the file mask computation incorporates other information.
	mode := perm
	f, err := os.OpenFile(filepath.Clean(dst)+"-go-tmp-umask", os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err == nil {
		fi, err := f.Stat()
		if err == nil {
			mode = fi.Mode() & 0777
		}
		name := f.Name()
		f.Close()
		os.Remove(name)
	}

	if err := os.Chmod(src, mode); err == nil {
		if err := os.Rename(src, dst); err == nil {
			if cfg.BuildX {
				sh.ShowCmd("", "mv %s %s", src, dst)
			}
			return nil
		}
	}

	return sh.CopyFile(dst, src, perm, force)
}

// copyFile is like 'cp src dst'.
func (sh *Shell) CopyFile(dst, src string, perm fs.FileMode, force bool) error {
	if cfg.BuildN || cfg.BuildX {
		sh.ShowCmd("", "cp %s %s", src, dst)
		if cfg.BuildN {
			return nil
		}
	}

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	// Be careful about removing/overwriting dst.
	// Do not remove/overwrite if dst exists and is a directory
	// or a non-empty non-object file.
	if fi, err := os.Stat(dst); err == nil {
		if fi.IsDir() {
			return fmt.Errorf("build output %q already exists and is a directory", dst)
		}
		if !force && fi.Mode().IsRegular() && fi.Size() != 0 && !isObject(dst) {
			return fmt.Errorf("build output %q already exists and is not an object file", dst)
		}
	}

	// On Windows, remove lingering ~ file from last attempt.
	if runtime.GOOS == "windows" {
		if _, err := os.Stat(dst + "~"); err == nil {
			os.Remove(dst + "~")
		}
	}

	mayberemovefile(dst)
	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil && runtime.GOOS == "windows" {
		// Windows does not allow deletion of a binary file
		// while it is executing. Try to move it out of the way.
		// If the move fails, which is likely, we'll try again the
		// next time we do an install of this binary.
		if err := os.Rename(dst, dst+"~"); err == nil {
			os.Remove(dst + "~")
		}
		df, err = os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	}
	if err != nil {
		return fmt.Errorf("copying %s: %w", src, err) // err should already refer to dst
	}

	_, err = io.Copy(df, sf)
	df.Close()
	if err != nil {
		mayberemovefile(dst)
		return fmt.Errorf("copying %s to %s: %v", src, dst, err)
	}
	return nil
}

// mayberemovefile removes a file only if it is a regular file
// When running as a user with sufficient privileges, we may delete
// even device files, for example, which is not intended.
func mayberemovefile(s string) {
	if fi, err := os.Lstat(s); err == nil && !fi.Mode().IsRegular() {
		return
	}
	os.Remove(s)
}

// writeFile writes the text to file.
func (sh *Shell) writeFile(file string, text []byte) error {
	if cfg.BuildN || cfg.BuildX {
		switch {
		case len(text) == 0:
			sh.ShowCmd("", "echo -n > %s # internal", file)
		case bytes.IndexByte(text, '\n') == len(text)-1:
			// One line. Use a simpler "echo" command.
			sh.ShowCmd("", "echo '%s' > %s # internal", bytes.TrimSuffix(text, []byte("\n")), file)
		default:
			// Use the most general form.
			sh.ShowCmd("", "cat >%s << 'EOF' # internal\n%sEOF", file, text)
		}
	}
	if cfg.BuildN {
		return nil
	}
	return os.WriteFile(file, text, 0666)
}

// Mkdir makes the named directory.
func (sh *Shell) Mkdir(dir string) error {
	// Make Mkdir(a.Objdir) a no-op instead of an error when a.Objdir == "".
	if dir == "" {
		return nil
	}

	// We can be a little aggressive about being
	// sure directories exist. Skip repeated calls.
	return sh.mkdirCache.Do(dir, func() error {
		if cfg.BuildN || cfg.BuildX {
			sh.ShowCmd("", "mkdir -p %s", dir)
			if cfg.BuildN {
				return nil
			}
		}

		return os.MkdirAll(dir, 0777)
	})
}

// RemoveAll is like 'rm -rf'. It attempts to remove all paths even if there's
// an error, and returns the first error.
func (sh *Shell) RemoveAll(paths ...string) error {
	if cfg.BuildN || cfg.BuildX {
		// Don't say we are removing the directory if we never created it.
		show := func() bool {
			for _, path := range paths {
				if _, ok := sh.mkdirCache.Get(path); ok {
					return true
				}
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					return true
				}
			}
			return false
		}
		if show() {
			sh.ShowCmd("", "rm -rf %s", strings.Join(paths, " "))
		}
	}
	if cfg.BuildN {
		return nil
	}

	var err error
	for _, path := range paths {
		if err2 := os.RemoveAll(path); err2 != nil && err == nil {
			err = err2
		}
	}
	return err
}

// Symlink creates a symlink newname -> oldname.
func (sh *Shell) Symlink(oldname, newname string) error {
	// It's not an error to try to recreate an existing symlink.
	if link, err := os.Readlink(newname); err == nil && link == oldname {
		return nil
	}

	if cfg.BuildN || cfg.BuildX {
		sh.ShowCmd("", "ln -s %s %s", oldname, newname)
		if cfg.BuildN {
			return nil
		}
	}
	return os.Symlink(oldname, newname)
}

// fmtCmd formats a command in the manner of fmt.Sprintf but also:
//
//	fmtCmd replaces the value of b.WorkDir with $WORK.
func (sh *Shell) fmtCmd(dir string, format string, args ...any) string {
	cmd := fmt.Sprintf(format, args...)
	if sh.workDir != "" && !strings.HasPrefix(cmd, "cat ") {
		cmd = strings.ReplaceAll(cmd, sh.workDir, "$WORK")
		escaped := strconv.Quote(sh.workDir)
		escaped = escaped[1 : len(escaped)-1] // strip quote characters
		if escaped != sh.workDir {
			cmd = strings.ReplaceAll(cmd, escaped, "$WORK")
		}
	}
	return cmd
}

// ShowCmd prints the given command to standard output
// for the implementation of -n or -x.
//
// ShowCmd also replaces the name of the current script directory with dot (.)
// but only when it is at the beginning of a space-separated token.
//
// If dir is not "" or "/" and not the current script directory, ShowCmd first
// prints a "cd" command to switch to dir and updates the script directory.
func (sh *Shell) ShowCmd(dir string, format string, args ...any) {
	// Use the output lock directly so we can manage scriptDir.
	sh.printLock.Lock()
	defer sh.printLock.Unlock()

	cmd := sh.fmtCmd(dir, format, args...)

	if dir != "" && dir != "/" {
		if dir != sh.scriptDir {
			// Show changing to dir and update the current directory.
			sh.printfLocked("%s", sh.fmtCmd("", "cd %s\n", dir))
			sh.scriptDir = dir
		}
		// Replace scriptDir is our working directory. Replace it
		// with "." in the command.
		dot := " ."
		if dir[len(dir)-1] == filepath.Separator {
			dot += string(filepath.Separator)
		}
		cmd = strings.ReplaceAll(" "+cmd, " "+dir, dot)[1:]
	}

	sh.printfLocked("%s\n", cmd)
}

// reportCmd reports the output and exit status of a command. The cmdOut and
// cmdErr arguments are the output and exit error of the command, respectively.
//
// The exact reporting behavior is as follows:
//
//	cmdOut  cmdErr  Result
//	""      nil     print nothing, return nil
//	!=""    nil     print output, return nil
//	""      !=nil   print nothing, return cmdErr (later printed)
//	!=""    !=nil   print nothing, ignore err, return output as error (later printed)
//
// reportCmd returns a non-nil error if and only if cmdErr != nil. It assumes
// that the command output, if non-empty, is more detailed than the command
// error (which is usually just an exit status), so prefers using the output as
// the ultimate error. Typically, the caller should return this error from an
// Action, which it will be printed by the Builder.
//
// reportCmd formats the output as "# desc" followed by the given output. The
// output is expected to contain references to 'dir', usually the source
// directory for the package that has failed to build. reportCmd rewrites
// mentions of dir with a relative path to dir when the relative path is
// shorter. This is usually more pleasant. For example, if fmt doesn't compile
// and we are in src/html, the output is
//
//	$ go build
//	# fmt
//	../fmt/print.go:1090: undefined: asdf
//	$
//
// instead of
//
//	$ go build
//	# fmt
//	/usr/gopher/go/src/fmt/print.go:1090: undefined: asdf
//	$
//
// reportCmd also replaces references to the work directory with $WORK, replaces
// cgo file paths with the original file path, and replaces cgo-mangled names
// with "C.name".
//
// desc is optional. If "", a.Package.Desc() is used.
//
// dir is optional. If "", a.Package.Dir is used.
func (sh *Shell) reportCmd(desc, dir string, cmdOut []byte, cmdErr error) error {
	if len(cmdOut) == 0 && cmdErr == nil {
		// Common case
		return nil
	}
	if len(cmdOut) == 0 && cmdErr != nil {
		// Just return the error.
		//
		// TODO: This is what we've done for a long time, but it may be a
		// mistake because it loses all of the extra context and results in
		// ultimately less descriptive output. We should probably just take the
		// text of cmdErr as the output in this case and do everything we
		// otherwise would. We could chain the errors if we feel like it.
		return cmdErr
	}

	// Fetch defaults from the package.
	var p *load.Package
	a := sh.action
	if a != nil {
		p = a.Package
	}
	var importPath string
	if p != nil {
		importPath = p.ImportPath
		if desc == "" {
			desc = p.Desc()
		}
		if dir == "" {
			dir = p.Dir
		}
	}

	out := string(cmdOut)

	if !strings.HasSuffix(out, "\n") {
		out = out + "\n"
	}

	// Replace workDir with $WORK
	out = replacePrefix(out, sh.workDir, "$WORK")

	// Rewrite mentions of dir with a relative path to dir
	// when the relative path is shorter.
	for {
		// Note that dir starts out long, something like
		// /foo/bar/baz/root/a
		// The target string to be reduced is something like
		// (blah-blah-blah) /foo/bar/baz/root/sibling/whatever.go:blah:blah
		// /foo/bar/baz/root/a doesn't match /foo/bar/baz/root/sibling, but the prefix
		// /foo/bar/baz/root does.  And there may be other niblings sharing shorter
		// prefixes, the only way to find them is to look.
		// This doesn't always produce a relative path --
		// /foo is shorter than ../../.., for example.
		if reldir := base.ShortPath(dir); reldir != dir {
			out = replacePrefix(out, dir, reldir)
			if filepath.Separator == '\\' {
				// Don't know why, sometimes this comes out with slashes, not backslashes.
				wdir := strings.ReplaceAll(dir, "\\", "/")
				out = replacePrefix(out, wdir, reldir)
			}
		}
		dirP := filepath.Dir(dir)
		if dir == dirP {
			break
		}
		dir = dirP
	}

	// Fix up output referring to cgo-generated code to be more readable.
	// Replace x.go:19[/tmp/.../x.cgo1.go:18] with x.go:19.
	// Replace *[100]_Ctype_foo with *[100]C.foo.
	// If we're using -x, assume we're debugging and want the full dump, so disable the rewrite.
	if !cfg.BuildX && cgoLine.MatchString(out) {
		out = cgoLine.ReplaceAllString(out, "")
		out = cgoTypeSigRe.ReplaceAllString(out, "C.")
	}

	// Usually desc is already p.Desc(), but if not, signal cmdError.Error to
	// add a line explicitly mentioning the import path.
	needsPath := importPath != "" && p != nil && desc != p.Desc()

	err := &cmdError{desc, out, importPath, needsPath}
	if cmdErr != nil {
		// The command failed. Report the output up as an error.
		return err
	}
	// The command didn't fail, so just print the output as appropriate.
	if a != nil && a.output != nil {
		// The Action is capturing output.
		a.output = append(a.output, err.Error()...)
	} else {
		// Write directly to the Builder output.
		sh.Printf("%s", err)
	}
	return nil
}

// replacePrefix is like strings.ReplaceAll, but only replaces instances of old
// that are preceded by ' ', '\t', or appear at the beginning of a line.
func replacePrefix(s, old, new string) string {
	n := strings.Count(s, old)
	if n == 0 {
		return s
	}

	s = strings.ReplaceAll(s, " "+old, " "+new)
	s = strings.ReplaceAll(s, "\n"+old, "\n"+new)
	s = strings.ReplaceAll(s, "\n\t"+old, "\n\t"+new)
	if strings.HasPrefix(s, old) {
		s = new + s[len(old):]
	}
	return s
}

type cmdError struct {
	desc       string
	text       string
	importPath string
	needsPath  bool // Set if desc does not already include the import path
}

func (e *cmdError) Error() string {
	var msg string
	if e.needsPath {
		// Ensure the import path is part of the message.
		// Clearly distinguish the description from the import path.
		msg = fmt.Sprintf("# %s\n# [%s]\n", e.importPath, e.desc)
	} else {
		msg = "# " + e.desc + "\n"
	}
	return msg + e.text
}

func (e *cmdError) ImportPath() string {
	return e.importPath
}

var cgoLine = lazyregexp.New(`\[[^\[\]]+\.(cgo1|cover)\.go:[0-9]+(:[0-9]+)?\]`)
var cgoTypeSigRe = lazyregexp.New(`\b_C2?(type|func|var|macro)_\B`)

// run runs the command given by cmdline in the directory dir.
// If the command fails, run prints information about the failure
// and returns a non-nil error.
func (sh *Shell) run(dir string, desc string, env []string, cmdargs ...any) error {
	out, err := sh.runOut(dir, env, cmdargs...)
	if desc == "" {
		desc = sh.fmtCmd(dir, "%s", strings.Join(str.StringList(cmdargs...), " "))
	}
	return sh.reportCmd(desc, dir, out, err)
}

// runOut runs the command given by cmdline in the directory dir.
// It returns the command output and any errors that occurred.
// It accumulates execution time in a.
func (sh *Shell) runOut(dir string, env []string, cmdargs ...any) ([]byte, error) {
	a := sh.action

	cmdline := str.StringList(cmdargs...)

	for _, arg := range cmdline {
		// GNU binutils commands, including gcc and gccgo, interpret an argument
		// @foo anywhere in the command line (even following --) as meaning
		// "read and insert arguments from the file named foo."
		// Don't say anything that might be misinterpreted that way.
		if strings.HasPrefix(arg, "@") {
			return nil, fmt.Errorf("invalid command-line argument %s in command: %s", arg, joinUnambiguously(cmdline))
		}
	}

	if cfg.BuildN || cfg.BuildX {
		var envcmdline string
		for _, e := range env {
			if j := strings.IndexByte(e, '='); j != -1 {
				if strings.ContainsRune(e[j+1:], '\'') {
					envcmdline += fmt.Sprintf("%s=%q", e[:j], e[j+1:])
				} else {
					envcmdline += fmt.Sprintf("%s='%s'", e[:j], e[j+1:])
				}
				envcmdline += " "
			}
		}
		envcmdline += joinUnambiguously(cmdline)
		sh.ShowCmd(dir, "%s", envcmdline)
		if cfg.BuildN {
			return nil, nil
		}
	}

	var buf bytes.Buffer
	path, err := pathcache.LookPath(cmdline[0])
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(path, cmdline[1:]...)
	if cmd.Path != "" {
		cmd.Args[0] = cmd.Path
	}
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cleanup := passLongArgsInResponseFiles(cmd)
	defer cleanup()
	if dir != "." {
		cmd.Dir = dir
	}
	cmd.Env = cmd.Environ() // Pre-allocate with correct PWD.

	// Add the TOOLEXEC_IMPORTPATH environment variable for -toolexec tools.
	// It doesn't really matter if -toolexec isn't being used.
	// Note that a.Package.Desc is not really an import path,
	// but this is consistent with 'go list -f {{.ImportPath}}'.
	// Plus, it is useful to uniquely identify packages in 'go list -json'.
	if a != nil && a.Package != nil {
		cmd.Env = append(cmd.Env, "TOOLEXEC_IMPORTPATH="+a.Package.Desc())
	}

	cmd.Env = append(cmd.Env, env...)
	start := time.Now()
	err = cmd.Run()
	if a != nil && a.json != nil {
		aj := a.json
		aj.Cmd = append(aj.Cmd, joinUnambiguously(cmdline))
		aj.CmdReal += time.Since(start)
		if ps := cmd.ProcessState; ps != nil {
			aj.CmdUser += ps.UserTime()
			aj.CmdSys += ps.SystemTime()
		}
	}

	// err can be something like 'exit status 1'.
	// Add information about what program was running.
	// Note that if buf.Bytes() is non-empty, the caller usually
	// shows buf.Bytes() and does not print err at all, so the
	// prefix here does not make most output any more verbose.
	if err != nil {
		err = errors.New(cmdline[0] + ": " + err.Error())
	}
	return buf.Bytes(), err
}

// joinUnambiguously prints the slice, quoting where necessary to make the
// output unambiguous.
// TODO: See issue 5279. The printing of commands needs a complete redo.
func joinUnambiguously(a []string) string {
	var buf strings.Builder
	for i, s := range a {
		if i > 0 {
			buf.WriteByte(' ')
		}
		q := strconv.Quote(s)
		// A gccgo command line can contain -( and -).
		// Make sure we quote them since they are special to the shell.
		// The trimpath argument can also contain > (part of =>) and ;. Quote those too.
		if s == "" || strings.ContainsAny(s, " ()>;") || len(q) > len(s)+2 {
			buf.WriteString(q)
		} else {
			buf.WriteString(s)
		}
	}
	return buf.String()
}
