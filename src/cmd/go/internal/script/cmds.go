// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package script

import (
	"cmd/go/internal/cfg"
	"cmd/internal/robustio"
	"errors"
	"fmt"
	"internal/diff"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DefaultCmds returns a set of broadly useful script commands.
//
// Run the 'help' command within a script engine to view a list of the available
// commands.
func DefaultCmds() map[string]Cmd {
	return map[string]Cmd{
		"cat":     Cat(),
		"cd":      Cd(),
		"chmod":   Chmod(),
		"cmp":     Cmp(),
		"cmpenv":  Cmpenv(),
		"cp":      Cp(),
		"echo":    Echo(),
		"env":     Env(),
		"exec":    Exec(func(cmd *exec.Cmd) error { return cmd.Process.Signal(os.Interrupt) }, 100*time.Millisecond), // arbitrary grace period
		"exists":  Exists(),
		"grep":    Grep(),
		"help":    Help(),
		"mkdir":   Mkdir(),
		"mv":      Mv(),
		"rm":      Rm(),
		"replace": Replace(),
		"sleep":   Sleep(),
		"stderr":  Stderr(),
		"stdout":  Stdout(),
		"stop":    Stop(),
		"symlink": Symlink(),
		"wait":    Wait(),
	}
}

// Command returns a new Cmd with a Usage method that returns a copy of the
// given CmdUsage and a Run method calls the given function.
func Command(usage CmdUsage, run func(*State, ...string) (WaitFunc, error)) Cmd {
	return &funcCmd{
		usage: usage,
		run:   run,
	}
}

// A funcCmd implements Cmd using a function value.
type funcCmd struct {
	usage CmdUsage
	run   func(*State, ...string) (WaitFunc, error)
}

func (c *funcCmd) Run(s *State, args ...string) (WaitFunc, error) {
	return c.run(s, args...)
}

func (c *funcCmd) Usage() *CmdUsage { return &c.usage }

// firstNonFlag returns a slice containing the index of the first argument in
// rawArgs that is not a flag, or nil if all arguments are flags.
func firstNonFlag(rawArgs ...string) []int {
	for i, arg := range rawArgs {
		if !strings.HasPrefix(arg, "-") {
			return []int{i}
		}
		if arg == "--" {
			return []int{i + 1}
		}
	}
	return nil
}

// Cat writes the concatenated contents of the named file(s) to the script's
// stdout buffer.
func Cat() Cmd {
	return Command(
		CmdUsage{
			Summary: "concatenate files and print to the script's stdout buffer",
			Args:    "files...",
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) == 0 {
				return nil, ErrUsage
			}

			paths := make([]string, 0, len(args))
			for _, arg := range args {
				paths = append(paths, s.Path(arg))
			}

			var buf strings.Builder
			errc := make(chan error, 1)
			go func() {
				for _, p := range paths {
					b, err := os.ReadFile(p)
					buf.Write(b)
					if err != nil {
						errc <- err
						return
					}
				}
				errc <- nil
			}()

			wait := func(*State) (stdout, stderr string, err error) {
				err = <-errc
				return buf.String(), "", err
			}
			return wait, nil
		})
}

// Cd changes the current working directory.
func Cd() Cmd {
	return Command(
		CmdUsage{
			Summary: "change the working directory",
			Args:    "dir",
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) != 1 {
				return nil, ErrUsage
			}
			return nil, s.Chdir(args[0])
		})
}

// Chmod changes the permissions of a file or a directory..
func Chmod() Cmd {
	return Command(
		CmdUsage{
			Summary: "change file mode bits",
			Args:    "perm paths...",
			Detail: []string{
				"Changes the permissions of the named files or directories to be equal to perm.",
				"Only numerical permissions are supported.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) < 2 {
				return nil, ErrUsage
			}

			perm, err := strconv.ParseUint(args[0], 0, 32)
			if err != nil || perm&uint64(fs.ModePerm) != perm {
				return nil, fmt.Errorf("invalid mode: %s", args[0])
			}

			for _, arg := range args[1:] {
				err := os.Chmod(s.Path(arg), fs.FileMode(perm))
				if err != nil {
					return nil, err
				}
			}
			return nil, nil
		})
}

// Cmp compares the contents of two files, or the contents of either the
// "stdout" or "stderr" buffer and a file, returning a non-nil error if the
// contents differ.
func Cmp() Cmd {
	return Command(
		CmdUsage{
			Args:    "[-q] file1 file2",
			Summary: "compare files for differences",
			Detail: []string{
				"By convention, file1 is the actual data and file2 is the expected data.",
				"The command succeeds if the file contents are identical.",
				"File1 can be 'stdout' or 'stderr' to compare the stdout or stderr buffer from the most recent command.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			return nil, doCompare(s, false, args...)
		})
}

// Cmpenv is like Compare, but also performs environment substitutions
// on the contents of both arguments.
func Cmpenv() Cmd {
	return Command(
		CmdUsage{
			Args:    "[-q] file1 file2",
			Summary: "compare files for differences, with environment expansion",
			Detail: []string{
				"By convention, file1 is the actual data and file2 is the expected data.",
				"The command succeeds if the file contents are identical after substituting variables from the script environment.",
				"File1 can be 'stdout' or 'stderr' to compare the script's stdout or stderr buffer.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			return nil, doCompare(s, true, args...)
		})
}

func doCompare(s *State, env bool, args ...string) error {
	quiet := false
	if len(args) > 0 && args[0] == "-q" {
		quiet = true
		args = args[1:]
	}
	if len(args) != 2 {
		return ErrUsage
	}

	name1, name2 := args[0], args[1]
	var text1, text2 string
	switch name1 {
	case "stdout":
		text1 = s.Stdout()
	case "stderr":
		text1 = s.Stderr()
	default:
		data, err := os.ReadFile(s.Path(name1))
		if err != nil {
			return err
		}
		text1 = string(data)
	}

	data, err := os.ReadFile(s.Path(name2))
	if err != nil {
		return err
	}
	text2 = string(data)

	if env {
		text1 = s.ExpandEnv(text1, false)
		text2 = s.ExpandEnv(text2, false)
	}

	if text1 != text2 {
		if !quiet {
			diffText := diff.Diff(name1, []byte(text1), name2, []byte(text2))
			s.Logf("%s\n", diffText)
		}
		return fmt.Errorf("%s and %s differ", name1, name2)
	}
	return nil
}

// Cp copies one or more files to a new location.
func Cp() Cmd {
	return Command(
		CmdUsage{
			Summary: "copy files to a target file or directory",
			Args:    "src... dst",
			Detail: []string{
				"src can include 'stdout' or 'stderr' to copy from the script's stdout or stderr buffer.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) < 2 {
				return nil, ErrUsage
			}

			dst := s.Path(args[len(args)-1])
			info, err := os.Stat(dst)
			dstDir := err == nil && info.IsDir()
			if len(args) > 2 && !dstDir {
				return nil, &fs.PathError{Op: "cp", Path: dst, Err: errors.New("destination is not a directory")}
			}

			for _, arg := range args[:len(args)-1] {
				var (
					src  string
					data []byte
					mode fs.FileMode
				)
				switch arg {
				case "stdout":
					src = arg
					data = []byte(s.Stdout())
					mode = 0666
				case "stderr":
					src = arg
					data = []byte(s.Stderr())
					mode = 0666
				default:
					src = s.Path(arg)
					info, err := os.Stat(src)
					if err != nil {
						return nil, err
					}
					mode = info.Mode() & 0777
					data, err = os.ReadFile(src)
					if err != nil {
						return nil, err
					}
				}
				targ := dst
				if dstDir {
					targ = filepath.Join(dst, filepath.Base(src))
				}
				err := os.WriteFile(targ, data, mode)
				if err != nil {
					return nil, err
				}
			}

			return nil, nil
		})
}

// Echo writes its arguments to stdout, followed by a newline.
func Echo() Cmd {
	return Command(
		CmdUsage{
			Summary: "display a line of text",
			Args:    "string...",
		},
		func(s *State, args ...string) (WaitFunc, error) {
			var buf strings.Builder
			for i, arg := range args {
				if i > 0 {
					buf.WriteString(" ")
				}
				buf.WriteString(arg)
			}
			buf.WriteString("\n")
			out := buf.String()

			// Stuff the result into a callback to satisfy the OutputCommandFunc
			// interface, even though it isn't really asynchronous even if run in the
			// background.
			//
			// Nobody should be running 'echo' as a background command, but it's not worth
			// defining yet another interface, and also doesn't seem worth shoehorning
			// into a SimpleCommand the way we did with Wait.
			return func(*State) (stdout, stderr string, err error) {
				return out, "", nil
			}, nil
		})
}

// Env sets or logs the values of environment variables.
//
// With no arguments, Env reports all variables in the environment.
// "key=value" arguments set variables, and arguments without "="
// cause the corresponding value to be printed to the stdout buffer.
func Env() Cmd {
	return Command(
		CmdUsage{
			Summary: "set or log the values of environment variables",
			Args:    "[key[=value]...]",
			Detail: []string{
				"With no arguments, print the script environment to the log.",
				"Otherwise, add the listed key=value pairs to the environment or print the listed keys.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			out := new(strings.Builder)
			if len(args) == 0 {
				for _, kv := range s.env {
					fmt.Fprintf(out, "%s\n", kv)
				}
			} else {
				for _, env := range args {
					i := strings.Index(env, "=")
					if i < 0 {
						// Display value instead of setting it.
						fmt.Fprintf(out, "%s=%s\n", env, s.envMap[env])
						continue
					}
					if err := s.Setenv(env[:i], env[i+1:]); err != nil {
						return nil, err
					}
				}
			}
			var wait WaitFunc
			if out.Len() > 0 || len(args) == 0 {
				wait = func(*State) (stdout, stderr string, err error) {
					return out.String(), "", nil
				}
			}
			return wait, nil
		})
}

// Exec runs an arbitrary executable as a subprocess.
//
// When the Script's context is canceled, Exec sends the interrupt signal, then
// waits for up to the given delay for the subprocess to flush output before
// terminating it with os.Kill.
func Exec(cancel func(*exec.Cmd) error, waitDelay time.Duration) Cmd {
	return Command(
		CmdUsage{
			Summary: "run an executable program with arguments",
			Args:    "program [args...]",
			Detail: []string{
				"Note that 'exec' does not terminate the script (unlike Unix shells).",
			},
			Async: true,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) < 1 {
				return nil, ErrUsage
			}

			// Use the script's PATH to look up the command (if it does not contain a separator)
			// instead of the test process's PATH (see lookPath).
			// Don't use filepath.Clean, since that changes "./foo" to "foo".
			name := filepath.FromSlash(args[0])
			path := name
			if !strings.Contains(name, string(filepath.Separator)) {
				var err error
				path, err = lookPath(s, name)
				if err != nil {
					return nil, err
				}
			}

			return startCommand(s, name, path, args[1:], cancel, waitDelay)
		})
}

func startCommand(s *State, name, path string, args []string, cancel func(*exec.Cmd) error, waitDelay time.Duration) (WaitFunc, error) {
	var (
		cmd                  *exec.Cmd
		stdoutBuf, stderrBuf strings.Builder
	)
	for {
		cmd = exec.CommandContext(s.Context(), path, args...)
		if cancel == nil {
			cmd.Cancel = nil
		} else {
			cmd.Cancel = func() error { return cancel(cmd) }
		}
		cmd.WaitDelay = waitDelay
		cmd.Args[0] = name
		cmd.Dir = s.Getwd()
		cmd.Env = s.env
		cmd.Stdout = &stdoutBuf
		cmd.Stderr = &stderrBuf
		err := cmd.Start()
		if err == nil {
			break
		}
		if isETXTBSY(err) {
			// If the script (or its host process) just wrote the executable we're
			// trying to run, a fork+exec in another thread may be holding open the FD
			// that we used to write the executable (see https://go.dev/issue/22315).
			// Since the descriptor should have CLOEXEC set, the problem should
			// resolve as soon as the forked child reaches its exec call.
			// Keep retrying until that happens.
		} else {
			return nil, err
		}
	}

	wait := func(s *State) (stdout, stderr string, err error) {
		err = cmd.Wait()
		return stdoutBuf.String(), stderrBuf.String(), err
	}
	return wait, nil
}

// lookPath is (roughly) like exec.LookPath, but it uses the script's current
// PATH to find the executable.
func lookPath(s *State, command string) (string, error) {
	var strEqual func(string, string) bool
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		// Using GOOS as a proxy for case-insensitive file system.
		// TODO(bcmills): Remove this assumption.
		strEqual = strings.EqualFold
	} else {
		strEqual = func(a, b string) bool { return a == b }
	}

	var pathExt []string
	var searchExt bool
	var isExecutable func(os.FileInfo) bool
	if runtime.GOOS == "windows" {
		// Use the test process's PathExt instead of the script's.
		// If PathExt is set in the command's environment, cmd.Start fails with
		// "parameter is invalid". Not sure why.
		// If the command already has an extension in PathExt (like "cmd.exe")
		// don't search for other extensions (not "cmd.bat.exe").
		pathExt = strings.Split(os.Getenv("PathExt"), string(filepath.ListSeparator))
		searchExt = true
		cmdExt := filepath.Ext(command)
		for _, ext := range pathExt {
			if strEqual(cmdExt, ext) {
				searchExt = false
				break
			}
		}
		isExecutable = func(fi os.FileInfo) bool {
			return fi.Mode().IsRegular()
		}
	} else {
		isExecutable = func(fi os.FileInfo) bool {
			return fi.Mode().IsRegular() && fi.Mode().Perm()&0111 != 0
		}
	}

	pathEnv, _ := s.LookupEnv(pathEnvName())
	for _, dir := range strings.Split(pathEnv, string(filepath.ListSeparator)) {
		if dir == "" {
			continue
		}

		// Determine whether dir needs a trailing path separator.
		// Note: we avoid filepath.Join in this function because it cleans the
		// result: we want to preserve the exact dir prefix from the environment.
		sep := string(filepath.Separator)
		if os.IsPathSeparator(dir[len(dir)-1]) {
			sep = ""
		}

		if searchExt {
			ents, err := os.ReadDir(dir)
			if err != nil {
				continue
			}
			for _, ent := range ents {
				for _, ext := range pathExt {
					if !ent.IsDir() && strEqual(ent.Name(), command+ext) {
						return dir + sep + ent.Name(), nil
					}
				}
			}
		} else {
			path := dir + sep + command
			if fi, err := os.Stat(path); err == nil && isExecutable(fi) {
				return path, nil
			}
		}
	}
	return "", &exec.Error{Name: command, Err: exec.ErrNotFound}
}

// pathEnvName returns the platform-specific variable used by os/exec.LookPath
// to look up executable names (either "PATH" or "path").
//
// TODO(bcmills): Investigate whether we can instead use PATH uniformly and
// rewrite it to $path when executing subprocesses.
func pathEnvName() string {
	switch runtime.GOOS {
	case "plan9":
		return "path"
	default:
		return "PATH"
	}
}

// Exists checks that the named file(s) exist.
func Exists() Cmd {
	return Command(
		CmdUsage{
			Summary: "check that files exist",
			Args:    "[-readonly] [-exec] file...",
		},
		func(s *State, args ...string) (WaitFunc, error) {
			var readonly, exec bool
		loop:
			for len(args) > 0 {
				switch args[0] {
				case "-readonly":
					readonly = true
					args = args[1:]
				case "-exec":
					exec = true
					args = args[1:]
				default:
					break loop
				}
			}
			if len(args) == 0 {
				return nil, ErrUsage
			}

			for _, file := range args {
				file = s.Path(file)
				info, err := os.Stat(file)
				if err != nil {
					return nil, err
				}
				if readonly && info.Mode()&0222 != 0 {
					return nil, fmt.Errorf("%s exists but is writable", file)
				}
				if exec && runtime.GOOS != "windows" && info.Mode()&0111 == 0 {
					return nil, fmt.Errorf("%s exists but is not executable", file)
				}
			}

			return nil, nil
		})
}

// Grep checks that file content matches a regexp.
// Like stdout/stderr and unlike Unix grep, it accepts Go regexp syntax.
//
// Grep does not modify the State's stdout or stderr buffers.
// (Its output goes to the script log, not stdout.)
func Grep() Cmd {
	return Command(
		CmdUsage{
			Summary: "find lines in a file that match a pattern",
			Args:    matchUsage + " file",
			Detail: []string{
				"The command succeeds if at least one match (or the exact count, if given) is found.",
				"The -q flag suppresses printing of matches.",
			},
			RegexpArgs: firstNonFlag,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			return nil, match(s, args, "", "grep")
		})
}

const matchUsage = "[-count=N] [-q] 'pattern'"

// match implements the Grep, Stdout, and Stderr commands.
func match(s *State, args []string, text, name string) error {
	n := 0
	if len(args) >= 1 && strings.HasPrefix(args[0], "-count=") {
		var err error
		n, err = strconv.Atoi(args[0][len("-count="):])
		if err != nil {
			return fmt.Errorf("bad -count=: %v", err)
		}
		if n < 1 {
			return fmt.Errorf("bad -count=: must be at least 1")
		}
		args = args[1:]
	}
	quiet := false
	if len(args) >= 1 && args[0] == "-q" {
		quiet = true
		args = args[1:]
	}

	isGrep := name == "grep"

	wantArgs := 1
	if isGrep {
		wantArgs = 2
	}
	if len(args) != wantArgs {
		return ErrUsage
	}

	pattern := `(?m)` + args[0]
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}

	if isGrep {
		name = args[1] // for error messages
		data, err := os.ReadFile(s.Path(args[1]))
		if err != nil {
			return err
		}
		text = string(data)
	}

	if n > 0 {
		count := len(re.FindAllString(text, -1))
		if count != n {
			return fmt.Errorf("found %d matches for %#q in %s", count, pattern, name)
		}
		return nil
	}

	if !re.MatchString(text) {
		return fmt.Errorf("no match for %#q in %s", pattern, name)
	}

	if !quiet {
		// Print the lines containing the match.
		loc := re.FindStringIndex(text)
		for loc[0] > 0 && text[loc[0]-1] != '\n' {
			loc[0]--
		}
		for loc[1] < len(text) && text[loc[1]] != '\n' {
			loc[1]++
		}
		lines := strings.TrimSuffix(text[loc[0]:loc[1]], "\n")
		s.Logf("matched: %s\n", lines)
	}
	return nil
}

// Help writes command documentation to the script log.
func Help() Cmd {
	return Command(
		CmdUsage{
			Summary: "log help text for commands and conditions",
			Args:    "[-v] name...",
			Detail: []string{
				"To display help for a specific condition, enclose it in brackets: 'help [amd64]'.",
				"To display complete documentation when listing all commands, pass the -v flag.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if s.engine == nil {
				return nil, errors.New("no engine configured")
			}

			verbose := false
			if len(args) > 0 {
				verbose = true
				if args[0] == "-v" {
					args = args[1:]
				}
			}

			var cmds, conds []string
			for _, arg := range args {
				if strings.HasPrefix(arg, "[") && strings.HasSuffix(arg, "]") {
					conds = append(conds, arg[1:len(arg)-1])
				} else {
					cmds = append(cmds, arg)
				}
			}

			out := new(strings.Builder)

			if len(conds) > 0 || (len(args) == 0 && len(s.engine.Conds) > 0) {
				if conds == nil {
					out.WriteString("conditions:\n\n")
				}
				s.engine.ListConds(out, s, conds...)
			}

			if len(cmds) > 0 || len(args) == 0 {
				if len(args) == 0 {
					out.WriteString("\ncommands:\n\n")
				}
				s.engine.ListCmds(out, verbose, cmds...)
			}

			wait := func(*State) (stdout, stderr string, err error) {
				return out.String(), "", nil
			}
			return wait, nil
		})
}

// Mkdir creates a directory and any needed parent directories.
func Mkdir() Cmd {
	return Command(
		CmdUsage{
			Summary: "create directories, if they do not already exist",
			Args:    "path...",
			Detail: []string{
				"Unlike Unix mkdir, parent directories are always created if needed.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) < 1 {
				return nil, ErrUsage
			}
			for _, arg := range args {
				if err := os.MkdirAll(s.Path(arg), 0777); err != nil {
					return nil, err
				}
			}
			return nil, nil
		})
}

// Mv renames an existing file or directory to a new path.
func Mv() Cmd {
	return Command(
		CmdUsage{
			Summary: "rename a file or directory to a new path",
			Args:    "old new",
			Detail: []string{
				"OS-specific restrictions may apply when old and new are in different directories.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) != 2 {
				return nil, ErrUsage
			}
			return nil, os.Rename(s.Path(args[0]), s.Path(args[1]))
		})
}

// Program returns a new command that runs the named program, found from the
// host process's PATH (not looked up in the script's PATH).
func Program(name string, cancel func(*exec.Cmd) error, waitDelay time.Duration) Cmd {
	var (
		shortName    string
		summary      string
		lookPathOnce sync.Once
		path         string
		pathErr      error
	)
	if filepath.IsAbs(name) {
		lookPathOnce.Do(func() { path = filepath.Clean(name) })
		shortName = strings.TrimSuffix(filepath.Base(path), ".exe")
		summary = "run the '" + shortName + "' program provided by the script host"
	} else {
		shortName = name
		summary = "run the '" + shortName + "' program from the script host's PATH"
	}

	return Command(
		CmdUsage{
			Summary: summary,
			Args:    "[args...]",
			Async:   true,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			lookPathOnce.Do(func() {
				path, pathErr = cfg.LookPath(name)
			})
			if pathErr != nil {
				return nil, pathErr
			}
			return startCommand(s, shortName, path, args, cancel, waitDelay)
		})
}

// Replace replaces all occurrences of a string in a file with another string.
func Replace() Cmd {
	return Command(
		CmdUsage{
			Summary: "replace strings in a file",
			Args:    "[old new]... file",
			Detail: []string{
				"The 'old' and 'new' arguments are unquoted as if in quoted Go strings.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args)%2 != 1 {
				return nil, ErrUsage
			}

			oldNew := make([]string, 0, len(args)-1)
			for _, arg := range args[:len(args)-1] {
				s, err := strconv.Unquote(`"` + arg + `"`)
				if err != nil {
					return nil, err
				}
				oldNew = append(oldNew, s)
			}

			r := strings.NewReplacer(oldNew...)
			file := s.Path(args[len(args)-1])

			data, err := os.ReadFile(file)
			if err != nil {
				return nil, err
			}
			replaced := r.Replace(string(data))

			return nil, os.WriteFile(file, []byte(replaced), 0666)
		})
}

// Rm removes a file or directory.
//
// If a directory, Rm also recursively removes that directory's
// contents.
func Rm() Cmd {
	return Command(
		CmdUsage{
			Summary: "remove a file or directory",
			Args:    "path...",
			Detail: []string{
				"If the path is a directory, its contents are removed recursively.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) < 1 {
				return nil, ErrUsage
			}
			for _, arg := range args {
				if err := removeAll(s.Path(arg)); err != nil {
					return nil, err
				}
			}
			return nil, nil
		})
}

// removeAll removes dir and all files and directories it contains.
//
// Unlike os.RemoveAll, removeAll attempts to make the directories writable if
// needed in order to remove their contents.
func removeAll(dir string) error {
	// module cache has 0444 directories;
	// make them writable in order to remove content.
	filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
		// chmod not only directories, but also things that we couldn't even stat
		// due to permission errors: they may also be unreadable directories.
		if err != nil || info.IsDir() {
			os.Chmod(path, 0777)
		}
		return nil
	})
	return robustio.RemoveAll(dir)
}

// Sleep sleeps for the given Go duration or until the script's context is
// canceled, whichever happens first.
func Sleep() Cmd {
	return Command(
		CmdUsage{
			Summary: "sleep for a specified duration",
			Args:    "duration",
			Detail: []string{
				"The duration must be given as a Go time.Duration string.",
			},
			Async: true,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) != 1 {
				return nil, ErrUsage
			}

			d, err := time.ParseDuration(args[0])
			if err != nil {
				return nil, err
			}

			timer := time.NewTimer(d)
			wait := func(s *State) (stdout, stderr string, err error) {
				ctx := s.Context()
				select {
				case <-ctx.Done():
					timer.Stop()
					return "", "", ctx.Err()
				case <-timer.C:
					return "", "", nil
				}
			}
			return wait, nil
		})
}

// Stderr searches for a regular expression in the stderr buffer.
func Stderr() Cmd {
	return Command(
		CmdUsage{
			Summary: "find lines in the stderr buffer that match a pattern",
			Args:    matchUsage + " file",
			Detail: []string{
				"The command succeeds if at least one match (or the exact count, if given) is found.",
				"The -q flag suppresses printing of matches.",
			},
			RegexpArgs: firstNonFlag,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			return nil, match(s, args, s.Stderr(), "stderr")
		})
}

// Stdout searches for a regular expression in the stdout buffer.
func Stdout() Cmd {
	return Command(
		CmdUsage{
			Summary: "find lines in the stdout buffer that match a pattern",
			Args:    matchUsage + " file",
			Detail: []string{
				"The command succeeds if at least one match (or the exact count, if given) is found.",
				"The -q flag suppresses printing of matches.",
			},
			RegexpArgs: firstNonFlag,
		},
		func(s *State, args ...string) (WaitFunc, error) {
			return nil, match(s, args, s.Stdout(), "stdout")
		})
}

// Stop returns a sentinel error that causes script execution to halt
// and s.Execute to return with a nil error.
func Stop() Cmd {
	return Command(
		CmdUsage{
			Summary: "stop execution of the script",
			Args:    "[msg]",
			Detail: []string{
				"The message is written to the script log, but no error is reported from the script engine.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) > 1 {
				return nil, ErrUsage
			}
			// TODO(bcmills): The argument passed to stop seems redundant with comments.
			// Either use it systematically or remove it.
			if len(args) == 1 {
				return nil, stopError{msg: args[0]}
			}
			return nil, stopError{}
		})
}

// stopError is the sentinel error type returned by the Stop command.
type stopError struct {
	msg string
}

func (s stopError) Error() string {
	if s.msg == "" {
		return "stop"
	}
	return "stop: " + s.msg
}

// Symlink creates a symbolic link.
func Symlink() Cmd {
	return Command(
		CmdUsage{
			Summary: "create a symlink",
			Args:    "path -> target",
			Detail: []string{
				"Creates path as a symlink to target.",
				"The '->' token (like in 'ls -l' output on Unix) is required.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) != 3 || args[1] != "->" {
				return nil, ErrUsage
			}

			// Note that the link target args[2] is not interpreted with s.Path:
			// it will be interpreted relative to the directory file is in.
			return nil, os.Symlink(filepath.FromSlash(args[2]), s.Path(args[0]))
		})
}

// Wait waits for the completion of background commands.
//
// When Wait returns, the stdout and stderr buffers contain the concatenation of
// the background commands' respective outputs in the order in which those
// commands were started.
func Wait() Cmd {
	return Command(
		CmdUsage{
			Summary: "wait for completion of background commands",
			Args:    "",
			Detail: []string{
				"Waits for all background commands to complete.",
				"The output (and any error) from each command is printed to the log in the order in which the commands were started.",
				"After the call to 'wait', the script's stdout and stderr buffers contain the concatenation of the background commands' outputs.",
			},
		},
		func(s *State, args ...string) (WaitFunc, error) {
			if len(args) > 0 {
				return nil, ErrUsage
			}

			var stdouts, stderrs []string
			var errs []*CommandError
			for _, bg := range s.background {
				stdout, stderr, err := bg.wait(s)

				beforeArgs := ""
				if len(bg.args) > 0 {
					beforeArgs = " "
				}
				s.Logf("[background] %s%s%s\n", bg.name, beforeArgs, quoteArgs(bg.args))

				if stdout != "" {
					s.Logf("[stdout]\n%s", stdout)
					stdouts = append(stdouts, stdout)
				}
				if stderr != "" {
					s.Logf("[stderr]\n%s", stderr)
					stderrs = append(stderrs, stderr)
				}
				if err != nil {
					s.Logf("[%v]\n", err)
				}
				if cmdErr := checkStatus(bg.command, err); cmdErr != nil {
					errs = append(errs, cmdErr.(*CommandError))
				}
			}

			s.stdout = strings.Join(stdouts, "")
			s.stderr = strings.Join(stderrs, "")
			s.background = nil
			if len(errs) > 0 {
				return nil, waitError{errs: errs}
			}
			return nil, nil
		})
}

// A waitError wraps one or more errors returned by background commands.
type waitError struct {
	errs []*CommandError
}

func (w waitError) Error() string {
	b := new(strings.Builder)
	for i, err := range w.errs {
		if i != 0 {
			b.WriteString("\n")
		}
		b.WriteString(err.Error())
	}
	return b.String()
}

func (w waitError) Unwrap() error {
	if len(w.errs) == 1 {
		return w.errs[0]
	}
	return nil
}
