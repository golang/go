// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
)

func TestRemoveDevNull(t *testing.T) {
	fi, err := os.Lstat(os.DevNull)
	if err != nil {
		t.Skip(err)
	}
	if fi.Mode().IsRegular() {
		t.Errorf("Lstat(%s).Mode().IsRegular() = true; expected false", os.DevNull)
	}
	mayberemovefile(os.DevNull)
	_, err = os.Lstat(os.DevNull)
	if err != nil {
		t.Errorf("mayberemovefile(%s) did remove it; oops", os.DevNull)
	}
}

func TestSplitPkgConfigOutput(t *testing.T) {
	for _, test := range []struct {
		in   []byte
		want []string
	}{
		{[]byte(`-r:foo -L/usr/white\ space/lib -lfoo\ bar -lbar\ baz`), []string{"-r:foo", "-L/usr/white space/lib", "-lfoo bar", "-lbar baz"}},
		{[]byte(`-lextra\ fun\ arg\\`), []string{`-lextra fun arg\`}},
		{[]byte("\textra     whitespace\r\n"), []string{"extra", "whitespace\r"}},
		{[]byte("     \r\n      "), []string{"\r"}},
		{[]byte(`"-r:foo" "-L/usr/white space/lib" "-lfoo bar" "-lbar baz"`), []string{"-r:foo", "-L/usr/white space/lib", "-lfoo bar", "-lbar baz"}},
		{[]byte(`"-lextra fun arg\\"`), []string{`-lextra fun arg\`}},
		{[]byte(`"     \r\n\      "`), []string{`     \r\n\      `}},
		{[]byte(`""`), []string{""}},
		{[]byte(``), nil},
		{[]byte(`"\\"`), []string{`\`}},
		{[]byte(`"\x"`), []string{`\x`}},
		{[]byte(`"\\x"`), []string{`\x`}},
		{[]byte(`'\\'`), []string{`\\`}},
		{[]byte(`'\x'`), []string{`\x`}},
		{[]byte(`"\\x"`), []string{`\x`}},
		{[]byte("\\\n"), nil},
		{[]byte(`-fPIC -I/test/include/foo -DQUOTED='"/test/share/doc"'`), []string{"-fPIC", "-I/test/include/foo", `-DQUOTED="/test/share/doc"`}},
		{[]byte(`-fPIC -I/test/include/foo -DQUOTED="/test/share/doc"`), []string{"-fPIC", "-I/test/include/foo", "-DQUOTED=/test/share/doc"}},
		{[]byte(`-fPIC -I/test/include/foo -DQUOTED=\"/test/share/doc\"`), []string{"-fPIC", "-I/test/include/foo", `-DQUOTED="/test/share/doc"`}},
		{[]byte(`-fPIC -I/test/include/foo -DQUOTED='/test/share/doc'`), []string{"-fPIC", "-I/test/include/foo", "-DQUOTED=/test/share/doc"}},
		{[]byte(`-DQUOTED='/te\st/share/d\oc'`), []string{`-DQUOTED=/te\st/share/d\oc`}},
		{[]byte(`-Dhello=10 -Dworld=+32 -DDEFINED_FROM_PKG_CONFIG=hello\ world`), []string{"-Dhello=10", "-Dworld=+32", "-DDEFINED_FROM_PKG_CONFIG=hello world"}},
		{[]byte(`"broken\"" \\\a "a"`), []string{"broken\"", "\\a", "a"}},
	} {
		got, err := splitPkgConfigOutput(test.in)
		if err != nil {
			t.Errorf("splitPkgConfigOutput on %#q failed with error %v", test.in, err)
			continue
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("splitPkgConfigOutput(%#q) = %#q; want %#q", test.in, got, test.want)
		}
	}

	for _, test := range []struct {
		in   []byte
		want []string
	}{
		// broken quotation
		{[]byte(`"     \r\n      `), nil},
		{[]byte(`"-r:foo" "-L/usr/white space/lib "-lfoo bar" "-lbar baz"`), nil},
		{[]byte(`"-lextra fun arg\\`), nil},
		// broken char escaping
		{[]byte(`broken flag\`), nil},
		{[]byte(`extra broken flag \`), nil},
		{[]byte(`\`), nil},
		{[]byte(`"broken\"" "extra" \`), nil},
	} {
		got, err := splitPkgConfigOutput(test.in)
		if err == nil {
			t.Errorf("splitPkgConfigOutput(%v) = %v; haven't failed with error as expected.", test.in, got)
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("splitPkgConfigOutput(%v) = %v; want %v", test.in, got, test.want)
		}
	}

}

func TestSharedLibName(t *testing.T) {
	// TODO(avdva) - make these values platform-specific
	prefix := "lib"
	suffix := ".so"
	testData := []struct {
		args      []string
		pkgs      []*load.Package
		expected  string
		expectErr bool
		rootedAt  string
	}{
		{
			args:     []string{"std"},
			pkgs:     []*load.Package{},
			expected: "std",
		},
		{
			args:     []string{"std", "cmd"},
			pkgs:     []*load.Package{},
			expected: "std,cmd",
		},
		{
			args:     []string{},
			pkgs:     []*load.Package{pkgImportPath("gopkg.in/somelib")},
			expected: "gopkg.in-somelib",
		},
		{
			args:     []string{"./..."},
			pkgs:     []*load.Package{pkgImportPath("somelib")},
			expected: "somelib",
			rootedAt: "somelib",
		},
		{
			args:     []string{"../somelib", "../somelib"},
			pkgs:     []*load.Package{pkgImportPath("somelib")},
			expected: "somelib",
		},
		{
			args:     []string{"../lib1", "../lib2"},
			pkgs:     []*load.Package{pkgImportPath("gopkg.in/lib1"), pkgImportPath("gopkg.in/lib2")},
			expected: "gopkg.in-lib1,gopkg.in-lib2",
		},
		{
			args: []string{"./..."},
			pkgs: []*load.Package{
				pkgImportPath("gopkg.in/dir/lib1"),
				pkgImportPath("gopkg.in/lib2"),
				pkgImportPath("gopkg.in/lib3"),
			},
			expected: "gopkg.in",
			rootedAt: "gopkg.in",
		},
		{
			args:      []string{"std", "../lib2"},
			pkgs:      []*load.Package{},
			expectErr: true,
		},
		{
			args:      []string{"all", "./"},
			pkgs:      []*load.Package{},
			expectErr: true,
		},
		{
			args:      []string{"cmd", "fmt"},
			pkgs:      []*load.Package{},
			expectErr: true,
		},
	}
	for _, data := range testData {
		func() {
			if data.rootedAt != "" {
				tmpGopath, err := os.MkdirTemp("", "gopath")
				if err != nil {
					t.Fatal(err)
				}
				cwd := base.Cwd()
				oldGopath := cfg.BuildContext.GOPATH
				defer func() {
					cfg.BuildContext.GOPATH = oldGopath
					os.Chdir(cwd)
					err := os.RemoveAll(tmpGopath)
					if err != nil {
						t.Error(err)
					}
				}()
				root := filepath.Join(tmpGopath, "src", data.rootedAt)
				err = os.MkdirAll(root, 0755)
				if err != nil {
					t.Fatal(err)
				}
				cfg.BuildContext.GOPATH = tmpGopath
				os.Chdir(root)
			}
			computed, err := libname(data.args, data.pkgs)
			if err != nil {
				if !data.expectErr {
					t.Errorf("libname returned an error %q, expected a name", err.Error())
				}
			} else if data.expectErr {
				t.Errorf("libname returned %q, expected an error", computed)
			} else {
				expected := prefix + data.expected + suffix
				if expected != computed {
					t.Errorf("libname returned %q, expected %q", computed, expected)
				}
			}
		}()
	}
}

func pkgImportPath(pkgpath string) *load.Package {
	return &load.Package{
		PackagePublic: load.PackagePublic{
			ImportPath: pkgpath,
		},
	}
}

// When installing packages, the installed package directory should
// respect the SetGID bit and group name of the destination
// directory.
// See https://golang.org/issue/18878.
func TestRespectSetgidDir(t *testing.T) {
	// Check that `cp` is called instead of `mv` by looking at the output
	// of `(*Shell).ShowCmd` afterwards as a sanity check.
	cfg.BuildX = true
	var cmdBuf strings.Builder
	sh := NewShell("", func { a -> cmdBuf.WriteString(fmt.Sprint(a...)) })

	setgiddir, err := os.MkdirTemp("", "SetGroupID")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(setgiddir)

	// BSD mkdir(2) inherits the parent directory group, and other platforms
	// can inherit the parent directory group via setgid. The test setup (chmod
	// setgid) will fail if the process does not have the group permission to
	// the new temporary directory.
	err = os.Chown(setgiddir, os.Getuid(), os.Getgid())
	if err != nil {
		if testenv.SyscallIsNotSupported(err) {
			t.Skip("skipping: chown is not supported on " + runtime.GOOS)
		}
		t.Fatal(err)
	}

	// Change setgiddir's permissions to include the SetGID bit.
	if err := os.Chmod(setgiddir, 0755|fs.ModeSetgid); err != nil {
		if testenv.SyscallIsNotSupported(err) {
			t.Skip("skipping: chmod is not supported on " + runtime.GOOS)
		}
		t.Fatal(err)
	}
	if fi, err := os.Stat(setgiddir); err != nil {
		t.Fatal(err)
	} else if fi.Mode()&fs.ModeSetgid == 0 {
		t.Skip("skipping: Chmod ignored ModeSetgid on " + runtime.GOOS)
	}

	pkgfile, err := os.CreateTemp("", "pkgfile")
	if err != nil {
		t.Fatalf("os.CreateTemp(\"\", \"pkgfile\"): %v", err)
	}
	defer os.Remove(pkgfile.Name())
	defer pkgfile.Close()

	dirGIDFile := filepath.Join(setgiddir, "setgid")
	if err := sh.moveOrCopyFile(dirGIDFile, pkgfile.Name(), 0666, true); err != nil {
		t.Fatalf("moveOrCopyFile: %v", err)
	}

	got := strings.TrimSpace(cmdBuf.String())
	want := sh.fmtCmd("", "cp %s %s", pkgfile.Name(), dirGIDFile)
	if got != want {
		t.Fatalf("moveOrCopyFile(%q, %q): want %q, got %q", dirGIDFile, pkgfile.Name(), want, got)
	}
}
