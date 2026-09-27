// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

var goodCompilerFlags = [][]string{
	{"-DFOO"},
	{"-Dfoo=bar"},
	{"-Ufoo"},
	{"-Ufoo1"},
	{"-F/Qt"},
	{"-F", "/Qt"},
	{"-I/"},
	{"-I/etc/passwd"},
	{"-I."},
	{"-O"},
	{"-O2"},
	{"-Osmall"},
	{"-W"},
	{"-Wall"},
	{"-Wp,-Dfoo=bar"},
	{"-Wp,-Ufoo"},
	{"-Wp,-Dfoo1"},
	{"-Wp,-Ufoo1"},
	{"-flto"},
	{"-fobjc-arc"},
	{"-fno-objc-arc"},
	{"-fomit-frame-pointer"},
	{"-fno-omit-frame-pointer"},
	{"-fpic"},
	{"-fno-pic"},
	{"-fPIC"},
	{"-fno-PIC"},
	{"-fpie"},
	{"-fno-pie"},
	{"-fPIE"},
	{"-fno-PIE"},
	{"-fsplit-stack"},
	{"-fno-split-stack"},
	{"-fstack-xxx"},
	{"-fno-stack-xxx"},
	{"-fsanitize=hands"},
	{"-ftls-model=local-dynamic"},
	{"-g"},
	{"-ggdb"},
	{"-mabi=lp64d"},
	{"-march=souza"},
	{"-mcmodel=medium"},
	{"-mcpu=123"},
	{"-mfpu=123"},
	{"-mtls-dialect=gnu"},
	{"-mtls-dialect=gnu2"},
	{"-mtls-dialect=trad"},
	{"-mtls-dialect=desc"},
	{"-mtls-dialect=xyz"},
	{"-msimd=lasx"},
	{"-msimd=xyz"},
	{"-mdouble-float"},
	{"-mrelax"},
	{"-mstrict-align"},
	{"-mlsx"},
	{"-mlasx"},
	{"-mfrecipe"},
	{"-mlam-bh"},
	{"-mlamcas"},
	{"-mld-seq-sa"},
	{"-mno-relax"},
	{"-mno-strict-align"},
	{"-mno-lsx"},
	{"-mno-lasx"},
	{"-mno-frecipe"},
	{"-mno-lam-bh"},
	{"-mno-lamcas"},
	{"-mno-ld-seq-sa"},
	{"-mlarge-data-threshold=16"},
	{"-mtune=happybirthday"},
	{"-mstack-overflow"},
	{"-mno-stack-overflow"},
	{"-mmacosx-version"},
	{"-mnop-fun-dllimport"},
	{"-pthread"},
	{"-std=c99"},
	{"-xc"},
	{"-D", "FOO"},
	{"-D", "foo=bar"},
	{"-I", "."},
	{"-I", "/etc/passwd"},
	{"-I", "世界"},
	{"-I", "=/usr/include/libxml2"},
	{"-I", "dir"},
	{"-I", "$SYSROOT/dir"},
	{"-isystem", "/usr/include/mozjs-68"},
	{"-include", "/usr/include/mozjs-68/RequiredDefines.h"},
	{"-framework", "Chocolate"},
	{"-isysroot", "/tmp"},
	{"-x", "c"},
	{"-v"},
}

var badCompilerFlags = [][]string{
	{"-D@X"},
	{"-D-X"},
	{"-Ufoo=bar"},
	{"-F@dir"},
	{"-F-dir"},
	{"-I@dir"},
	{"-I-dir"},
	{"-O@1"},
	{"-Wa,-foo"},
	{"-W@foo"},
	{"-Wp,-DX,-D@X"},
	{"-Wp,-UX,-U@X"},
	{"-g@gdb"},
	{"-g-gdb"},
	{"-march=@dawn"},
	{"-march=-dawn"},
	{"-mcmodel=@model"},
	{"-mfpu=@0"},
	{"-mfpu=-0"},
	{"-mlarge-data-threshold=@12"},
	{"-mtls-dialect=@gnu"},
	{"-mtls-dialect=-gnu"},
	{"-msimd=@none"},
	{"-msimd=-none"},
	{"-std=@c99"},
	{"-std=-c99"},
	{"-x@c"},
	{"-x-c"},
	{"-D", "@foo"},
	{"-D", "-foo"},
	{"-I", "@foo"},
	{"-I", "-foo"},
	{"-I", "=@obj"},
	{"-include", "@foo"},
	{"-framework", "-Caffeine"},
	{"-framework", "@Home"},
	{"-x", "--c"},
	{"-x", "@obj"},
	{"-xada"},
	{"-x", "ada"},
	{"-x", "none"},
	{"--sysroot=/tmp"},
	{"--sysroot", "/tmp"},
	{"-target", "x86_64-scei-ps4"},
	{"-target", "x86_64-linux-gnu"},
	{"-fsanitize=@foo"},
	{"-fsanitize=-foo"},
	{"-gnatO"},
	{"-gnatO", "foo.o"},
	{"-gen-cdb-fragment-path"},
	{"-gen-cdb-fragment-path", "."},
}

func TestCheckCompilerFlags(t *testing.T) {
	for _, f := range goodCompilerFlags {
		if err := checkCompilerFlags("test", "test", f); err != nil {
			t.Errorf("unexpected error for %q: %v", f, err)
		}
	}
	for _, f := range badCompilerFlags {
		if err := checkCompilerFlags("test", "test", f); err == nil {
			t.Errorf("missing error for %q", f)
		}
	}
}

var goodLinkerFlags = [][]string{
	{"-Fbar"},
	{"-lbar"},
	{"-Lbar"},
	{"-fpic"},
	{"-fno-pic"},
	{"-fPIC"},
	{"-fno-PIC"},
	{"-fpie"},
	{"-fno-pie"},
	{"-fPIE"},
	{"-fno-PIE"},
	{"-fsanitize=hands"},
	{"-g"},
	{"-ggdb"},
	{"-march=souza"},
	{"-mcpu=123"},
	{"-mfpu=123"},
	{"-mtune=happybirthday"},
	{"-pic"},
	{"-pthread"},
	{"-Wl,--hash-style=both"},
	{"-Wl,-rpath,foo"},
	{"-Wl,-rpath,$ORIGIN/foo"},
	{"-Wl,-R", "/foo"},
	{"-Wl,-R", "foo"},
	{"-Wl,-R,foo"},
	{"-Wl,--just-symbols=foo"},
	{"-Wl,--just-symbols,foo"},
	{"-Wl,--warn-error"},
	{"-Wl,--no-warn-error"},
	{"foo.so"},
	{"_世界.dll"},
	{"./x.o"},
	{"libcgosotest.dylib"},
	{"-F", "framework"},
	{"-l", "."},
	{"-l", "/etc/passwd"},
	{"-l", "世界"},
	{"-L", "framework"},
	{"-framework", "Chocolate"},
	{"-v"},
	{"-Wl,-sectcreate,__TEXT,__info_plist,${SRCDIR}/Info.plist"},
	{"-Wl,-framework", "-Wl,Chocolate"},
	{"-Wl,-framework,Chocolate"},
	{"-Wl,-unresolved-symbols=ignore-all"},
	{"-Wl,-z,relro"},
	{"-Wl,-z,relro,-z,now"},
	{"-Wl,-z,now"},
	{"-Wl,-z,noexecstack"},
	{"libcgotbdtest.tbd"},
	{"./libcgotbdtest.tbd"},
	{"-Wl,--push-state"},
	{"-Wl,--pop-state"},
	{"-Wl,--push-state,--as-needed"},
	{"-Wl,--push-state,--no-as-needed,-Bstatic"},
	{"-Wl,--just-symbols,."},
	{"-Wl,-framework,."},
	{"-Wl,-rpath,."},
	{"-Wl,-rpath-link,."},
	{"-Wl,-sectcreate,.,.,."},
	{"-Wl,-syslibroot,."},
	{"-Wl,-undefined,."},
}

var badLinkerFlags = [][]string{
	{"-DFOO"},
	{"-Dfoo=bar"},
	{"-W"},
	{"-Wall"},
	{"-fobjc-arc"},
	{"-fno-objc-arc"},
	{"-fomit-frame-pointer"},
	{"-fno-omit-frame-pointer"},
	{"-fsplit-stack"},
	{"-fno-split-stack"},
	{"-fstack-xxx"},
	{"-fno-stack-xxx"},
	{"-mstack-overflow"},
	{"-mno-stack-overflow"},
	{"-mnop-fun-dllimport"},
	{"-std=c99"},
	{"-xc"},
	{"-D", "FOO"},
	{"-D", "foo=bar"},
	{"-I", "FOO"},
	{"-L", "@foo"},
	{"-L", "-foo"},
	{"-x", "c"},
	{"-D@X"},
	{"-D-X"},
	{"-I@dir"},
	{"-I-dir"},
	{"-O@1"},
	{"-Wa,-foo"},
	{"-W@foo"},
	{"-g@gdb"},
	{"-g-gdb"},
	{"-march=@dawn"},
	{"-march=-dawn"},
	{"-std=@c99"},
	{"-std=-c99"},
	{"-x@c"},
	{"-x-c"},
	{"-D", "@foo"},
	{"-D", "-foo"},
	{"-I", "@foo"},
	{"-I", "-foo"},
	{"-l", "@foo"},
	{"-l", "-foo"},
	{"-framework", "-Caffeine"},
	{"-framework", "@Home"},
	{"-Wl,-framework,-Caffeine"},
	{"-Wl,-framework", "-Wl,@Home"},
	{"-Wl,-framework", "@Home"},
	{"-Wl,-framework,Chocolate,@Home"},
	{"-Wl,--hash-style=foo"},
	{"-x", "--c"},
	{"-x", "@obj"},
	{"-Wl,-rpath,@foo"},
	{"-Wl,-R,foo,bar"},
	{"-Wl,-R,@foo"},
	{"-Wl,--just-symbols,@foo"},
	{"../x.o"},
	{"-Wl,-R,"},
	{"-Wl,-O"},
	{"-Wl,-e="},
	{"-Wl,-e,"},
	{"-Wl,-R,-flag"},
	{"-Wl,--push-state,"},
	{"-Wl,--push-state,@foo"},
	{"-fplugin=./-Wl,--push-state,-R.so"},
	{"./-Wl,--push-state,-R.c"},
	{"--sysroot=/tmp"},
	{"--sysroot", "/tmp"},
	{"-target", "x86_64-scei-ps4"},
	{"-target", "x86_64-linux-gnu"},
	{"-fsanitize=@foo"},
	{"-fsanitize=-foo"},
	{"-gnatO"},
	{"-gnatO", "-L", "foo.ld"},
	{"-gen-cdb-fragment-path"},
	{"-gen-cdb-fragment-path", "-Wl,-framework", "-Wl,foo.so"},
}

func TestCheckLinkerFlags(t *testing.T) {
	for _, f := range goodLinkerFlags {
		if err := checkLinkerFlags("test", "test", f); err != nil {
			t.Errorf("unexpected error for %q: %v", f, err)
		}
	}
	for _, f := range badLinkerFlags {
		if err := checkLinkerFlags("test", "test", f); err == nil {
			t.Errorf("missing error for %q", f)
		}
	}
}

func TestCheckLinkerSysrootFlags(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "script.ld")
	if err := os.WriteFile(script, []byte("PLUGIN(\"/tmp/pwn.so\")\n"), 0644); err != nil {
		t.Fatal(err)
	}
	for _, f := range [][]string{
		{"-isysroot", dir},
		{"-Wl,-syslibroot," + dir},
		{"-Wl,-syslibroot=" + dir},
	} {
		if err := checkLinkerFlags("test", "test", f); err != nil {
			t.Errorf("unexpected error for %q: %v", f, err)
		}
	}
	for _, f := range [][]string{
		{"-isysroot", script},
		{"-Wl,-syslibroot," + script},
		{"-Wl,-syslibroot=" + script},
		{"-Wl,--push-state,-syslibroot=" + script},
	} {
		if err := checkLinkerFlags("test", "test", f); err == nil {
			t.Errorf("missing error for %q", f)
		}
	}
}

func TestCheckGCCSpecFiles(t *testing.T) {
	// Compiler invocations always pass -c, which never evaluates GCC's
	// *link_command spec (%:include(libsanitizer.spec) / %:include(libgomp.spec)).
	cwdSpecDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cwdSpecDir, "libsanitizer.spec"), []byte("*link_libubsan:\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(cwdSpecDir)
	for _, f := range [][]string{
		{"-fsanitize=memory"},
		{"-fsanitize=undefined"},
		{"-fopenmp"},
	} {
		if err := checkCompilerFlags("test", "test", f); err != nil {
			t.Errorf("unexpected error for compiler flag %q with libsanitizer.spec in CWD: %v", f, err)
		}
	}

	// Verify -L, -isysroot, and -Wl,-syslibroot reject directories containing
	// GCC spec files (directly, in subdirectories, or in multilib sibling dirs).
	specDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(specDir, "libgomp.spec"), []byte("*link_gomp: %{!static:}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	for _, f := range [][]string{
		{"-L" + specDir, "-fopenmp"},
		{"-L", specDir, "-fopenmp"},
		{"-L=" + specDir, "-fopenmp"},
		{"-L$SYSROOT" + specDir, "-fopenmp"},
		{"-isysroot", specDir, "-fopenmp"},
		{"-Wl,-syslibroot," + specDir, "-fopenmp"},
		{"-Wl,--push-state,-syslibroot=" + specDir, "-fopenmp"},
	} {
		if err := checkLinkerFlags("test", "test", f); err == nil {
			t.Errorf("expected error for GCC spec file in %q, got nil", f)
		}
	}

	// Subdirectory spec file inside sysroot or -L directory.
	nestedRoot := t.TempDir()
	nestedLib := filepath.Join(nestedRoot, "usr", "lib")
	if err := os.MkdirAll(nestedLib, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nestedLib, "libsanitizer.spec"), []byte("*link_libasan: -lasan\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := checkLinkerFlags("test", "test", []string{"-L" + nestedRoot, "-fsanitize=address"}); err == nil {
		t.Errorf("expected error for nested spec file in -L directory, got nil")
	}

	// Multilib sibling directory (../lib64/libgomp.spec relative to -L<dir>/sub,
	// both when <dir>/sub exists and when <dir>/nonexistent does not exist).
	multiRoot := t.TempDir()
	subDir := filepath.Join(multiRoot, "sub")
	lib64Dir := filepath.Join(multiRoot, "lib64")
	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(lib64Dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(lib64Dir, "libgomp.spec"), []byte("*link_gomp:\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := checkLinkerFlags("test", "test", []string{"-L" + subDir, "-fopenmp"}); err == nil {
		t.Errorf("expected error for multilib sibling spec file in ../lib64, got nil")
	}
	if err := checkLinkerFlags("test", "test", []string{"-L" + filepath.Join(multiRoot, "nonexistent"), "-fopenmp"}); err == nil {
		t.Errorf("expected error for multilib sibling spec file via nonexistent child dir, got nil")
	}

	// Symlinked directory pointing to a directory with a spec file, and
	// subdirectory symlink inside a -L directory (N-2).
	linkParent := t.TempDir()
	symDir := filepath.Join(linkParent, "symlink-lib")
	if err := os.Symlink(specDir, symDir); err == nil {
		if err := checkLinkerFlags("test", "test", []string{"-L" + symDir, "-fopenmp"}); err == nil {
			t.Errorf("expected error for symlinked -L directory containing spec file, got nil")
		}
		if err := checkLinkerFlags("test", "test", []string{"-L" + linkParent, "-fopenmp"}); err == nil {
			t.Errorf("expected error for -L directory with subdirectory symlink to spec file, got nil")
		}
	}
	// Symlinked directory pointing to a child directory whose real parent has ../lib64/libgomp.spec.
	symSub := filepath.Join(linkParent, "symlink-sub")
	if err := os.Symlink(subDir, symSub); err == nil {
		if err := checkLinkerFlags("test", "test", []string{"-L" + symSub, "-fopenmp"}); err == nil {
			t.Errorf("expected error for symlinked -L directory with multilib sibling spec file, got nil")
		}
	}

	// CWD fallback when -fsanitize= is combined with -static/-static-lib* or
	// when -fopenmp is used without -L (N-3), while preserving standalone
	// -fsanitize=memory, -fsanitize=address, -fno-openmp, -fopenmp-simd, and -static.
	t.Chdir(specDir)
	for _, f := range [][]string{
		{"-fsanitize=memory"},
		{"-fsanitize=address"},
		{"-fsanitize=undefined"},
		{"-fno-openmp"},
		{"-fopenmp-simd"},
		{"-static"},
	} {
		if err := checkLinkerFlags("test", "test", f); err != nil {
			t.Errorf("unexpected error for safe flag %q with spec file in CWD: %v", f, err)
		}
	}
	for _, f := range [][]string{
		{"-fsanitize=undefined", "-static-libubsan"},
		{"-fsanitize=address", "-static"},
		{"-fopenmp"},
	} {
		if err := checkLinkerFlags("test", "test", f); err == nil {
			t.Errorf("expected error for %q with spec file in CWD and no -L, got nil", f)
		}
	}

	// CWD/package-dir spec checks must not walk ../lib* multilib siblings
	// (e.g. ./../lib -> host /lib when CWD is a depth-1 directory).
	cwdParent := t.TempDir()
	cwdChild := filepath.Join(cwdParent, "app")
	cwdSiblingLib := filepath.Join(cwdParent, "lib")
	if err := os.MkdirAll(cwdChild, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(cwdSiblingLib, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cwdSiblingLib, "libgomp.spec"), []byte("*link_gomp: -lgomp\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(cwdChild)
	if err := checkLinkerFlags("test", "test", []string{"-fopenmp"}); err != nil {
		t.Errorf("unexpected error for CWD check with ../lib/libgomp.spec sibling: %v", err)
	}

	// Host default library trees (/lib*, /usr/lib*) must be skipped when passed
	// without a sysroot prefix (e.g. -L/usr/lib or -L/usr/local/../lib), while
	// sysroot-prefixed -L= paths are still scanned.
	if cfg.Goos != "windows" {
		for _, sysFlag := range [][]string{
			{"-L/usr/lib", "-fopenmp"},
			{"-L/usr/lib64", "-fopenmp"},
		} {
			if err := checkLinkerFlags("test", "test", sysFlag); err != nil {
				t.Errorf("unexpected error for host system %q: %v", sysFlag, err)
			}
		}
	}
}

func TestCheckFlagAllowDisallow(t *testing.T) {
	if err := checkCompilerFlags("TEST", "test", []string{"-disallow"}); err == nil {
		t.Fatalf("missing error for -disallow")
	}
	os.Setenv("CGO_TEST_ALLOW", "-disallo")
	if err := checkCompilerFlags("TEST", "test", []string{"-disallow"}); err == nil {
		t.Fatalf("missing error for -disallow with CGO_TEST_ALLOW=-disallo")
	}
	os.Setenv("CGO_TEST_ALLOW", "-disallow")
	if err := checkCompilerFlags("TEST", "test", []string{"-disallow"}); err != nil {
		t.Fatalf("unexpected error for -disallow with CGO_TEST_ALLOW=-disallow: %v", err)
	}
	os.Unsetenv("CGO_TEST_ALLOW")

	if err := checkCompilerFlags("TEST", "test", []string{"-Wall"}); err != nil {
		t.Fatalf("unexpected error for -Wall: %v", err)
	}
	os.Setenv("CGO_TEST_DISALLOW", "-Wall")
	if err := checkCompilerFlags("TEST", "test", []string{"-Wall"}); err == nil {
		t.Fatalf("missing error for -Wall with CGO_TEST_DISALLOW=-Wall")
	}
	os.Setenv("CGO_TEST_ALLOW", "-Wall") // disallow wins
	if err := checkCompilerFlags("TEST", "test", []string{"-Wall"}); err == nil {
		t.Fatalf("missing error for -Wall with CGO_TEST_DISALLOW=-Wall and CGO_TEST_ALLOW=-Wall")
	}

	os.Setenv("CGO_TEST_ALLOW", "-fplugin.*")
	os.Setenv("CGO_TEST_DISALLOW", "-fplugin=lint.so")
	if err := checkCompilerFlags("TEST", "test", []string{"-fplugin=faster.so"}); err != nil {
		t.Fatalf("unexpected error for -fplugin=faster.so: %v", err)
	}
	if err := checkCompilerFlags("TEST", "test", []string{"-fplugin=lint.so"}); err == nil {
		t.Fatalf("missing error for -fplugin=lint.so: %v", err)
	}
}

func TestCheckCompilerFlagsForInternalLink(t *testing.T) {
	// Any "bad" compiler flag should trigger external linking.
	for _, f := range badCompilerFlags {
		if err := checkCompilerFlagsForInternalLink("test", "test", f); err == nil {
			t.Errorf("missing error for %q", f)
		}
	}

	// All "good" compiler flags should not trigger external linking,
	// except for anything that begins with "-flto".
	for _, f := range goodCompilerFlags {
		foundLTO := false
		for _, s := range f {
			if strings.Contains(s, "-flto") {
				foundLTO = true
			}
		}
		if err := checkCompilerFlagsForInternalLink("test", "test", f); err != nil {
			// expect error for -flto
			if !foundLTO {
				t.Errorf("unexpected error for %q: %v", f, err)
			}
		} else {
			// expect no error for everything else
			if foundLTO {
				t.Errorf("missing error for %q: %v", f, err)
			}
		}
	}
}
