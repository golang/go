// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"strings"
	"testing"
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
	{"-g"},
	{"-ggdb"},
	{"-march=souza"},
	{"-mcpu=123"},
	{"-mfpu=123"},
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
