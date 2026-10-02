// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package ld

import (
	"bytes"
	imacho "cmd/internal/macho"
	"debug/macho"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

func TestMachoSectionsReadOnly(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	const (
		prog  = `package main; func main() {}`
		progC = `package main; import "C"; func main() {}`
	)

	tests := []struct {
		name             string
		args             []string
		prog             string
		wantSecsRO       []string
		mustHaveCGO      bool
		mustInternalLink bool
	}{
		{
			name:             "linkmode-internal",
			args:             []string{"-ldflags", "-linkmode=internal"},
			prog:             prog,
			mustInternalLink: true,
			wantSecsRO:       []string{"__got", "__rodata"},
		},
		{
			name:        "linkmode-external",
			args:        []string{"-ldflags", "-linkmode=external"},
			prog:        prog,
			mustHaveCGO: true,
			wantSecsRO:  []string{"__got", "__rodata"},
		},
		{
			name:             "cgo-linkmode-internal",
			args:             []string{"-ldflags", "-linkmode=external"},
			prog:             progC,
			mustHaveCGO:      true,
			mustInternalLink: true,
			wantSecsRO:       []string{"__got", "__rodata"},
		},
		{
			name:        "cgo-linkmode-external",
			args:        []string{"-ldflags", "-linkmode=external"},
			prog:        progC,
			mustHaveCGO: true,
			wantSecsRO:  []string{"__got", "__rodata"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.mustInternalLink {
				testenv.MustInternalLink(t, testenv.SpecialBuildTypes{Cgo: test.mustHaveCGO})
			}
			if test.mustHaveCGO {
				testenv.MustHaveCGO(t)
			}

			var (
				dir     = t.TempDir()
				src     = filepath.Join(dir, fmt.Sprintf("macho_%s.go", test.name))
				binFile = filepath.Join(dir, test.name)
			)

			if err := os.WriteFile(src, []byte(test.prog), 0666); err != nil {
				t.Fatal(err)
			}

			cmdArgs := append([]string{"build", "-o", binFile}, append(test.args, src)...)
			cmd := testenv.Command(t, testenv.GoToolPath(t), cmdArgs...)

			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("failed to build %v: %v:\n%s", cmd.Args, err, out)
			}

			fi, err := os.Open(binFile)
			if err != nil {
				t.Fatalf("failed to open built file: %v", err)
			}
			defer fi.Close()

			machoFile, err := macho.NewFile(fi)
			if err != nil {
				t.Fatalf("failed to parse macho file: %v", err)
			}
			defer machoFile.Close()

			// Load segments
			segs := make(map[string]*macho.Segment)
			for _, l := range machoFile.Loads {
				if s, ok := l.(*macho.Segment); ok {
					segs[s.Name] = s
				}
			}

			for _, wsroname := range test.wantSecsRO {
				// Now walk the sections. Section should be part of
				// some segment that is made readonly after
				// relocations are appied.
				var wsro *macho.Section
				foundRO := false
				for _, s := range machoFile.Sections {
					if s.Name == wsroname {
						seg := segs[s.Seg]
						if seg == nil {
							t.Fatalf("test %s: can't locate segment for %q section",
								test.name, wsroname)
						}
						if seg.Flag == 0x10 { // SG_READ_ONLY
							foundRO = true
							wsro = s
							break
						}
					}
				}
				if wsro == nil {
					t.Fatalf("test %s: can't locate %q section",
						test.name, wsroname)
					continue
				}
				if !foundRO {
					// Things went off the rails. Write out some
					// useful information for a human looking at the
					// test failure.
					t.Logf("test %s: %q section not in readonly segment",
						wsro.Name, test.name)
					t.Logf("section %s location: st=0x%x en=0x%x\n",
						wsro.Name, wsro.Addr, wsro.Addr+wsro.Size)
					t.Logf("sec %s found in this segment: ", wsro.Seg)
					t.Logf("\nall segments: \n")
					for _, l := range machoFile.Loads {
						if s, ok := l.(*macho.Segment); ok {
							t.Logf("cmd=%s fl=%d st=0x%x en=0x%x\n",
								s.Cmd, s.Flag, s.Addr, s.Addr+s.Filesz)
						}
					}
					t.Fatalf("test %s failed", test.name)
				}
			}
		})
	}
}

func TestMachoSplitDWARF(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	const prog = `package main; func main() {}`

	uuid := func(f *macho.File) []byte {
		for _, l := range f.Loads {
			raw := l.Raw()
			if len(raw) >= 24 && f.ByteOrder.Uint32(raw[:4]) == imacho.LC_UUID {
				return raw[8:24]
			}
		}
		return nil
	}

	for _, linkmode := range []string{"internal", "external"} {
		for _, split := range []bool{true, false} {
			name := fmt.Sprintf("%s_split_%v", linkmode, split)
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				if linkmode == "internal" {
					testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})
				} else {
					testenv.MustHaveCGO(t)
				}

				dir := t.TempDir()
				src := filepath.Join(dir, "main.go")
				if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
					t.Fatal(err)
				}
				binFile := filepath.Join(dir, "main")
				ldflags := fmt.Sprintf("-linkmode=%s -splitdwarf=%v", linkmode, split)
				cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", binFile, "-ldflags", ldflags, src)
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("build failed: %v\n%s", err, out)
				}

				mf, err := macho.Open(binFile)
				if err != nil {
					t.Fatal(err)
				}
				defer mf.Close()

				if _, err := mf.DWARF(); err != nil {
					t.Errorf("expected DWARF from executable (or its dSYM), got error: %v", err)
				}

				hasDWARFSeg := mf.Segment("__DWARF") != nil
				dsymPath := filepath.Join(binFile+".dSYM", "Contents", "Resources", "DWARF", filepath.Base(binFile))
				_, dsymStatErr := os.Stat(dsymPath)

				if split {
					if hasDWARFSeg {
						t.Errorf("expected no __DWARF segment in executable, but found one")
					}
					if dsymStatErr != nil {
						t.Fatalf("expected dSYM file at %s, got error: %v", dsymPath, dsymStatErr)
					}
					df, err := macho.Open(dsymPath)
					if err != nil {
						t.Fatalf("failed to open dSYM file: %v", err)
					}
					defer df.Close()
					u1, u2 := uuid(mf), uuid(df)
					if len(u1) == 0 || !bytes.Equal(u1, u2) {
						t.Errorf("UUID mismatch: executable=%x, dSYM=%x", u1, u2)
					}
					if _, err := df.DWARF(); err != nil {
						t.Errorf("expected DWARF in dSYM file, got error: %v", err)
					}
					if df.Symtab == nil || len(df.Symtab.Syms) == 0 {
						t.Errorf("expected symbol table in dSYM file")
					}
					if df.Section("__zdebug_aranges") == nil && df.Section("__debug_aranges") == nil {
						t.Errorf("expected debug_aranges section in dSYM file")
					}
				} else {
					if !hasDWARFSeg {
						t.Errorf("expected __DWARF segment in executable with -splitdwarf=false, but found none")
					}
					if !os.IsNotExist(dsymStatErr) {
						t.Errorf("expected no dSYM file with -splitdwarf=false, but stat returned: %v", dsymStatErr)
					}
				}
			})
		}
	}
}
