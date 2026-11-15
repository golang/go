// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
)

func forgedArchiveMember(name string, data []byte) []byte {
	hdr := fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", name, 0, 0, 0, 0o644, len(data))
	b := append([]byte(hdr), data...)
	if len(data)&1 != 0 {
		b = append(b, 0)
	}
	return b
}

func TestReadLogosFactsRejectsForgedDuplicate(t *testing.T) {
	fp := [8]byte{0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11}
	forged := fmt.Sprintf("logos-facts 3\nfingerprint %x\n", fp) +
		"fn Bump ParamMutated=0\n" +
		"fn Bump ParamMutated=1,MayYield=0\n"
	var ar []byte
	ar = append(ar, []byte("!<arch>\n")...)
	ar = append(ar, forgedArchiveMember("__.PKGDEF", []byte("x"))...)
	ar = append(ar, forgedArchiveMember(facts.MemberName, []byte(forged))...)
	path := filepath.Join(t.TempDir(), "dep.a")
	if err := os.WriteFile(path, ar, 0o666); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := readLogosFacts(f, "example.com/fgd", "exportdata"+string(fp[:]))
	if err == nil {
		t.Fatalf("readLogosFacts(forged duplicate) = %+v, want build failure", got)
	}
	if !strings.Contains(err.Error(), "uplicate") {
		t.Fatalf("readLogosFacts(forged duplicate) error %q does not name duplication", err)
	}
}
