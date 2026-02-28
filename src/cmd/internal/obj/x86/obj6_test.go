// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const testdata = `
MOVQ AX, AX -> MOVQ AX, AX

LEAQ name(SB), AX -> MOVQ name@GOT(SB), AX
LEAQ name+10(SB), AX -> MOVQ name@GOT(SB), AX; LEAQ 10(AX), AX
MOVQ $name(SB), AX -> MOVQ name@GOT(SB), AX
MOVQ $name+10(SB), AX -> MOVQ name@GOT(SB), AX; LEAQ 10(AX), AX

MOVQ name(SB), AX -> NOP; MOVQ name@GOT(SB), R15; MOVQ (R15), AX
MOVQ name+10(SB), AX -> NOP; MOVQ name@GOT(SB), R15; MOVQ 10(R15), AX

CMPQ name(SB), $0 -> NOP; MOVQ name@GOT(SB), R15; CMPQ (R15), $0

MOVQ $1, name(SB) -> NOP; MOVQ name@GOT(SB), R15; MOVQ $1, (R15)
MOVQ $1, name+10(SB) -> NOP; MOVQ name@GOT(SB), R15; MOVQ $1, 10(R15)
`

type ParsedTestData struct {
	input              string
	marks              []int
	marker_to_input    map[int][]string
	marker_to_expected map[int][]string
	marker_to_output   map[int][]string
}

const marker_start = 1234

func parseTestData(t *testing.T) *ParsedTestData {
	r := &ParsedTestData{}
	scanner := bufio.NewScanner(strings.NewReader(testdata))
	r.marker_to_input = make(map[int][]string)
	r.marker_to_expected = make(map[int][]string)
	marker := marker_start
	input_insns := []string{}
	for scanner.Scan() {
		line := scanner.Text()
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}
		parts := strings.Split(line, "->")
		if len(parts) != 2 {
			t.Fatalf("malformed line %v", line)
		}
		r.marks = append(r.marks, marker)
		marker_insn := fmt.Sprintf("MOVQ $%d, AX", marker)
		input_insns = append(input_insns, marker_insn)
		for _, input_insn := range strings.Split(parts[0], ";") {
			input_insns = append(input_insns, input_insn)
			r.marker_to_input[marker] = append(r.marker_to_input[marker], normalize(input_insn))
		}
		for _, expected_insn := range strings.Split(parts[1], ";") {
			r.marker_to_expected[marker] = append(r.marker_to_expected[marker], normalize(expected_insn))
		}
		marker++
	}
	r.input = "TEXT ·foo(SB),$0\n" + strings.Join(input_insns, "\n") + "\n"
	return r
}

var spaces_re *regexp.Regexp = regexp.MustCompile(`\s+`)

func normalize(s string) string {
	return spaces_re.ReplaceAllLiteralString(strings.TrimSpace(s), " ")
}

func asmOutput(t *testing.T, s string) []byte {
	tmpdir := t.TempDir()
	tmpfile, err := os.Create(filepath.Join(tmpdir, "input.s"))
	if err != nil {
		t.Fatal(err)
	}
	defer tmpfile.Close()
	_, err = tmpfile.WriteString(s)
	if err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t,
		testenv.GoToolPath(t), "tool", "asm", "-S", "-dynlink",
		"-o", filepath.Join(tmpdir, "output.6"), tmpfile.Name())

	cmd.Env = append(os.Environ(),
		"GOARCH=amd64", "GOOS=linux", "GOPATH="+filepath.Join(tmpdir, "_gopath"))
	asmout, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("error %s output %s", err, asmout)
	}
	return asmout
}

func parseOutput(t *testing.T, td *ParsedTestData, asmout []byte) {
	scanner := bufio.NewScanner(bytes.NewReader(asmout))
	marker := regexp.MustCompile(`MOVQ \$([0-9]+), AX`)
	mark := -1
	td.marker_to_output = make(map[int][]string)
	for scanner.Scan() {
		line := scanner.Text()
		if line[0] != '\t' {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) != 3 {
			continue
		}
		n := normalize(parts[2])
		mark_matches := marker.FindStringSubmatch(n)
		if mark_matches != nil {
			mark, _ = strconv.Atoi(mark_matches[1])
			if _, ok := td.marker_to_input[mark]; !ok {
				t.Fatalf("unexpected marker %d", mark)
			}
		} else if mark != -1 {
			td.marker_to_output[mark] = append(td.marker_to_output[mark], n)
		}
	}
}

func TestDynlink(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if os.Getenv("GOHOSTARCH") != "" {
		// TODO: make this work? It was failing due to the
		// GOARCH= filtering above and skipping is easiest for
		// now.
		t.Skip("skipping when GOHOSTARCH is set")
	}

	testdata := parseTestData(t)
	asmout := asmOutput(t, testdata.input)
	parseOutput(t, testdata, asmout)
	for _, m := range testdata.marks {
		i := strings.Join(testdata.marker_to_input[m], "; ")
		o := strings.Join(testdata.marker_to_output[m], "; ")
		e := strings.Join(testdata.marker_to_expected[m], "; ")
		if o != e {
			if o == i {
				t.Errorf("%s was unchanged; should have become %s", i, e)
			} else {
				t.Errorf("%s became %s; should have become %s", i, o, e)
			}
		} else if i != e {
			t.Logf("%s correctly became %s", i, o)
		}
	}
}
