// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var remasterflag = flag.Bool("update-expected", false, "if true, generate updated golden results in testcases for all props tests")

func TestFuncProperties(t *testing.T) {
	td := t.TempDir()
	// td = "/tmp/qqq"
	// os.RemoveAll(td)
	// os.Mkdir(td, 0777)
	testenv.MustHaveGoBuild(t)

	// NOTE: this testpoint has the unfortunate characteristic that it
	// relies on the installed compiler, meaning that if you make
	// changes to the inline heuristics code in your working copy and
	// then run the test, it will test the installed compiler and not
	// your local modifications. TODO: decide whether to convert this
	// to building a fresh compiler on the fly, or using some other
	// scheme.

	testcases := []string{"funcflags", "returns", "params",
		"acrosscall", "calls", "returns2"}
	for _, tc := range testcases {
		dumpfile, err := gatherPropsDumpForFile(t, tc, td)
		if err != nil {
			t.Fatalf("dumping func props for %q: error %v", tc, err)
		}
		// Read in the newly generated dump.
		dentries, dcsites, derr := readDump(t, dumpfile)
		if derr != nil {
			t.Fatalf("reading func prop dump: %v", derr)
		}
		if *remasterflag {
			updateExpected(t, tc, dentries, dcsites)
			continue
		}
		// Generate expected dump.
		epath, egerr := genExpected(td, tc)
		if egerr != nil {
			t.Fatalf("generating expected func prop dump: %v", egerr)
		}
		// Read in the expected result entries.
		eentries, ecsites, eerr := readDump(t, epath)
		if eerr != nil {
			t.Fatalf("reading expected func prop dump: %v", eerr)
		}
		// Compare new vs expected.
		n := len(dentries)
		eidx := 0
		for i := 0; i < n; i++ {
			dentry := dentries[i]
			dcst := dcsites[i]
			if !interestingToCompare(dentry.fname) {
				continue
			}
			if eidx >= len(eentries) {
				t.Errorf("testcase %s missing expected entry for %s, skipping", tc, dentry.fname)
				continue
			}
			eentry := eentries[eidx]
			ecst := ecsites[eidx]
			eidx++
			if dentry.fname != eentry.fname {
				t.Errorf("got fn %q wanted %q, skipping checks",
					dentry.fname, eentry.fname)
				continue
			}
			compareEntries(t, tc, &dentry, dcst, &eentry, ecst)
		}
	}
}

func propBitsToString[T interface{ String() string }](sl []T) string {
	var sb strings.Builder
	for i, f := range sl {
		fmt.Fprintf(&sb, "%d: %s\n", i, f.String())
	}
	return sb.String()
}

func compareEntries(t *testing.T, tc string, dentry *fnInlHeur, dcsites encodedCallSiteTab, eentry *fnInlHeur, ecsites encodedCallSiteTab) {
	dfp := dentry.props
	efp := eentry.props
	dfn := dentry.fname

	// Compare function flags.
	if dfp.Flags != efp.Flags {
		t.Errorf("testcase %q: Flags mismatch for %q: got %s, wanted %s",
			tc, dfn, dfp.Flags.String(), efp.Flags.String())
	}
	// Compare returns
	rgot := propBitsToString[ResultPropBits](dfp.ResultFlags)
	rwant := propBitsToString[ResultPropBits](efp.ResultFlags)
	if rgot != rwant {
		t.Errorf("testcase %q: Results mismatch for %q: got:\n%swant:\n%s",
			tc, dfn, rgot, rwant)
	}
	// Compare receiver + params.
	pgot := propBitsToString[ParamPropBits](dfp.ParamFlags)
	pwant := propBitsToString[ParamPropBits](efp.ParamFlags)
	if pgot != pwant {
		t.Errorf("testcase %q: Params mismatch for %q: got:\n%swant:\n%s",
			tc, dfn, pgot, pwant)
	}
	// Compare call sites.
	for k, ve := range ecsites {
		if vd, ok := dcsites[k]; !ok {
			t.Errorf("testcase %q missing expected callsite %q in func %q", tc, k, dfn)
			continue
		} else {
			if vd != ve {
				t.Errorf("testcase %q callsite %q in func %q: got %+v want %+v",
					tc, k, dfn, vd.String(), ve.String())
			}
		}
	}
	for k := range dcsites {
		if _, ok := ecsites[k]; !ok {
			t.Errorf("testcase %q unexpected extra callsite %q in func %q", tc, k, dfn)
		}
	}
}

type dumpReader struct {
	s  *bufio.Scanner
	t  *testing.T
	p  string
	ln int
}

// readDump reads in the contents of a dump file produced
// by the "-d=dumpinlfuncprops=..." command line flag by the Go
// compiler. It breaks the dump down into separate sections
// by function, then deserializes each func section into a
// fnInlHeur object and returns a slice of those objects.
func readDump(t *testing.T, path string) ([]fnInlHeur, []encodedCallSiteTab, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	dr := &dumpReader{
		s:  bufio.NewScanner(strings.NewReader(string(content))),
		t:  t,
		p:  path,
		ln: 1,
	}
	// consume header comment until preamble delimiter.
	found := false
	for dr.scan() {
		if dr.curLine() == preambleDelimiter {
			found = true
			break
		}
	}
	if !found {
		return nil, nil, fmt.Errorf("malformed testcase file %s, missing preamble delimiter", path)
	}
	res := []fnInlHeur{}
	csres := []encodedCallSiteTab{}
	for {
		dentry, dcst, err := dr.readEntry()
		if err != nil {
			t.Fatalf("reading func prop dump: %v", err)
		}
		if dentry.fname == "" {
			break
		}
		res = append(res, dentry)
		csres = append(csres, dcst)
	}
	return res, csres, nil
}

func (dr *dumpReader) scan() bool {
	v := dr.s.Scan()
	if v {
		dr.ln++
	}
	return v
}

func (dr *dumpReader) curLine() string {
	res := strings.TrimSpace(dr.s.Text())
	if !strings.HasPrefix(res, "// ") {
		dr.t.Fatalf("malformed line %s:%d, no comment: %s", dr.p, dr.ln, res)
	}
	return res[3:]
}

// readObjBlob reads in a series of commented lines until
// it hits a delimiter, then returns the contents of the comments.
func (dr *dumpReader) readObjBlob(delim string) (string, error) {
	var sb strings.Builder
	foundDelim := false
	for dr.scan() {
		line := dr.curLine()
		if delim == line {
			foundDelim = true
			break
		}
		sb.WriteString(line + "\n")
	}
	if err := dr.s.Err(); err != nil {
		return "", err
	}
	if !foundDelim {
		return "", fmt.Errorf("malformed input %s, missing delimiter %q",
			dr.p, delim)
	}
	return sb.String(), nil
}

// readEntry reads a single function's worth of material from
// a file produced by the "-d=dumpinlfuncprops=..." command line
// flag. It deserializes the json for the func properties and
// returns the resulting properties and function name. EOF is
// signaled by a nil FuncProps return (with no error
func (dr *dumpReader) readEntry() (fnInlHeur, encodedCallSiteTab, error) {
	var funcInlHeur fnInlHeur
	var callsites encodedCallSiteTab
	if !dr.scan() {
		return funcInlHeur, callsites, nil
	}
	// first line contains info about function: file/name/line
	info := dr.curLine()
	chunks := strings.Fields(info)
	funcInlHeur.file = chunks[0]
	funcInlHeur.fname = chunks[1]
	if _, err := fmt.Sscanf(chunks[2], "%d", &funcInlHeur.line); err != nil {
		return funcInlHeur, callsites, fmt.Errorf("scanning line %q: %v", info, err)
	}
	// consume comments until and including delimiter
	for {
		if !dr.scan() {
			break
		}
		if dr.curLine() == comDelimiter {
			break
		}
	}

	// Consume JSON for encoded props.
	dr.scan()
	line := dr.curLine()
	fp := &FuncProps{}
	if err := json.Unmarshal([]byte(line), fp); err != nil {
		return funcInlHeur, callsites, err
	}
	funcInlHeur.props = fp

	// Consume callsites.
	callsites = make(encodedCallSiteTab)
	for dr.scan() {
		line := dr.curLine()
		if line == csDelimiter {
			break
		}
		// expected format: "// callsite: <expanded pos> flagstr <desc> flagval <flags> score <score> mask <scoremask> maskstr <scoremaskstring>"
		fields := strings.Fields(line)
		if len(fields) != 12 {
			return funcInlHeur, nil, fmt.Errorf("malformed callsite (nf=%d) %s line %d: %s", len(fields), dr.p, dr.ln, line)
		}
		if fields[2] != "flagstr" || fields[4] != "flagval" || fields[6] != "score" || fields[8] != "mask" || fields[10] != "maskstr" {
			return funcInlHeur, nil, fmt.Errorf("malformed callsite %s line %d: %s",
				dr.p, dr.ln, line)
		}
		tag := fields[1]
		flagstr := fields[5]
		flags, err := strconv.Atoi(flagstr)
		if err != nil {
			return funcInlHeur, nil, fmt.Errorf("bad flags val %s line %d: %q err=%v",
				dr.p, dr.ln, line, err)
		}
		scorestr := fields[7]
		score, err2 := strconv.Atoi(scorestr)
		if err2 != nil {
			return funcInlHeur, nil, fmt.Errorf("bad score val %s line %d: %q err=%v",
				dr.p, dr.ln, line, err2)
		}
		maskstr := fields[9]
		mask, err3 := strconv.Atoi(maskstr)
		if err3 != nil {
			return funcInlHeur, nil, fmt.Errorf("bad mask val %s line %d: %q err=%v",
				dr.p, dr.ln, line, err3)
		}
		callsites[tag] = propsAndScore{
			props: CSPropBits(flags),
			score: score,
			mask:  scoreAdjustTyp(mask),
		}
	}

	// Consume function delimiter.
	dr.scan()
	line = dr.curLine()
	if line != fnDelimiter {
		return funcInlHeur, nil, fmt.Errorf("malformed testcase file %q, missing delimiter %q", dr.p, fnDelimiter)
	}

	return funcInlHeur, callsites, nil
}

// gatherPropsDumpForFile builds the specified testcase 'testcase' from
// testdata/props passing the "-d=dumpinlfuncprops=..." compiler option,
// to produce a properties dump, then returns the path of the newly
// created file. NB: we can't use "go tool compile" here, since
// some of the test cases import stdlib packages (such as "os").
// This means using "go build", which is problematic since the
// Go command can potentially cache the results of the compile step,
// causing the test to fail when being run interactively. E.g.
//
//	$ rm -f dump.txt
//	$ go build -o foo.a -gcflags=-d=dumpinlfuncprops=dump.txt foo.go
//	$ rm -f dump.txt foo.a
//	$ go build -o foo.a -gcflags=-d=dumpinlfuncprops=dump.txt foo.go
//	$ ls foo.a dump.txt > /dev/null
//	ls : cannot access 'dump.txt': No such file or directory
//	$
//
// For this reason, pick a unique filename for the dump, so as to
// defeat the caching.
func gatherPropsDumpForFile(t *testing.T, testcase string, td string) (string, error) {
	t.Helper()
	gopath := "testdata/props/" + testcase + ".go"
	outpath := filepath.Join(td, testcase+".a")
	salt := fmt.Sprintf(".p%dt%d", os.Getpid(), time.Now().UnixNano())
	dumpfile := filepath.Join(td, testcase+salt+".dump.txt")
	run := []string{testenv.GoToolPath(t), "build",
		"-gcflags=-d=dumpinlfuncprops=" + dumpfile, "-o", outpath, gopath}
	out, err := testenv.Command(t, run[0], run[1:]...).CombinedOutput()
	if err != nil {
		t.Logf("compile command: %+v", run)
	}
	if strings.TrimSpace(string(out)) != "" {
		t.Logf("%s", out)
	}
	return dumpfile, err
}

// genExpected reads in a given Go testcase file, strips out all the
// unindented (column 0) commands, writes them out to a new file, and
// returns the path of that new file. By picking out just the comments
// from the Go file we wind up with something that resembles the
// output from a "-d=dumpinlfuncprops=..." compilation.
func genExpected(td string, testcase string) (string, error) {
	epath := filepath.Join(td, testcase+".expected")
	outf, err := os.OpenFile(epath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return "", err
	}
	gopath := "testdata/props/" + testcase + ".go"
	content, err := os.ReadFile(gopath)
	if err != nil {
		return "", err
	}
	lines := strings.Split(string(content), "\n")
	for _, line := range lines[3:] {
		if !strings.HasPrefix(line, "// ") {
			continue
		}
		fmt.Fprintf(outf, "%s\n", line)
	}
	if err := outf.Close(); err != nil {
		return "", err
	}
	return epath, nil
}

type upexState struct {
	dentries   []fnInlHeur
	newgolines []string
	atline     map[uint]uint
}

func mkUpexState(dentries []fnInlHeur) *upexState {
	atline := make(map[uint]uint)
	for _, e := range dentries {
		atline[e.line] = atline[e.line] + 1
	}
	return &upexState{
		dentries: dentries,
		atline:   atline,
	}
}

// updateExpected takes a given Go testcase file X.go and writes out a
// new/updated version of the file to X.go.new, where the column-0
// "expected" comments have been updated using fresh data from
// "dentries".
//
// Writing of expected results is complicated by closures and by
// generics, where you can have multiple functions that all share the
// same starting line. Currently we combine up all the dups and
// closures into the single pre-func comment.
func updateExpected(t *testing.T, testcase string, dentries []fnInlHeur, dcsites []encodedCallSiteTab) {
	nd := len(dentries)

	ues := mkUpexState(dentries)

	gopath := "testdata/props/" + testcase + ".go"
	newgopath := "testdata/props/" + testcase + ".go.new"

	// Read the existing Go file.
	content, err := os.ReadFile(gopath)
	if err != nil {
		t.Fatalf("opening %s: %v", gopath, err)
	}
	golines := strings.Split(string(content), "\n")

	// Preserve copyright.
	ues.newgolines = append(ues.newgolines, golines[:4]...)
	if !strings.HasPrefix(golines[0], "// Copyright") {
		t.Fatalf("missing copyright from existing testcase")
	}
	golines = golines[4:]

	clore := regexp.MustCompile(`.+\.func\d+[\.\d]*$`)

	emitFunc := func(e *fnInlHeur, dcsites encodedCallSiteTab,
		instance, atl uint) {
		var sb strings.Builder
		dumpFnPreamble(&sb, e, dcsites, instance, atl)
		ues.newgolines = append(ues.newgolines,
			strings.Split(strings.TrimSpace(sb.String()), "\n")...)
	}

	// Write file preamble with "DO NOT EDIT" message and such.
	var sb strings.Builder
	dumpFilePreamble(&sb)
	ues.newgolines = append(ues.newgolines,
		strings.Split(strings.TrimSpace(sb.String()), "\n")...)

	// Helper to add a clump of functions to the output file.
	processClump := func(idx int, emit bool) int {
		// Process func itself, plus anything else defined
		// on the same line
		atl := ues.atline[dentries[idx].line]
		for k := uint(0); k < atl; k++ {
			if emit {
				emitFunc(&dentries[idx], dcsites[idx], k, atl)
			}
			idx++
		}
		// now process any closures it contains
		ncl := 0
		for idx < nd {
			nfn := dentries[idx].fname
			if !clore.MatchString(nfn) {
				break
			}
			ncl++
			if emit {
				emitFunc(&dentries[idx], dcsites[idx], 0, 1)
			}
			idx++
		}
		return idx
	}

	didx := 0
	for _, line := range golines {
		if strings.HasPrefix(line, "func ") {

			// We have a function definition.
			// Pick out the corresponding entry or entries in the dump
			// and emit if interesting (or skip if not).
			dentry := dentries[didx]
			emit := interestingToCompare(dentry.fname)
			didx = processClump(didx, emit)
		}

		// Consume all existing comments.
		if strings.HasPrefix(line, "//") {
			continue
		}
		ues.newgolines = append(ues.newgolines, line)
	}

	if didx != nd {
		t.Logf("didx=%d wanted %d", didx, nd)
	}

	// Open new Go file and write contents.
	of, err := os.OpenFile(newgopath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		t.Fatalf("opening %s: %v", newgopath, err)
	}
	fmt.Fprintf(of, "%s", strings.Join(ues.newgolines, "\n"))
	if err := of.Close(); err != nil {
		t.Fatalf("closing %s: %v", newgopath, err)
	}

	t.Logf("update-expected: emitted updated file %s", newgopath)
	t.Logf("please compare the two files, then overwrite %s with %s\n",
		gopath, newgopath)
}

// interestingToCompare returns TRUE if we want to compare results
// for function 'fname'.
func interestingToCompare(fname string) bool {
	if strings.HasPrefix(fname, "init.") {
		return true
	}
	if strings.HasPrefix(fname, "T_") {
		return true
	}
	f := strings.Split(fname, ".")
	if len(f) == 2 && strings.HasPrefix(f[1], "T_") {
		return true
	}
	return false
}
