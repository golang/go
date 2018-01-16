// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for testing against external disassembler program.
// Copied and simplified from ../../arm/armasm/ext_test.go.

package arm64asm

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	dumpTest = flag.Bool("dump", false, "dump all encodings")
	mismatch = flag.Bool("mismatch", false, "log allowed mismatches")
	longTest = flag.Bool("long", false, "long test")
	keep     = flag.Bool("keep", false, "keep object files around")
	debug    = false
)

// An ExtInst represents a single decoded instruction parsed
// from an external disassembler's output.
type ExtInst struct {
	addr uint64
	enc  [4]byte
	nenc int
	text string
}

func (r ExtInst) String() string {
	return fmt.Sprintf("%#x: % x: %s", r.addr, r.enc, r.text)
}

// An ExtDis is a connection between an external disassembler and a test.
type ExtDis struct {
	Arch     Mode
	Dec      chan ExtInst
	File     *os.File
	Size     int
	KeepFile bool
	Cmd      *exec.Cmd
}

// InstJson describes instruction fields value got from ARMv8-A Reference Manual
type InstJson struct {
	Name   string
	Bits   string
	Arch   string
	Syntax string
	Code   string
	Alias  string
	Enc    uint32
}

// A Mode is an instruction execution mode.
type Mode int

const (
	_ Mode = iota
	ModeARM64
)

// Run runs the given command - the external disassembler - and returns
// a buffered reader of its standard output.
func (ext *ExtDis) Run(cmd ...string) (*bufio.Reader, error) {
	if *keep {
		log.Printf("%s\n", strings.Join(cmd, " "))
	}
	ext.Cmd = exec.Command(cmd[0], cmd[1:]...)
	out, err := ext.Cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdoutpipe: %v", err)
	}
	if err := ext.Cmd.Start(); err != nil {
		return nil, fmt.Errorf("exec: %v", err)
	}

	b := bufio.NewReaderSize(out, 1<<20)
	return b, nil
}

// Wait waits for the command started with Run to exit.
func (ext *ExtDis) Wait() error {
	return ext.Cmd.Wait()
}

// testExtDis tests a set of byte sequences against an external disassembler.
// The disassembler is expected to produce the given syntax and run
// in the given architecture mode (16, 32, or 64-bit).
// The extdis function must start the external disassembler
// and then parse its output, sending the parsed instructions on ext.Dec.
// The generate function calls its argument f once for each byte sequence
// to be tested. The generate function itself will be called twice, and it must
// make the same sequence of calls to f each time.
// When a disassembly does not match the internal decoding,
// allowedMismatch determines whether this mismatch should be
// allowed, or else considered an error.
func testExtDis(
	t *testing.T,
	syntax string,
	arch Mode,
	extdis func(ext *ExtDis) error,
	generate func(f func([]byte)),
	allowedMismatch func(text string, inst *Inst, dec ExtInst) bool,
) {
	start := time.Now()
	ext := &ExtDis{
		Dec:  make(chan ExtInst),
		Arch: arch,
	}
	errc := make(chan error)

	// First pass: write instructions to input file for external disassembler.
	file, f, size, err := writeInst(generate)
	if err != nil {
		t.Fatal(err)
	}
	ext.Size = size
	ext.File = f
	defer func() {
		f.Close()
		if !*keep {
			os.Remove(file)
		}
	}()

	// Second pass: compare disassembly against our decodings.
	var (
		totalTests  = 0
		totalSkips  = 0
		totalErrors = 0

		errors = make([]string, 0, 100) // Sampled errors, at most cap
	)
	go func() {
		errc <- extdis(ext)
	}()

	generate(func(enc []byte) {
		dec, ok := <-ext.Dec
		if !ok {
			t.Errorf("decoding stream ended early")
			return
		}
		inst, text := disasm(syntax, pad(enc))

		totalTests++
		if *dumpTest {
			fmt.Printf("%x -> %s [%d]\n", enc[:len(enc)], dec.text, dec.nenc)
		}
		if text != dec.text && !strings.Contains(dec.text, "unknown") && syntax == "gnu" {
			suffix := ""
			if allowedMismatch(text, &inst, dec) {
				totalSkips++
				if !*mismatch {
					return
				}
				suffix += " (allowed mismatch)"
			}
			totalErrors++
			cmp := fmt.Sprintf("decode(%x) = %q, %d, want %q, %d%s\n", enc, text, len(enc), dec.text, dec.nenc, suffix)

			if len(errors) >= cap(errors) {
				j := rand.Intn(totalErrors)
				if j >= cap(errors) {
					return
				}
				errors = append(errors[:j], errors[j+1:]...)
			}
			errors = append(errors, cmp)
		}
	})

	if *mismatch {
		totalErrors -= totalSkips
	}

	for _, b := range errors {
		t.Log(b)
	}

	if totalErrors > 0 {
		t.Fail()
	}
	t.Logf("%d test cases, %d expected mismatches, %d failures; %.0f cases/second", totalTests, totalSkips, totalErrors, float64(totalTests)/time.Since(start).Seconds())
	t.Logf("decoder coverage: %.1f%%;\n", decodeCoverage())
	if err := <-errc; err != nil {
		t.Fatalf("external disassembler: %v", err)
	}

}

// Start address of text.
const start = 0x8000

// writeInst writes the generated byte sequences to a new file
// starting at offset start. That file is intended to be the input to
// the external disassembler.
func writeInst(generate func(func([]byte))) (file string, f *os.File, size int, err error) {
	f, err = ioutil.TempFile("", "arm64asm")
	if err != nil {
		return
	}

	file = f.Name()

	f.Seek(start, io.SeekStart)
	w := bufio.NewWriter(f)
	defer w.Flush()
	size = 0
	generate(func(x []byte) {
		if debug {
			fmt.Printf("%#x: %x%x\n", start+size, x, zeros[len(x):])
		}
		w.Write(x)
		w.Write(zeros[len(x):])
		size += len(zeros)
	})
	return file, f, size, nil
}

var zeros = []byte{0, 0, 0, 0}

// pad pads the code sequence with pops.
func pad(enc []byte) []byte {
	if len(enc) < 4 {
		enc = append(enc[:len(enc):len(enc)], zeros[:4-len(enc)]...)
	}
	return enc
}

// disasm returns the decoded instruction and text
// for the given source bytes, using the given syntax and mode.
func disasm(syntax string, src []byte) (inst Inst, text string) {
	var err error
	inst, err = Decode(src)
	if err != nil {
		text = "error: " + err.Error()
		return
	}
	text = inst.String()
	switch syntax {
	case "gnu":
		text = GNUSyntax(inst)
	case "plan9": // [sic]
		text = GoSyntax(inst, 0, nil, nil)
	default:
		text = "error: unknown syntax " + syntax
	}
	return
}

// decodecoverage returns a floating point number denoting the
// decoder coverage.
func decodeCoverage() float64 {
	n := 0
	for _, t := range decoderCover {
		if t {
			n++
		}
	}
	return 100 * float64(1+n) / float64(1+len(decoderCover))
}

// Helpers for writing disassembler output parsers.

// hasPrefix reports whether any of the space-separated words in the text s
// begins with any of the given prefixes.
func hasPrefix(s string, prefixes ...string) bool {
	for _, prefix := range prefixes {
		for cur_s := s; cur_s != ""; {
			if strings.HasPrefix(cur_s, prefix) {
				return true
			}
			i := strings.Index(cur_s, " ")
			if i < 0 {
				break
			}
			cur_s = cur_s[i+1:]
		}
	}
	return false
}

// isHex reports whether b is a hexadecimal character (0-9a-fA-F).
func isHex(b byte) bool {
	return ('0' <= b && b <= '9') || ('a' <= b && b <= 'f') || ('A' <= b && b <= 'F')
}

// parseHex parses the hexadecimal byte dump in hex,
// appending the parsed bytes to raw and returning the updated slice.
// The returned bool reports whether any invalid hex was found.
// Spaces and tabs between bytes are okay but any other non-hex is not.
func parseHex(hex []byte, raw []byte) ([]byte, bool) {
	hex = bytes.TrimSpace(hex)
	for j := 0; j < len(hex); {
		for hex[j] == ' ' || hex[j] == '\t' {
			j++
		}
		if j >= len(hex) {
			break
		}
		if j+2 > len(hex) || !isHex(hex[j]) || !isHex(hex[j+1]) {
			return nil, false
		}
		raw = append(raw, unhex(hex[j])<<4|unhex(hex[j+1]))
		j += 2
	}
	return raw, true
}

func unhex(b byte) byte {
	if '0' <= b && b <= '9' {
		return b - '0'
	} else if 'A' <= b && b <= 'F' {
		return b - 'A' + 10
	} else if 'a' <= b && b <= 'f' {
		return b - 'a' + 10
	}
	return 0
}

// index is like bytes.Index(s, []byte(t)) but avoids the allocation.
func index(s []byte, t string) int {
	i := 0
	for {
		j := bytes.IndexByte(s[i:], t[0])
		if j < 0 {
			return -1
		}
		i = i + j
		if i+len(t) > len(s) {
			return -1
		}
		for k := 1; k < len(t); k++ {
			if s[i+k] != t[k] {
				goto nomatch
			}
		}
		return i
	nomatch:
		i++
	}
}

// fixSpace rewrites runs of spaces, tabs, and newline characters into single spaces in s.
// If s must be rewritten, it is rewritten in place.
func fixSpace(s []byte) []byte {
	s = bytes.TrimSpace(s)
	for i := 0; i < len(s); i++ {
		if s[i] == '\t' || s[i] == '\n' || i > 0 && s[i] == ' ' && s[i-1] == ' ' {
			goto Fix
		}
	}
	return s

Fix:
	b := s
	w := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '\t' || c == '\n' {
			c = ' '
		}
		if c == ' ' && w > 0 && b[w-1] == ' ' {
			continue
		}
		b[w] = c
		w++
	}
	if w > 0 && b[w-1] == ' ' {
		w--
	}
	return b[:w]
}

// Fllowing regular expressions matches instructions using relative addressing mode.
// pcrel matches B instructions and BL instructions.
// pcrelr matches instrucions which consisted of register arguments and label arguments.
// pcrelim matches instructions which consisted of register arguments, immediate
// arguments and lable arguments.
// pcrelrzr and prcelimzr matches instructions when register arguments is zero register.
// pcrelprfm matches PRFM instructions when arguments consisted of register and lable.
// pcrelprfmim matches PRFM instructions when arguments consisted of immediate and lable.
var (
	pcrel       = regexp.MustCompile(`^((?:.* )?(?:b|bl)x?(?:\.)?(?:eq|ne|cs|cc|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv)?) 0x([0-9a-f]+)$`)
	pcrelr      = regexp.MustCompile(`^((?:.*)?(?:ldr|adrp|adr|cbnz|cbz|ldrsw) (?:x|w|s|d|q)(?:[0-9]+,)) 0x([0-9a-f]+)$`)
	pcrelrzr    = regexp.MustCompile(`^((?:.*)?(?:ldr|adrp|adr|cbnz|cbz|ldrsw) (?:x|w)zr,) 0x([0-9a-f]+)$`)
	pcrelim     = regexp.MustCompile(`^((?:.*)?(?:tbnz|tbz) (?:x|w)(?:[0-9]+,) (?:#[0-9a-f]+,)) 0x([0-9a-f]+)$`)
	pcrelimzr   = regexp.MustCompile(`^((?:.*)?(?:tbnz|tbz) (?:x|w)zr, (?:#[0-9a-f]+,)) 0x([0-9a-f]+)$`)
	pcrelprfm   = regexp.MustCompile(`^((?:.*)?(?:prfm) (?:[0-9a-z]+,)) 0x([0-9a-f]+)$`)
	pcrelprfmim = regexp.MustCompile(`^((?:.*)?(?:prfm) (?:#0x[0-9a-f]+,)) 0x([0-9a-f]+)$`)
)

// Round is the multiple of the number of instructions that read from Json file.
// Round used as seed value for pseudo-random number generator provides the same sequence
// in the same round run for the external disassembler and decoder.
var Round int

// condmark is used to mark conditional instructions when need to generate and test
// conditional instructions.
var condmark bool = false

// Generate instruction binary according to Json file
// Encode variable field of instruction with random value
func doFuzzy(inst *InstJson, Ninst int) {
	var testdata uint32
	var NonDigRE = regexp.MustCompile(`[\D]`)
	rand.Seed(int64(Round + Ninst))
	off := 0
	DigBit := ""
	if condmark == true && !strings.Contains(inst.Bits, "cond") {
		inst.Enc = 0xffffffff
	} else {
		for _, f := range strings.Split(inst.Bits, "|") {
			if i := strings.Index(f, ":"); i >= 0 {
				// consider f contains "01:2" and "Rm:5"
				DigBit = f[:i]
				m := NonDigRE.FindStringSubmatch(DigBit)
				if m == nil {
					DigBit = strings.TrimSpace(DigBit)
					s := strings.Split(DigBit, "")
					for i := 0; i < len(s); i++ {
						switch s[i] {
						case "1", "(1)":
							testdata |= 1 << uint(31-off)
						}
						off++
					}
				} else {
					// DigBit is "Rn" or "imm3"
					n, _ := strconv.Atoi(f[i+1:])
					if DigBit == "cond" && condmark == true {
						r := uint8(Round)
						for i := n - 1; i >= 0; i-- {
							switch (r >> uint(i)) & 1 {
							case 1:
								testdata |= 1 << uint(31-off)
							}
							off++
						}
					} else {
						for i := 0; i < n; i++ {
							r := rand.Intn(2)
							switch r {
							case 1:
								testdata |= 1 << uint(31-off)
							}
							off++
						}
					}
				}
				continue
			}
			for _, bit := range strings.Fields(f) {
				switch bit {
				case "0", "(0)":
					off++
					continue
				case "1", "(1)":
					testdata |= 1 << uint(31-off)
				default:
					r := rand.Intn(2)
					switch r {
					case 1:
						testdata |= 1 << uint(31-off)
					}
				}
				off++
			}
		}
		if off != 32 {
			log.Printf("incorrect bit count for %s %s: have %d", inst.Name, inst.Bits, off)
		}
		inst.Enc = testdata
	}
}

// Generators.
//
// The test cases are described as functions that invoke a callback repeatedly,
// with a new input sequence each time. These helpers make writing those
// a little easier.

// JSONCases generates ARM64 instructions according to inst.json.
func JSONCases(t *testing.T) func(func([]byte)) {
	return func(try func([]byte)) {
		data, err := ioutil.ReadFile("inst.json")
		if err != nil {
			t.Fatal(err)
		}
		var insts []InstJson
		var instsN []InstJson
		// Change N value to get more cases only when condmark=false.
		N := 100
		if condmark == true {
			N = 16
		}
		if err := json.Unmarshal(data, &insts); err != nil {
			t.Fatal(err)
		}
		// Append instructions to get more test cases.
		for i := 0; i < N; {
			for _, inst := range insts {
				instsN = append(instsN, inst)
			}
			i++
		}
		Round = 0
		for i := range instsN {
			if i%len(insts) == 0 {
				Round++
			}
			doFuzzy(&instsN[i], i)
		}
		for _, inst := range instsN {
			if condmark == true && inst.Enc == 0xffffffff {
				continue
			}
			enc := inst.Enc
			try([]byte{byte(enc), byte(enc >> 8), byte(enc >> 16), byte(enc >> 24)})
		}
	}
}

// condCases generates conditional instructions.
func condCases(t *testing.T) func(func([]byte)) {
	return func(try func([]byte)) {
		condmark = true
		JSONCases(t)(func(enc []byte) {
			try(enc)
		})
	}
}

// hexCases generates the cases written in hexadecimal in the encoded string.
// Spaces in 'encoded' separate entire test cases, not individual bytes.
func hexCases(t *testing.T, encoded string) func(func([]byte)) {
	return func(try func([]byte)) {
		for _, x := range strings.Fields(encoded) {
			src, err := hex.DecodeString(x)
			if err != nil {
				t.Errorf("parsing %q: %v", x, err)
			}
			try(src)
		}
	}
}

// testdataCases generates the test cases recorded in testdata/cases.txt.
// It only uses the inputs; it ignores the answers recorded in that file.
func testdataCases(t *testing.T, syntax string) func(func([]byte)) {
	var codes [][]byte
	input := filepath.Join("testdata", syntax+"cases.txt")
	data, err := ioutil.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)[0]
		i := strings.Index(f, "|")
		if i < 0 {
			t.Errorf("parsing %q: missing | separator", f)
			continue
		}
		if i%2 != 0 {
			t.Errorf("parsing %q: misaligned | separator", f)
		}
		code, err := hex.DecodeString(f[:i] + f[i+1:])
		if err != nil {
			t.Errorf("parsing %q: %v", f, err)
			continue
		}
		codes = append(codes, code)
	}

	return func(try func([]byte)) {
		for _, code := range codes {
			try(code)
		}
	}
}
