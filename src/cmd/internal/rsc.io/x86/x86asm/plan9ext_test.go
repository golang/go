// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"testing"
)

const plan9Path = "testdata/libmach8db"

func testPlan9Arch(t *testing.T, arch int, generate func(func([]byte))) {
	if testing.Short() {
		t.Skip("skipping libmach test in short mode")
	}

	if _, err := os.Stat(plan9Path); err != nil {
		t.Fatal(err)
	}

	testExtDis(t, "plan9", arch, plan9, generate, allowedMismatchPlan9)
}

func testPlan932(t *testing.T, generate func(func([]byte))) {
	testPlan9Arch(t, 32, generate)
}

func testPlan964(t *testing.T, generate func(func([]byte))) {
	testPlan9Arch(t, 64, generate)
}

func plan9(ext *ExtDis) error {
	flag := "-8"
	if ext.Arch == 64 {
		flag = "-6"
	}
	b, err := ext.Run(plan9Path, flag, ext.File.Name())
	if err != nil {
		return err
	}

	nmatch := 0
	next := uint32(start)
	var (
		addr   uint32
		encbuf [32]byte
		enc    []byte
		text   string
	)

	for {
		line, err := b.ReadSlice('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("reading libmach8db output: %v", err)
		}
		if debug {
			os.Stdout.Write(line)
		}
		nmatch++
		addr, enc, text = parseLinePlan9(line, encbuf[:0])
		if addr > next {
			return fmt.Errorf("address out of sync expected <= %#x at %q in:\n%s", next, line, line)
		}
		if addr < next {
			continue
		}
		if m := pcrelw.FindStringSubmatch(text); m != nil {
			targ, _ := strconv.ParseUint(m[2], 16, 64)
			text = fmt.Sprintf("%s .%+#x", m[1], int16(uint32(targ)-uint32(uint16(addr))-uint32(len(enc))))
		}
		if m := pcrel.FindStringSubmatch(text); m != nil {
			targ, _ := strconv.ParseUint(m[2], 16, 64)
			text = fmt.Sprintf("%s .%+#x", m[1], int32(uint32(targ)-addr-uint32(len(enc))))
		}
		ext.Dec <- ExtInst{addr, encbuf, len(enc), text}
		encbuf = [32]byte{}
		enc = nil
		next += 32
	}
	if next != start+uint32(ext.Size) {
		return fmt.Errorf("not enough results found [%d %d]", next, start+ext.Size)
	}
	if err := ext.Wait(); err != nil {
		return fmt.Errorf("exec: %v", err)
	}

	return nil
}

func parseLinePlan9(line []byte, encstart []byte) (addr uint32, enc []byte, text string) {
	i := bytes.IndexByte(line, ' ')
	if i < 0 || line[0] != '0' || line[1] != 'x' {
		log.Fatalf("cannot parse disassembly: %q", line)
	}
	j := bytes.IndexByte(line[i+1:], ' ')
	if j < 0 {
		log.Fatalf("cannot parse disassembly: %q", line)
	}
	j += i + 1
	x, err := strconv.ParseUint(string(trimSpace(line[2:i])), 16, 32)
	if err != nil {
		log.Fatalf("cannot parse disassembly: %q", line)
	}
	addr = uint32(x)
	enc, ok := parseHex(line[i+1:j], encstart)
	if !ok {
		log.Fatalf("cannot parse disassembly: %q", line)
	}
	return addr, enc, string(fixSpace(line[j+1:]))
}
