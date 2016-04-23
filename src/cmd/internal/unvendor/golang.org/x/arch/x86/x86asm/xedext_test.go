package x86asm

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"
)

// xed binary from Intel sde-external-6.22.0-2014-03-06.
const xedPath = "/Users/rsc/bin/xed"

func testXedArch(t *testing.T, arch int, generate func(func([]byte))) {
	if testing.Short() {
		t.Skip("skipping xed test in short mode")
	}
	if _, err := os.Stat(xedPath); err != nil {
		t.Skip(err)
	}

	testExtDis(t, "intel", arch, xed, generate, allowedMismatchXed)
}

func testXed32(t *testing.T, generate func(func([]byte))) {
	testXedArch(t, 32, generate)
}

func testXed64(t *testing.T, generate func(func([]byte))) {
	testXedArch(t, 64, generate)
}

func xed(ext *ExtDis) error {
	b, err := ext.Run(xedPath, fmt.Sprintf("-%d", ext.Arch), "-n", "1G", "-ir", ext.File.Name())
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

	var xedEnd = []byte("# end of text section")
	var xedEnd1 = []byte("# Errors")

	eof := false
	for {
		line, err := b.ReadSlice('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("reading objdump output: %v", err)
		}
		if debug {
			os.Stdout.Write(line)
		}
		if bytes.HasPrefix(line, xedEnd) || bytes.HasPrefix(line, xedEnd1) {
			eof = true
		}
		if eof {
			continue
		}
		nmatch++
		addr, enc, text = parseLineXed(line, encbuf[:0])
		if addr > next {
			return fmt.Errorf("address out of sync expected <= %#x at %q in:\n%s", next, line, line)
		}
		if addr < next {
			continue
		}
		switch text {
		case "repz":
			text = "rep"
		case "repnz":
			text = "repn"
		default:
			text = strings.Replace(text, "repz ", "rep ", -1)
			text = strings.Replace(text, "repnz ", "repn ", -1)
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

var (
	xedInRaw    = []byte("In raw...")
	xedDots     = []byte("...")
	xdis        = []byte("XDIS ")
	xedError    = []byte("ERROR: ")
	xedNoDecode = []byte("Could not decode at offset: 0x")
)

func parseLineXed(line []byte, encstart []byte) (addr uint32, enc []byte, text string) {
	oline := line
	if bytes.HasPrefix(line, xedInRaw) || bytes.HasPrefix(line, xedDots) {
		return 0, nil, ""
	}
	if bytes.HasPrefix(line, xedError) {
		i := bytes.IndexByte(line[len(xedError):], ' ')
		if i < 0 {
			log.Fatalf("cannot parse error: %q", oline)
		}
		errstr := string(line[len(xedError):])
		i = bytes.Index(line, xedNoDecode)
		if i < 0 {
			log.Fatalf("cannot parse error: %q", oline)
		}
		i += len(xedNoDecode)
		j := bytes.IndexByte(line[i:], ' ')
		if j < 0 {
			log.Fatalf("cannot parse error: %q", oline)
		}
		x, err := strconv.ParseUint(string(trimSpace(line[i:i+j])), 16, 32)
		if err != nil {
			log.Fatalf("cannot parse disassembly: %q", oline)
		}
		addr = uint32(x)
		return addr, nil, errstr
	}

	if !bytes.HasPrefix(line, xdis) {
		log.Fatalf("cannot parse disassembly: %q", oline)
	}

	i := bytes.IndexByte(line, ':')
	if i < 0 {
		log.Fatalf("cannot parse disassembly: %q", oline)
	}
	x, err := strconv.ParseUint(string(trimSpace(line[len(xdis):i])), 16, 32)
	if err != nil {
		log.Fatalf("cannot parse disassembly: %q", oline)
	}
	addr = uint32(x)

	// spaces
	i++
	for i < len(line) && line[i] == ' ' {
		i++
	}
	// instruction class, spaces
	for i < len(line) && line[i] != ' ' {
		i++
	}
	for i < len(line) && line[i] == ' ' {
		i++
	}
	// instruction set, spaces
	for i < len(line) && line[i] != ' ' {
		i++
	}
	for i < len(line) && line[i] == ' ' {
		i++
	}

	// hex
	hexStart := i
	for i < len(line) && line[i] != ' ' {
		i++
	}
	hexEnd := i
	for i < len(line) && line[i] == ' ' {
		i++
	}

	// text
	textStart := i
	for i < len(line) && line[i] != '\n' {
		i++
	}
	textEnd := i

	enc, ok := parseHex(line[hexStart:hexEnd], encstart)
	if !ok {
		log.Fatalf("cannot parse disassembly: %q", oline)
	}

	return addr, enc, string(fixSpace(line[textStart:textEnd]))
}
