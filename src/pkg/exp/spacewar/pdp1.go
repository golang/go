// Copyright (c) 1996 Barry Silverman, Brian Silverman, Vadim Gerasimov.
// Portions Copyright (c) 2009 The Go Authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// This package and spacewar.go implement a simple PDP-1 emulator
// complete enough to run the original PDP-1 video game Spacewar!
//
// They are a translation of the Java emulator pdp1.java in
// http://spacewar.oversigma.com/sources/sources.zip.
//
// See also the PDP-1 handbook at http://www.dbit.com/~greeng3/pdp1/pdp1.html
//
// http://spacewar.oversigma.com/readme.html reads:
//
//	Spacewar! was conceived in 1961 by Martin Graetz, Stephen Russell,
//	and Wayne Wiitanen. It was first realized on the PDP-1 in 1962 by
//	Stephen Russell, Peter Samson, Dan Edwards, and Martin Graetz,
//	together with Alan Kotok, Steve Piner, and Robert A Saunders.
//	Spacewar! is in the public domain, but this credit paragraph must
//	accompany all distributed versions of the program.
//
//	This is the original version! Martin Graetz provided us with a
//	printed version of the source. We typed in in again - it was about
//	40 pages long - and re-assembled it with a PDP-1 assembler written
//	in PERL. The resulting binary runs on a PDP-1 emulator written as
//	a Java applet. The code is extremely faithful to the original. There
//	are only two changes. 1)The spaceships have been made bigger and
//	2) The overall timing has been special cased to deal with varying
//	machine speeds.
//
//	The "a", "s", "d", "f" keys control one of the spaceships. The "k",
//	"l", ";", "'" keys control the other. The controls are spin one
//	way, spin the other, thrust, and fire.
//
//	Barry Silverman
//	Brian Silverman
//	Vadim Gerasimov
//
package pdp1

import (
	"bufio";
	"fmt";
	"os";
	"io";
)

type Word uint32

const mask = 0777777;
const sign = 0400000;

const (
	_ = iota;	// 00
	opAND;
	opIOR;
	opXOR;
	opXCT;
	_;
	_;
	opCALJDA;

	opLAC;	// 10
	opLIO;
	opDAC;
	opDAP;
	_;
	opDIO;
	opDZM;
	_;

	opADD;	// 20
	opSUB;
	opIDX;
	opISP;
	opSAD;
	opSAS;
	opMUS;
	opDIS;

	opJMP;	// 30
	opJSP;
	opSKP;
	opSFT;
	opLAW;
	opIOT;
	_;
	opOPR;
)

// A Trapper represents an object with a Trap method.
// The machine calls the Trap method to implement the
// PDP-1 IOT instruction.
type Trapper interface {
	Trap(y Word)
}

// An M represents the machine state of a PDP-1.
// Clients can set Display to install an output device.
type M struct {
	AC, IO, PC, OV Word;
	Mem [010000]Word;
	Flag [7]bool;
	Sense [7]bool;
	Halt bool;
}


// Step runs a single machine instruction.
func (m *M) Step(t Trapper) os.Error {
	inst := m.Mem[m.PC];
	m.PC++;
	return m.run(inst, t);
}

// Normalize actual 32-bit integer i to 18-bit ones-complement integer.
// Interpret mod 0777777, because 0777777 == -0 == +0 == 0000000.
func norm(i Word) Word {
	i += i>>18;
	i &= mask;
	if i == mask {
		i = 0;
	}
	return i;
}

type UnknownInstrError struct {
	Inst Word;
	PC Word;
}

func (e UnknownInstrError) String() string {
	return fmt.Sprintf("unknown instruction %06o at %06o", e.Inst, e.PC);
}

type HaltError Word

func (e HaltError) String() string {
	return fmt.Sprintf("executed HLT instruction at %06o", e);
}

type LoopError Word

func (e LoopError) String() string {
	return fmt.Sprintf("indirect load looping at %06o", e);
}

func (m *M) run(inst Word, t Trapper) os.Error {
	ib, y := (inst>>12)&1, inst&07777;
	op := inst>>13;
	if op < opSKP && op != opCALJDA {
		for n := 0; ib != 0; n++ {
			if n > 07777 {
				return LoopError(m.PC-1);
			}
			ib = (m.Mem[y]>>12) & 1;
			y = m.Mem[y] & 07777;
		}
	}

	switch op {
	case opAND:
		m.AC &= m.Mem[y];
	case opIOR:
		m.AC |= m.Mem[y];
	case opXOR:
		m.AC ^= m.Mem[y];
	case opXCT:
		m.run(m.Mem[y], t);
	case opCALJDA:
		a := y;
		if ib == 0 {
			a = 64;
		}
		m.Mem[a] = m.AC;
		m.AC = (m.OV<<17) + m.PC;
		m.PC = a + 1;
	case opLAC:
		m.AC = m.Mem[y];
	case opLIO:
		m.IO = m.Mem[y];
	case opDAC:
		m.Mem[y] = m.AC;
	case opDAP:
		m.Mem[y] = m.Mem[y]&0770000 | m.AC&07777;
	case opDIO:
		m.Mem[y] = m.IO;
	case opDZM:
		m.Mem[y] = 0;
	case opADD:
		m.AC += m.Mem[y];
		m.OV = m.AC>>18;
		m.AC = norm(m.AC);
	case opSUB:
		diffSigns := (m.AC ^ m.Mem[y])>>17 == 1;
		m.AC += m.Mem[y]^mask;
		m.AC = norm(m.AC);
		if diffSigns && m.Mem[y]>>17 == m.AC>>17 {
			m.OV = 1;
		}
	case opIDX:
		m.AC = norm(m.Mem[y]+1);
		m.Mem[y] = m.AC;
	case opISP:
		m.AC = norm(m.Mem[y]+1);
		m.Mem[y] = m.AC;
		if m.AC&sign == 0 {
			m.PC++;
		}
	case opSAD:
		if m.AC != m.Mem[y] {
			m.PC++;
		}
	case opSAS:
		if m.AC == m.Mem[y] {
			m.PC++;
		}
	case opMUS:
		if m.IO&1 == 1 {
			m.AC += m.Mem[y];
			m.AC = norm(m.AC)
		}
		m.IO = (m.IO>>1 | m.AC<<17) & mask;
		m.AC >>= 1;
	case opDIS:
		m.AC, m.IO = (m.AC<<1 | m.IO>>17) & mask,
			((m.IO<<1 | m.AC>>17) & mask) ^ 1;
		if m.IO&1 == 1 {
			m.AC = m.AC + (m.Mem[y]^mask);
		} else {
			m.AC = m.AC + 1 + m.Mem[y];
		}
		m.AC = norm(m.AC);
	case opJMP:
		m.PC = y;
	case opJSP:
		m.AC = (m.OV<<17) + m.PC;
		m.PC = y;
	case opSKP:
		cond := y&0100 == 0100 && m.AC == 0
			|| y&0200 == 0200 && m.AC>>17 == 0
			|| y&0400 == 0400 && m.AC>>17 == 1
			|| y&01000 == 01000 && m.OV == 0
			|| y&02000 == 02000 && m.IO>>17 == 0
			|| y&7 != 0 && !m.Flag[y&7]
			|| y&070 != 0 && !m.Sense[(y&070)>>3]
			|| y&070 == 010;
		if (ib==0) == cond {
			m.PC++;
		}
		if y&01000 == 01000 {
			m.OV = 0;
		}
	case opSFT:
		for count := inst&0777; count != 0; count >>= 1 {
			if count&1 == 0 {
				continue;
			}
			switch (inst>>9)&017 {
			case 001:	// rotate AC left
				m.AC = (m.AC<<1 | m.AC>>17) & mask;
			case 002:	// rotate IO left
				m.IO = (m.IO<<1 | m.IO>>17) & mask;
			case 003:	// rotate AC and IO left.
				w := uint64(m.AC)<<18 | uint64(m.IO);
				w = w<<1 | w>>35;
				m.AC = Word(w>>18) & mask;
				m.IO = Word(w) & mask;
			case 005:	// shift AC left (excluding sign bit)
				m.AC = (m.AC<<1 | m.AC>>17)&mask&^sign | m.AC&sign;
			case 006:	// shift IO left (excluding sign bit)
				m.IO = (m.IO<<1 | m.IO>>17)&mask&^sign | m.IO&sign;
			case 007:	// shift AC and IO left (excluding AC's sign bit)
				w := uint64(m.AC)<<18 | uint64(m.IO);
				w = w<<1 | w>>35;
				m.AC = Word(w>>18)&mask&^sign | m.AC&sign;
				m.IO = Word(w)&mask&^sign | m.AC&sign;
			case 011:	// rotate AC right
				m.AC = (m.AC>>1 | m.AC<<17) & mask;
			case 012:	// rotate IO right
				m.IO = (m.IO>>1 | m.IO<<17) & mask;
			case 013:	// rotate AC and IO right
				w := uint64(m.AC)<<18 | uint64(m.IO);
				w = w>>1 | w<<35;
				m.AC = Word(w>>18) & mask;
				m.IO = Word(w) & mask;
			case 015:	// shift AC right (excluding sign bit)
				m.AC = m.AC>>1 | m.AC&sign;
			case 016:	// shift IO right (excluding sign bit)
				m.IO = m.IO>>1 | m.IO&sign;
			case 017:	// shift AC and IO right (excluding AC's sign bit)
				w := uint64(m.AC)<<18 | uint64(m.IO);
				w = w>>1;
				m.AC = Word(w>>18) | m.AC&sign;
				m.IO = Word(w) & mask;
			default:
				goto Unknown;
			}
		}
	case opLAW:
		if ib == 0 {
			m.AC = y;
		} else {
			m.AC = y^mask;
		}
	case opIOT:
		t.Trap(y);
	case opOPR:
		if y&0200 == 0200 {
			m.AC = 0;
		}
		if y&04000 == 04000 {
			m.IO = 0;
		}
		if y&01000 == 01000 {
			m.AC ^= mask;
		}
		if y&0400 == 0400 {
			m.PC--;
			return HaltError(m.PC);
		}
		switch i, f := y&7, y&010==010; {
		case i == 7:
			for i := 2; i < 7; i++ {
				m.Flag[i] = f;
			}
		case i >= 2:
			m.Flag[i] = f;
		}
	default:
	Unknown:
		return UnknownInstrError{inst, m.PC-1};
	}
	return nil;
}

// Load loads the machine's memory from a text input file
// listing octal address-value pairs, one per line, matching the
// regular expression ^[ +]([0-7]+)\t([0-7]+).
func (m *M) Load(r io.Reader) os.Error {
	b := bufio.NewReader(r);
	for {
		line, err := b.ReadString('\n');
		if err != nil {
			if err != os.EOF {
				return err;
			}
			break;
		}
		// look for ^[ +]([0-9]+)\t([0-9]+)
		if line[0] != ' ' && line[0] != '+' {
			continue;
		}
		i := 1;
		a := Word(0);
		for ; i < len(line) && '0' <= line[i] && line[i] <= '7'; i++ {
			a = a*8 + Word(line[i] - '0');
		}
		if i >= len(line) || line[i] != '\t' || i == 1{
			continue;
		}
		v := Word(0);
		j := i;
		for i++; i < len(line) && '0' <= line[i] && line[i] <= '7'; i++ {
			v = v*8 + Word(line[i] - '0');
		}
		if i == j {
			continue;
		}
		m.Mem[a] = v;
	}
	return nil;
}

