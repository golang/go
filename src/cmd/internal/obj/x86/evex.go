// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/internal/obj"
	"errors"
	"fmt"
	"strings"
)

// evexBits stores EVEX prefix info that is used during instruction encoding.
type evexBits struct {
	b1 byte // [W1mmLLpp]
	b2 byte // [NNNbbZRS]

	// Associated instruction opcode.
	opcode byte
}

// newEVEXBits creates evexBits object from enc bytes at z position.
func newEVEXBits(z int, enc *opBytes) evexBits {
	return evexBits{
		b1:     enc[z+0],
		b2:     enc[z+1],
		opcode: enc[z+2],
	}
}

// P returns EVEX.pp value.
func (evex evexBits) P() byte { return (evex.b1 & evexP) >> 0 }

// L returns EVEX.L'L value.
func (evex evexBits) L() byte { return (evex.b1 & evexL) >> 2 }

// M returns EVEX.mm value.
func (evex evexBits) M() byte { return (evex.b1 & evexM) >> 4 }

// W returns EVEX.W value.
func (evex evexBits) W() byte { return (evex.b1 & evexW) >> 7 }

// BroadcastEnabled reports whether BCST suffix is permitted.
func (evex evexBits) BroadcastEnabled() bool {
	return evex.b2&evexBcst != 0
}

// ZeroingEnabled reports whether Z suffix is permitted.
func (evex evexBits) ZeroingEnabled() bool {
	return (evex.b2&evexZeroing)>>2 != 0
}

// RoundingEnabled reports whether RN_SAE, RZ_SAE, RD_SAE and RU_SAE suffixes
// are permitted.
func (evex evexBits) RoundingEnabled() bool {
	return (evex.b2&evexRounding)>>1 != 0
}

// SaeEnabled reports whether SAE suffix is permitted.
func (evex evexBits) SaeEnabled() bool {
	return (evex.b2&evexSae)>>0 != 0
}

// DispMultiplier returns displacement multiplier that is calculated
// based on tuple type, EVEX.W and input size.
// If embedded broadcast is used, bcst should be true.
func (evex evexBits) DispMultiplier(bcst bool) int32 {
	if bcst {
		switch evex.b2 & evexBcst {
		case evexBcstN4:
			return 4
		case evexBcstN8:
			return 8
		}
		return 1
	}

	switch evex.b2 & evexN {
	case evexN1:
		return 1
	case evexN2:
		return 2
	case evexN4:
		return 4
	case evexN8:
		return 8
	case evexN16:
		return 16
	case evexN32:
		return 32
	case evexN64:
		return 64
	case evexN128:
		return 128
	}
	return 1
}

// EVEX is described by using 2-byte sequence.
// See evexBits for more details.
const (
	evexW   = 0x80 // b1[W... ....]
	evexWIG = 0 << 7
	evexW0  = 0 << 7
	evexW1  = 1 << 7

	evexM    = 0x30 // b2[..mm ...]
	evex0F   = 1 << 4
	evex0F38 = 2 << 4
	evex0F3A = 3 << 4

	evexL   = 0x0C // b1[.... LL..]
	evexLIG = 0 << 2
	evex128 = 0 << 2
	evex256 = 1 << 2
	evex512 = 2 << 2

	evexP  = 0x03 // b1[.... ..pp]
	evex66 = 1 << 0
	evexF3 = 2 << 0
	evexF2 = 3 << 0

	// Precalculated Disp8 N value.
	// N acts like a multiplier for 8bit displacement.
	// Note that some N are not used, but their bits are reserved.
	evexN    = 0xE0 // b2[NNN. ....]
	evexN1   = 0 << 5
	evexN2   = 1 << 5
	evexN4   = 2 << 5
	evexN8   = 3 << 5
	evexN16  = 4 << 5
	evexN32  = 5 << 5
	evexN64  = 6 << 5
	evexN128 = 7 << 5

	// Disp8 for broadcasts.
	evexBcst   = 0x18 // b2[...b b...]
	evexBcstN4 = 1 << 3
	evexBcstN8 = 2 << 3

	// Flags that permit certain AVX512 features.
	// It's semantically illegal to combine evexZeroing and evexSae.
	evexZeroing         = 0x4 // b2[.... .Z..]
	evexZeroingEnabled  = 1 << 2
	evexRounding        = 0x2 // b2[.... ..R.]
	evexRoundingEnabled = 1 << 1
	evexSae             = 0x1 // b2[.... ...S]
	evexSaeEnabled      = 1 << 0
)

// compressedDisp8 calculates EVEX compressed displacement, if applicable.
func compressedDisp8(disp, elemSize int32) (disp8 byte, ok bool) {
	if disp%elemSize == 0 {
		v := disp / elemSize
		if v >= -128 && v <= 127 {
			return byte(v), true
		}
	}
	return 0, false
}

// evexZcase reports whether given Z-case belongs to EVEX group.
func evexZcase(zcase uint8) bool {
	return zcase > Zevex_first && zcase < Zevex_last
}

// evexSuffixBits carries instruction EVEX suffix set flags.
//
// Examples:
//
//	"RU_SAE.Z" => {rounding: 3, zeroing: true}
//	"Z" => {zeroing: true}
//	"BCST" => {broadcast: true}
//	"SAE.Z" => {sae: true, zeroing: true}
type evexSuffix struct {
	rounding  byte
	sae       bool
	zeroing   bool
	broadcast bool
}

// Rounding control values.
// Match exact value for EVEX.L'L field (with exception of rcUnset).
const (
	rcRNSAE = 0 // Round towards nearest
	rcRDSAE = 1 // Round towards -Inf
	rcRUSAE = 2 // Round towards +Inf
	rcRZSAE = 3 // Round towards zero
	rcUnset = 4
)

// newEVEXSuffix returns proper zero value for evexSuffix.
func newEVEXSuffix() evexSuffix {
	return evexSuffix{rounding: rcUnset}
}

// evexSuffixMap maps obj.X86suffix to its decoded version.
// Filled during init().
var evexSuffixMap [255]evexSuffix

func init() {
	// Decode all valid suffixes for later use.
	for i := range opSuffixTable {
		suffix := newEVEXSuffix()
		parts := strings.Split(opSuffixTable[i], ".")
		for j := range parts {
			switch parts[j] {
			case "Z":
				suffix.zeroing = true
			case "BCST":
				suffix.broadcast = true
			case "SAE":
				suffix.sae = true

			case "RN_SAE":
				suffix.rounding = rcRNSAE
			case "RD_SAE":
				suffix.rounding = rcRDSAE
			case "RU_SAE":
				suffix.rounding = rcRUSAE
			case "RZ_SAE":
				suffix.rounding = rcRZSAE
			}
		}
		evexSuffixMap[i] = suffix
	}
}

// toDisp8 tries to convert disp to proper 8-bit displacement value.
func toDisp8(disp int32, p *obj.Prog, asmbuf *AsmBuf) (disp8 byte, ok bool) {
	if asmbuf.evexflag {
		bcst := evexSuffixMap[p.Scond].broadcast
		elemSize := asmbuf.evex.DispMultiplier(bcst)
		return compressedDisp8(disp, elemSize)
	}
	return byte(disp), disp >= -128 && disp < 128
}

// EncodeRegisterRange packs [reg0-reg1] list into 64-bit value that
// is intended to be stored inside obj.Addr.Offset with TYPE_REGLIST.
func EncodeRegisterRange(reg0, reg1 int16) int64 {
	return (int64(reg0) << 0) |
		(int64(reg1) << 16) |
		obj.RegListX86Lo
}

// decodeRegisterRange unpacks [reg0-reg1] list from 64-bit value created by EncodeRegisterRange.
func decodeRegisterRange(list int64) (reg0, reg1 int) {
	return int((list >> 0) & 0xFFFF),
		int((list >> 16) & 0xFFFF)
}

// ParseSuffix handles the special suffix for the 386/AMD64.
// Suffix bits are stored into p.Scond.
//
// Leading "." in cond is ignored.
func ParseSuffix(p *obj.Prog, cond string) error {
	cond = strings.TrimPrefix(cond, ".")

	suffix := newOpSuffix(cond)
	if !suffix.IsValid() {
		return inferSuffixError(cond)
	}

	p.Scond = uint8(suffix)
	return nil
}

// inferSuffixError returns non-nil error that describes what could be
// the cause of suffix parse failure.
//
// At the point this function is executed there is already assembly error,
// so we can burn some clocks to construct good error message.
//
// Reported issues:
//   - duplicated suffixes
//   - illegal rounding/SAE+broadcast combinations
//   - unknown suffixes
//   - misplaced suffix (e.g. wrong Z suffix position)
func inferSuffixError(cond string) error {
	suffixSet := make(map[string]bool)  // Set for duplicates detection.
	unknownSet := make(map[string]bool) // Set of unknown suffixes.
	hasBcst := false
	hasRoundSae := false
	var msg []string // Error message parts

	suffixes := strings.Split(cond, ".")
	for i, suffix := range suffixes {
		switch suffix {
		case "Z":
			if i != len(suffixes)-1 {
				msg = append(msg, "Z suffix should be the last")
			}
		case "BCST":
			hasBcst = true
		case "SAE", "RN_SAE", "RZ_SAE", "RD_SAE", "RU_SAE":
			hasRoundSae = true
		default:
			if !unknownSet[suffix] {
				msg = append(msg, fmt.Sprintf("unknown suffix %q", suffix))
			}
			unknownSet[suffix] = true
		}

		if suffixSet[suffix] {
			msg = append(msg, fmt.Sprintf("duplicate suffix %q", suffix))
		}
		suffixSet[suffix] = true
	}

	if hasBcst && hasRoundSae {
		msg = append(msg, "can't combine rounding/SAE and broadcast")
	}

	if len(msg) == 0 {
		return errors.New("bad suffix combination")
	}
	return errors.New(strings.Join(msg, "; "))
}

// opSuffixTable is a complete list of possible opcode suffix combinations.
// It "maps" uint8 suffix bits to their string representation.
// With the exception of first and last elements, order is not important.
var opSuffixTable = [...]string{
	"", // Map empty suffix to empty string.

	"Z",

	"SAE",
	"SAE.Z",

	"RN_SAE",
	"RZ_SAE",
	"RD_SAE",
	"RU_SAE",
	"RN_SAE.Z",
	"RZ_SAE.Z",
	"RD_SAE.Z",
	"RU_SAE.Z",

	"BCST",
	"BCST.Z",

	"<bad suffix>",
}

// opSuffix represents instruction opcode suffix.
// Compound (multi-part) suffixes expressed with single opSuffix value.
//
// uint8 type is used to fit obj.Prog.Scond.
type opSuffix uint8

// badOpSuffix is used to represent all invalid suffix combinations.
const badOpSuffix = opSuffix(len(opSuffixTable) - 1)

// newOpSuffix returns opSuffix object that matches suffixes string.
//
// If no matching suffix is found, special "invalid" suffix is returned.
// Use IsValid method to check against this case.
func newOpSuffix(suffixes string) opSuffix {
	for i := range opSuffixTable {
		if opSuffixTable[i] == suffixes {
			return opSuffix(i)
		}
	}
	return badOpSuffix
}

// IsValid reports whether suffix is valid.
// Empty suffixes are valid.
func (suffix opSuffix) IsValid() bool {
	return suffix != badOpSuffix
}

// String returns suffix printed representation.
//
// It matches the string that was used to create suffix with NewX86Suffix()
// for valid suffixes.
// For all invalid suffixes, special marker is returned.
func (suffix opSuffix) String() string {
	return opSuffixTable[suffix]
}
