// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"internal/testenv"

	rand "math/rand/v2"
)

type diffCase struct {
	name   string
	ptypes []string
	args   []string
	rtype  string
	body   string
}

func diffType(name string) *Type {
	t := BasicType(name)
	if t == nil {
		panic("unknown type " + name)
	}
	return t
}

func diffFuncSrc(c diffCase) string {
	var b strings.Builder
	b.WriteString("package p\nfunc " + c.name + "(")

	for i, pt := range c.ptypes {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString("p" + strconv.Itoa(i) + " " + pt)
	}
	b.WriteString(") " + c.rtype + " { " + c.body + " }")
	return b.String()
}
func diffOracleSrc(c diffCase) string {
	return diffFuncSrc(c)[len("package p\n"):]
}

func diffRejectSrc(c diffCase) string {
	return "package reject\n" + diffFuncSrc(c)[len("package p\n"):]
}

func diffCall(c diffCase) string {
	return c.name + "(" + strings.Join(c.args, ", ") + ")"
}

func diffOraclePrint(c diffCase, call string) string {
	switch c.rtype {
	case "bool":
		return "\tfmt.Printf(\"" + c.name + "|%v\\n\", " + call + ")\n"
	case "string":
		return "\tfmt.Printf(\"" + c.name + "|%q\\n\", " + call + ")\n"
	case "float32":
		return "\tfmt.Printf(\"" + c.name + "|%d\\n\", math.Float64bits(float64(" + call + ")))\n"
	case "float64":
		return "\tfmt.Printf(\"" + c.name + "|%d\\n\", math.Float64bits(" + call + "))\n"
	case "complex64":
		return "\t{c := " + call + "; fmt.Printf(\"" + c.name + "|%d+%d\\n\", math.Float64bits(float64(real(c))), math.Float64bits(float64(imag(c)))" + ")}\n"
	case "complex128":
		return "\t{c := " + call + "; fmt.Printf(\"" + c.name + "|%d+%d\\n\", math.Float64bits(real(c)), math.Float64bits(imag(c))" + ")}\n"
	default:
		return "\tfmt.Printf(\"" + c.name + "|%d\\n\", " + call + ")\n"
	}
}

func diffRender(v Value, rt string) string {
	switch rt {
	case "bool":
		return strconv.FormatBool(v.Bool())
	case "string":
		return strconv.Quote(v.Str())
	case "float32", "float64":
		return strconv.FormatUint(math.Float64bits(v.Float64()), 10)
	case "complex64", "complex128":
		c := v.Complex()
		return strconv.FormatUint(math.Float64bits(real(c)), 10) + "+" + strconv.FormatUint(math.Float64bits(imag(c)), 10)
	default:
		return v.BigInt().String()
	}
}

func diffConst(name, rtype, body string) diffCase {
	return diffCase{name: name, rtype: rtype, body: body}
}

func diffDyn(name, rtype string, ptypes, args []string, body string) diffCase {
	return diffCase{name: name, ptypes: ptypes, args: args, rtype: rtype, body: body}
}

var diffValueCases = []diffCase{
	diffConst("CIntAdd", "int", "return 3 + 4"),
	diffConst("CIntSub", "int", "return 3 - 10"),
	diffConst("CIntMul", "int", "return -3 * 4"),
	diffConst("CIntQuo", "int", "return -7 / 3"),
	diffConst("CIntQuoNeg", "int", "return 7 / -3"),
	diffConst("CIntRem", "int", "return -7 % 3"),
	diffConst("CIntRemNeg", "int", "return 7 % -3"),
	diffConst("CIntAnd", "int", "return 12 & 10"),
	diffConst("CIntOr", "int", "return 12 | 10"),
	diffConst("CIntXor", "int", "return 12 ^ 10"),
	diffConst("CIntAndNot", "int", "return 12 &^ 10"),
	diffConst("CIntShl", "int", "return 3 << 4"),
	diffConst("CIntShr", "int", "return 48 >> 4"),
	diffConst("CInt8Add", "int8", "return int8(100) + int8(27)"),
	diffConst("CInt8Min", "int8", "return int8(-128)"),
	diffConst("CUint8Max", "uint8", "return uint8(255)"),
	diffConst("CUint64Shl", "uint64", "return uint64(1) << uint64(63)"),
	diffConst("CUint64Max", "uint64", "return uint64(18446744073709551615)"),
	diffConst("CInt64Min", "int64", "return int64(-9223372036854775808)"),
	diffConst("CFloatAdd", "float64", "return 0.5 + 0.25"),
	diffConst("CFloatMul", "float64", "return 1.5 * 2"),
	diffConst("CFloatDiv", "float64", "return 7.0 / 2"),
	diffConst("CFloatLt", "bool", "return 1.5 < 2"),
	diffConst("CFloatEq", "bool", "return 2.0 == 2"),
	diffConst("CFloat32Add", "float32", "return float32(0.5) + float32(0.25)"),
	diffConst("CStringCat", "string", `return "a" + "b"`),
	diffConst("CStringLt", "bool", `return "a" < "b"`),
	diffConst("CStringEq", "bool", `return "xy" == "xy"`),
	diffConst("CRuneAdd", "int32", "return 'a' + 1"),
	diffConst("CRuneCmp", "bool", "return 'b' > 'a'"),
	diffConst("CComplexAdd", "complex128", "return (1 + 2i) + (3 + 4i)"),
	diffConst("CComplexMul", "complex128", "return (1 + 2i) * (3 + 4i)"),
	diffConst("CComplexEq", "bool", "return (1 + 2i) == (1 + 2i)"),
	diffConst("CBoolAnd", "bool", "return true && false"),
	diffConst("CBoolOr", "bool", "return true || false"),
	diffConst("CBoolNot", "bool", "return !true"),
	diffConst("CConvIntFloat", "int", "return int(1.0)"),
	diffConst("CConvInt8", "int8", "return int8(127)"),
	diffConst("CConvUint8", "uint8", "return uint8(255)"),
	diffConst("CConvFloat64", "float64", "return float64(1)"),
	diffConst("CConvFloat32", "float32", "return float32(1.5)"),
	diffConst("CConvFloatExact", "float64", "return float64(int64(1) << 53)"),
	diffConst("CConvStrInt", "string", "return string(65)"),
	diffConst("CConvStrEmoji", "string", "return string(0x1F600)"),
	diffConst("CConvIntRune", "int", "return int('a')"),
	diffConst("CConvFloatRune", "float64", "return float64('a')"),
	diffConst("CConvComplex", "complex128", "return complex(1, 2)"),
	diffConst("CConvComplex64", "complex64", "return complex64(complex(1.5, 2.5))"),
	diffConst("CBigFloat", "float64", "return float64(1 << 100)"),
	diffConst("CBigStr", "string", "return string(1 << 100)"),
	diffConst("CBigComplex", "complex128", "return complex(1 << 100, 0)"),
	diffConst("CLenStr", "int", `return len("abc")`),
	diffConst("CStrFromBytes", "string", `return string([]byte("abc"))`),
	diffConst("CStrFromRunes", "string", `return string([]rune("hé"))`),
	diffConst("CBytesRound", "string", `return string([]byte(string([]rune("hé"))))`),

	diffDyn("DInt8Wrap", "int8", []string{"int8", "int8"}, []string{"100", "100"}, "return p0 + p1"),
	diffDyn("DInt8MinWrap", "int8", []string{"int8", "int8"}, []string{"-128", "1"}, "return p0 - p1"),
	diffDyn("DUint8Wrap", "uint8", []string{"uint8", "uint8"}, []string{"255", "1"}, "return p0 + p1"),
	diffDyn("DInt64MinWrap", "int64", []string{"int64", "int64"}, []string{"-9223372036854775808", "1"}, "return p0 - p1"),
	diffDyn("DUint64Wrap", "uint64", []string{"uint64", "uint64"}, []string{"18446744073709551615", "1"}, "return p0 + p1"),
	diffDyn("DIntMulWrap", "int32", []string{"int32", "int32"}, []string{"100000", "100000"}, "return p0 * p1"),
	diffDyn("DShlWrap", "int8", []string{"int8", "uint"}, []string{"1", "7"}, "return p0 << p1"),
	diffDyn("DShlZero", "int8", []string{"int8", "uint"}, []string{"1", "8"}, "return p0 << p1"),
	diffDyn("DShlHuge", "int", []string{"int", "uint"}, []string{"1", "100"}, "return p0 << p1"),
	diffDyn("DShrArith", "int8", []string{"int8", "uint"}, []string{"-128", "2"}, "return p0 >> p1"),
	diffDyn("DShrLogic", "uint8", []string{"uint8", "uint"}, []string{"255", "1"}, "return p0 >> p1"),
	diffDyn("DFloatDivInf", "float64", []string{"float64", "float64"}, []string{"1", "0"}, "return p0 / p1"),
	diffDyn("DFloatDivNaN", "float64", []string{"float64", "float64"}, []string{"0", "0"}, "return p0 / p1"),
	diffDyn("DFloatDivNegInf", "float64", []string{"float64", "float64"}, []string{"-1", "0"}, "return p0 / p1"),
	diffDyn("DFloatDiv", "float64", []string{"float64", "float64"}, []string{"5", "2"}, "return p0 / p1"),
	diffDyn("DFloatMulInf", "float64", []string{"float64", "float64"}, []string{"1e308", "10"}, "return p0 * p1"),
	diffDyn("DFloatTrunc", "int", []string{"float64"}, []string{"1.9"}, "return int(p0)"),
	diffDyn("DFloatTruncNeg", "int", []string{"float64"}, []string{"-1.9"}, "return int(p0)"),
	diffDyn("DFloat32Add", "float32", []string{"float32", "float32"}, []string{"0.5", "0.25"}, "return p0 + p1"),
	diffDyn("DNaNNe", "bool", []string{"float64"}, []string{"0"}, "n := p0 / p0; return n != n"),
	diffDyn("DInfAdd", "float64", []string{"float64"}, []string{"0"}, "n := p0 / p0; i := 1 / n; return i + 1"),
	diffDyn("DStrCat", "string", []string{"string", "string"}, []string{`"a"`, `"b"`}, "return p0 + p1"),
	diffDyn("DStrCmp", "bool", []string{"string", "string"}, []string{`"a"`, `"b"`}, "return p0 < p1"),
	diffDyn("DComplexMul", "complex128", []string{"complex128", "complex128"}, []string{"1+2i", "3-4i"}, "return p0 * p1"),
	diffDyn("DComplexAdd", "complex128", []string{"complex128", "complex128"}, []string{"1+2i", "3-4i"}, "return p0 + p1"),
	diffDyn("DBoolAnd", "bool", []string{"bool", "bool"}, []string{"true", "false"}, "return p0 && p1"),
	diffDyn("DConvFloatInt", "int", []string{"float64"}, []string{"42.0"}, "return int(p0)"),
	diffDyn("DConvIntFloat", "float64", []string{"int64"}, []string{"100"}, "return float64(p0)"),
	diffDyn("DConvStrInt", "string", []string{"int"}, []string{"65"}, "return string(p0)"),
	diffDyn("DConvIntStr", "int", []string{"string"}, []string{`"abc"`}, "return len(p0)"),
}

var diffPanicCases = []diffCase{
	diffDyn("PIntDivZero", "int", []string{"int", "int"}, []string{"1", "0"}, "return p0 / p1"),
	diffDyn("PIntRemZero", "int", []string{"int", "int"}, []string{"1", "0"}, "return p0 % p1"),
	diffDyn("PMinDivNeg", "int64", []string{"int64", "int64"}, []string{"-9223372036854775808", "-1"}, "return p0 / p1"),
	diffDyn("PNegShift", "int", []string{"int", "int"}, []string{"1", "-1"}, "return p0 << p1"),
}

var diffRejectCases = []diffCase{
	diffConst("RInt8Overflow", "int8", "return int8(127) + 1"),
	diffConst("RIntOverflow", "int", "return 1 << 100"),
	diffConst("RInt64Overflow", "int64", "return 1 << 100"),
	diffConst("RFloat64Overflow", "float64", "return float64(1e400)"),
	diffConst("RFloat32Overflow", "float32", "return float32(1e400)"),
	diffConst("RComplexOverflow", "complex128", "return complex(1e400, 0)"),
	diffConst("RIntTruncate", "int", "return int(1.5)"),
	diffConst("RIntDivZero", "int", "return 1 / 0"),
	diffConst("RIntRemZero", "int", "return 1 % 0"),
	diffConst("RShiftTypedOverflow", "int8", "return int8(1) << 8"),
	diffConst("RBigShift", "float64", "return float64(1 << 1000)"),
	diffConst("RStrToInt", "int", `return int("a")`),
	diffConst("RBoolToInt", "int", "return int(true)"),
	diffConst("RIntToBool", "bool", "return bool(1)"),
	diffDyn("RComplexTypedInt", "complex128", []string{"int"}, []string{"2"}, "return complex(2, p0)"),
	diffDyn("RComplexTypedReal", "complex128", []string{"int"}, []string{"2"}, "return complex(p0, 2)"),
}

var diffGenIntTypes = []string{"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "int", "uint"}

var diffGenIntOps = []string{"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}

var diffGenFloatVals = []string{"0.5", "1.5", "100.25", "-3.75", "42", "7", "0.25"}

var diffGenStrVals = []string{`"a"`, `"b"`, `""`, `"xy"`, `"hé"`}

func diffGenIntLit(rng *rand.Rand, signed bool) string {
	if signed {
		return strconv.Itoa(rng.IntN(199) - 99)
	}
	return strconv.Itoa(rng.IntN(100))
}

func diffGenOne(rng *rand.Rand, i int) diffCase {
	name := "G" + strconv.Itoa(i)
	switch i % 10 {
	case 0, 1:
		t := diffGenIntTypes[rng.IntN(len(diffGenIntTypes))]
		signed := !strings.HasPrefix(t, "u")
		op := diffGenIntOps[rng.IntN(len(diffGenIntOps))]
		a := diffGenIntLit(rng, signed)
		b := diffGenIntLit(rng, signed)
		if op == "/" || op == "%" {
			for b == "0" {
				b = diffGenIntLit(rng, signed)
			}
		}
		if op == "<<" || op == ">>" {
			b = strconv.Itoa(rng.IntN(8))
			return diffDyn(name, t, []string{t, "uint"}, []string{a, b}, "return p0 "+op+" p1")
		}
		return diffDyn(name, t, []string{t, t}, []string{a, b}, "return p0 "+op+" p1")
	case 2:
		t := diffGenIntTypes[rng.IntN(8)]
		a := diffGenIntLit(rng, !strings.HasPrefix(t, "u"))
		b := strconv.Itoa(rng.IntN(8))
		op := []string{"<<", ">>"}[rng.IntN(2)]
		return diffDyn(name, t, []string{t, "uint"}, []string{a, b}, "return p0 "+op+" p1")
	case 3:
		t := []string{"float32", "float64"}[rng.IntN(2)]
		a := diffGenFloatVals[rng.IntN(len(diffGenFloatVals))]
		b := diffGenFloatVals[rng.IntN(len(diffGenFloatVals))]
		op := []string{"+", "-", "*", "/"}[rng.IntN(4)]
		if op == "/" && b == "0" {
			b = "0.5"
		}
		return diffDyn(name, t, []string{t, t}, []string{a, b}, "return p0 "+op+" p1")
	case 4:
		pairs := [][2]string{{"int", "float64"}, {"int64", "float64"}, {"float64", "int"}, {"int", "float32"}, {"int", "string"}, {"int32", "string"}}
		p := pairs[rng.IntN(len(pairs))]
		var arg string
		if strings.HasPrefix(p[0], "float") {
			arg = diffGenFloatVals[rng.IntN(len(diffGenFloatVals))]
		} else {
			arg = strconv.Itoa(rng.IntN(90) + 32)
		}
		return diffDyn(name, p[1], []string{p[0]}, []string{arg}, "return "+p[1]+"(p0)")
	case 5:
		a := diffGenStrVals[rng.IntN(len(diffGenStrVals))]
		b := diffGenStrVals[rng.IntN(len(diffGenStrVals))]
		if rng.IntN(2) == 0 {
			return diffConst(name, "string", "return "+a+" + "+b)
		}
		return diffConst(name, "bool", "return "+a+" == "+b)
	case 6:
		ops := []string{"&&", "||"}
		a := []string{"true", "false"}[rng.IntN(2)]
		b := []string{"true", "false"}[rng.IntN(2)]
		return diffConst(name, "bool", "return "+a+" "+ops[rng.IntN(2)]+" "+b)
	case 7:
		ai := strconv.Itoa(rng.IntN(21) - 10)
		bi := strconv.Itoa(rng.IntN(21) - 10)
		ci := strconv.Itoa(rng.IntN(21) - 10)
		di := strconv.Itoa(rng.IntN(21) - 10)
		op := []string{"+", "*", "-"}[rng.IntN(3)]
		return diffConst(name, "complex128", "return ("+ai+"+"+bi+"i) "+op+" ("+ci+"+"+di+"i)")
	case 8:
		n := rng.IntN(65) + 64
		k := rng.IntN(9)
		return diffConst(name, "float64", "return float64((1 << "+strconv.Itoa(n)+") + "+strconv.Itoa(k)+")")
	default:
		t := diffGenIntTypes[rng.IntN(len(diffGenIntTypes))]
		signed := !strings.HasPrefix(t, "u")
		a := diffGenIntLit(rng, signed)
		b := diffGenIntLit(rng, signed)
		op := []string{"==", "!=", "<", "<=", ">", ">="}[rng.IntN(6)]
		return diffConst(name, "bool", "return "+t+"("+a+") "+op+" "+t+"("+b+")")
	}
}

func diffGenRandom(rng *rand.Rand, n int) []diffCase {
	out := make([]diffCase, 0, n)
	for i := range n {
		out = append(out, diffGenOne(rng, i))
	}
	return out
}

func diffEvalCase(t *testing.T, cfg Config, c diffCase) (v Value, e *Error, panicked any) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			panicked = r
		}
	}()
	var ptypes []*Type
	for _, pt := range c.ptypes {
		ptypes = append(ptypes, diffType(pt))
	}
	h := newFakeHost(t)
	fn := parseFunc(t, diffFuncSrc(c), true, ptypes, []*Type{diffType(c.rtype)})
	h.funcs[c.name] = fn
	r, _, err := Eval(h, cfg, parseExpr(t, diffCall(c)))
	if err != nil {
		if ce, ok := err.(*Error); ok {
			return Value{}, ce, nil
		}
		panic(err)
	}
	return r, nil, nil
}

func diffRunTool(t *testing.T, dir, tool string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command(tool, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOTOOLCHAIN=local", "GOPROXY=off", "GOFLAGS=-buildvcs=false", "GOCACHE="+filepath.Join(dir, "gocache"))
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func diffWantMap(t *testing.T, out string) map[string]string {
	t.Helper()
	want := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		name, val, ok := strings.Cut(line, "|")
		if !ok {
			t.Fatalf("bad oracle line %q", line)
		}
		want[name] = val
	}
	return want
}

func TestDifferentialCTFEAgainstRuntime(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("differential oracle assumes amd64 widths")
	}
	testenv.MustHaveGoBuild(t)
	cfg := Config{MaxSteps: 100000, MaxValueBytes: 1 << 20, WordSize: 8}
	dir := t.TempDir()

	var prog strings.Builder
	prog.WriteString("package main\n\nimport (\n\t\"fmt\"\n\t\"math\"\n)\n\n")
	for _, c := range diffValueCases {
		prog.WriteString(diffOracleSrc(c) + "\n")
	}
	for _, c := range diffPanicCases {
		prog.WriteString(diffOracleSrc(c) + "\n")
	}
	prog.WriteString("func main() {\n")
	for _, c := range diffValueCases {
		prog.WriteString(diffOraclePrint(c, diffCall(c)))
	}
	for _, c := range diffPanicCases {
		call := diffCall(c)
		prog.WriteString("\tfunc() {\n\t\tdefer func() {\n\t\t\tif recover() != nil {\n\t\t\t\tfmt.Printf(\"" + c.name + "|panic\\n\")\n\t\t\t}\n\t\t}()\n")
		prog.WriteString(diffOraclePrint(c, call))
		prog.WriteString("\t}()\n")
	}
	prog.WriteString("}\n")
	progPath := filepath.Join(dir, "oracle.go")
	if err := os.WriteFile(progPath, []byte(prog.String()), 0666); err != nil {
		t.Fatal(err)
	}
	out, err := diffRunTool(t, dir, testenv.GoToolPath(t), "run", progPath)
	if err != nil {
		t.Fatalf("oracle run failed: %v\n%s", err, out)
	}
	want := diffWantMap(t, out)
	diffCheck := func(c diffCase, allowPanic bool) {
		t.Helper()
		v, cerr, panicked := diffEvalCase(t, cfg, c)
		if panicked != nil {
			t.Errorf("%s: CTFE panicked: %v", c.name, panicked)
			return
		}
		got, ok := want[c.name]
		if !ok {
			t.Errorf("%s: missing oracle output", c.name)
			return
		}
		if allowPanic && got == "panic" {
			if cerr == nil {
				t.Errorf("%s: oracle panicked but CTFE returned %s", c.name, diffRender(v, c.rtype))
			}
			return
		}
		if cerr != nil {
			t.Errorf("%s: CTFE error %v, oracle has %q", c.name, cerr, got)
			return
		}
		if rendered := diffRender(v, c.rtype); rendered != got {
			t.Errorf("%s = %s, oracle %s", c.name, rendered, got)
		}
	}
	mustMatch := func(c diffCase) { diffCheck(c, false) }
	mustFailClosed := func(c diffCase) { diffCheck(c, true) }
	for _, c := range diffValueCases {
		mustMatch(c)
	}
	for _, c := range diffPanicCases {
		mustFailClosed(c)
	}

	rejDir := filepath.Join(dir, "reject")
	if err := os.Mkdir(rejDir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rejDir, "go.mod"), []byte("module reject\n\ngo 1.27\n"), 0666); err != nil {
		t.Fatal(err)
	}
	for i, c := range diffRejectCases {
		src := diffRejectSrc(c)
		name := "r" + strconv.Itoa(i) + ".go"
		if err := os.WriteFile(filepath.Join(rejDir, name), []byte(src), 0666); err != nil {
			t.Fatal(err)
		}
		if cout, cerr := diffRunTool(t, rejDir, testenv.GoToolPath(t), "tool", "compile", name); cerr == nil {
			t.Errorf("%s: reject file %s compiled, want error", c.name, name)
		} else if !strings.Contains(cout, name) {
			t.Errorf("%s: oracle error lacks file %s:\n%s", c.name, name, cout)
		}
	}
	for _, c := range diffRejectCases {
		_, cerr, panicked := diffEvalCase(t, cfg, c)
		if panicked != nil {
			t.Errorf("%s: CTFE panicked: %v", c.name, panicked)
			continue
		}
		if cerr == nil {
			t.Errorf("%s: CTFE accepted, oracle rejects", c.name)
		}
	}

	rng := rand.New(rand.NewPCG(1234, 5678))
	randoms := diffGenRandom(rng, 120)
	var rprog strings.Builder
	rprog.WriteString("package main\n\nimport (\n\t\"fmt\"\n\t\"math\"\n)\n\n")
	for _, c := range randoms {
		rprog.WriteString(diffOracleSrc(c) + "\n")
	}
	rprog.WriteString("func main() {\n")
	for _, c := range randoms {
		rprog.WriteString(diffOraclePrint(c, diffCall(c)))
	}
	rprog.WriteString("}\n")
	rprogPath := filepath.Join(dir, "random.go")
	if err := os.WriteFile(rprogPath, []byte(rprog.String()), 0666); err != nil {
		t.Fatal(err)
	}
	rout, rerr := diffRunTool(t, dir, testenv.GoToolPath(t), "run", rprogPath)
	if rerr != nil {
		t.Fatalf("random oracle failed: %v\n%s", rerr, rout)
	}
	rwant := diffWantMap(t, rout)
	for _, c := range randoms {
		v, cerr, panicked := diffEvalCase(t, cfg, c)
		if panicked != nil {
			t.Errorf("%s (%s): CTFE panicked: %v", c.name, c.body, panicked)
			continue
		}
		if cerr != nil {
			t.Errorf("%s (%s): CTFE error %v", c.name, c.body, cerr)
			continue
		}
		got, ok := rwant[c.name]
		if !ok {
			t.Errorf("%s: missing oracle output", c.name)
			continue
		}
		if rendered := diffRender(v, c.rtype); rendered != got {
			t.Errorf("%s (%s) = %s, oracle %s", c.name, c.body, rendered, got)
		}
	}
}
