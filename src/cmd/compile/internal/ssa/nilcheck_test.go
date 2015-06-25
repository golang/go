package ssa

import (
	"strconv"
	"testing"
)

func BenchmarkNilCheckDeep1(b *testing.B)     { benchmarkNilCheckDeep(b, 1) }
func BenchmarkNilCheckDeep10(b *testing.B)    { benchmarkNilCheckDeep(b, 10) }
func BenchmarkNilCheckDeep100(b *testing.B)   { benchmarkNilCheckDeep(b, 100) }
func BenchmarkNilCheckDeep1000(b *testing.B)  { benchmarkNilCheckDeep(b, 1000) }
func BenchmarkNilCheckDeep10000(b *testing.B) { benchmarkNilCheckDeep(b, 10000) }

// benchmarkNilCheckDeep is a stress test of nilcheckelim.
// It uses the worst possible input: A linear string of
// nil checks, none of which can be eliminated.
// Run with multiple depths to observe big-O behavior.
func benchmarkNilCheckDeep(b *testing.B, depth int) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing

	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto(blockn(0)),
		),
	)
	for i := 0; i < depth; i++ {
		blocs = append(blocs,
			Bloc(blockn(i),
				Valu(ptrn(i), OpAddr, ptrType, 0, nil, "sb"),
				Valu(booln(i), OpIsNonNil, TypeBool, 0, nil, ptrn(i)),
				If(booln(i), blockn(i+1), "exit"),
			),
		)
	}
	blocs = append(blocs,
		Bloc(blockn(depth), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	c := NewConfig("amd64", DummyFrontend{b})
	fun := Fun(c, "entry", blocs...)

	CheckFunc(fun.f)
	b.SetBytes(int64(depth)) // helps for eyeballing linearity
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		nilcheckelim(fun.f)
	}
}

func blockn(n int) string { return "b" + strconv.Itoa(n) }
func ptrn(n int) string   { return "p" + strconv.Itoa(n) }
func booln(n int) string  { return "c" + strconv.Itoa(n) }
