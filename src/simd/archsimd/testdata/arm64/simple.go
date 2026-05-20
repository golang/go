//go:build goexperiment.simd

// NEON SIMD example
// Run with: GOEXPERIMENT=simd go run simple.go
package main

import (
	"fmt"
	"os"
	"simd/archsimd"
)

//go:noinline
func testFloat32x4() {
	fmt.Println("=== Float32x4 ===")

	a := [4]float32{1.0, 2.0, 3.0, 4.0}
	b := [4]float32{5.0, 6.0, 7.0, 8.0}

	va := archsimd.LoadFloat32x4Array(&a)
	vb := archsimd.LoadFloat32x4Array(&b)

	sum := va.Add(vb)
	prod := va.Mul(vb)

	var sumOut [4]float32
	var prodOut [4]float32
	sum.StoreArray(&sumOut)
	prod.StoreArray(&prodOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
	fmt.Printf("a * b:  %v\n", prodOut)
}

//go:noinline
func testFloat64x2() {
	fmt.Println("\n=== Float64x2 ===")

	a := [2]float64{10.5, 20.5}
	b := [2]float64{2.0, 4.0}

	va := archsimd.LoadFloat64x2Array(&a)
	vb := archsimd.LoadFloat64x2Array(&b)

	sum := va.Add(vb)
	prod := va.Mul(vb)

	var sumOut [2]float64
	var prodOut [2]float64
	sum.StoreArray(&sumOut)
	prod.StoreArray(&prodOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
	fmt.Printf("a * b:  %v\n", prodOut)
}

//go:noinline
func testInt32x4() {
	fmt.Println("\n=== Int32x4 ===")

	a := [4]int32{10, 20, 30, 40}
	b := [4]int32{5, 6, 7, 8}

	va := archsimd.LoadInt32x4Array(&a)
	vb := archsimd.LoadInt32x4Array(&b)

	sum := va.Add(vb)
	prod := va.Mul(vb)

	var sumOut [4]int32
	var prodOut [4]int32
	sum.StoreArray(&sumOut)
	prod.StoreArray(&prodOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
	fmt.Printf("a * b:  %v\n", prodOut)
}

//go:noinline
func testInt64x2() {
	fmt.Println("\n=== Int64x2 ===")

	a := [2]int64{100, 200}
	b := [2]int64{50, 75}

	va := archsimd.LoadInt64x2Array(&a)
	vb := archsimd.LoadInt64x2Array(&b)

	sum := va.Add(vb)

	var sumOut [2]int64
	sum.StoreArray(&sumOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
}

//go:noinline
func testInt16x8() {
	fmt.Println("\n=== Int16x8 ===")

	a := [8]int16{10, 20, 30, 40, 50, 60, 70, 80}
	b := [8]int16{2, 3, 4, 5, 6, 7, 8, 9}

	va := archsimd.LoadInt16x8Array(&a)
	vb := archsimd.LoadInt16x8Array(&b)

	sum := va.Add(vb)
	prod := va.Mul(vb)

	var sumOut [8]int16
	var prodOut [8]int16
	sum.StoreArray(&sumOut)
	prod.StoreArray(&prodOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
	fmt.Printf("a * b:  %v\n", prodOut)
}

//go:noinline
func testInt8x16() {
	fmt.Println("\n=== Int8x16 ===")

	a := [16]int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b := [16]int8{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}

	va := archsimd.LoadInt8x16Array(&a)
	vb := archsimd.LoadInt8x16Array(&b)

	sum := va.Add(vb)
	prod := va.Mul(vb)

	var sumOut [16]int8
	var prodOut [16]int8
	sum.StoreArray(&sumOut)
	prod.StoreArray(&prodOut)

	fmt.Printf("a:      %v\n", a)
	fmt.Printf("b:      %v\n", b)
	fmt.Printf("a + b:  %v\n", sumOut)
	fmt.Printf("a * b:  %v\n", prodOut)
}

func main() {
	testFloat32x4()
	testFloat64x2()
	testInt32x4()
	testInt64x2()
	testInt16x8()
	testInt8x16()

	// Test validation - return non-zero on unexpected results
	fail := false

	// Test Float32x4 Add and Mul
	a32 := [4]float32{1.0, 2.0, 3.0, 4.0}
	b32 := [4]float32{5.0, 6.0, 7.0, 8.0}
	va32 := archsimd.LoadFloat32x4Array(&a32)
	vb32 := archsimd.LoadFloat32x4Array(&b32)
	sum32 := va32.Add(vb32)
	prod32 := va32.Mul(vb32)
	var sumOut32 [4]float32
	var prodOut32 [4]float32
	sum32.StoreArray(&sumOut32)
	prod32.StoreArray(&prodOut32)

	expectedSum32 := [4]float32{6.0, 8.0, 10.0, 12.0}
	expectedProd32 := [4]float32{5.0, 12.0, 21.0, 32.0}
	for i := range sumOut32 {
		if sumOut32[i] != expectedSum32[i] {
			fmt.Printf("Float32x4 Add test failed: expected %v, got %v\n", expectedSum32, sumOut32)
			fail = true
			break
		}
		if prodOut32[i] != expectedProd32[i] {
			fmt.Printf("Float32x4 Mul test failed: expected %v, got %v\n", expectedProd32, prodOut32)
			fail = true
			break
		}
	}

	// Test Float64x2 Add and Mul
	a64 := [2]float64{10.5, 20.5}
	b64 := [2]float64{2.0, 4.0}
	va64 := archsimd.LoadFloat64x2Array(&a64)
	vb64 := archsimd.LoadFloat64x2Array(&b64)
	sum64 := va64.Add(vb64)
	prod64 := va64.Mul(vb64)
	var sumOut64 [2]float64
	var prodOut64 [2]float64
	sum64.StoreArray(&sumOut64)
	prod64.StoreArray(&prodOut64)

	expectedSum64 := [2]float64{12.5, 24.5}
	expectedProd64 := [2]float64{21.0, 82.0}
	for i := range sumOut64 {
		if sumOut64[i] != expectedSum64[i] {
			fmt.Printf("Float64x2 Add test failed: expected %v, got %v\n", expectedSum64, sumOut64)
			fail = true
			break
		}
		if prodOut64[i] != expectedProd64[i] {
			fmt.Printf("Float64x2 Mul test failed: expected %v, got %v\n", expectedProd64, prodOut64)
			fail = true
			break
		}
	}

	// Test Int32x4 Add and Mul
	a_i32 := [4]int32{10, 20, 30, 40}
	b_i32 := [4]int32{5, 6, 7, 8}
	va_i32 := archsimd.LoadInt32x4Array(&a_i32)
	vb_i32 := archsimd.LoadInt32x4Array(&b_i32)
	sum_i32 := va_i32.Add(vb_i32)
	prod_i32 := va_i32.Mul(vb_i32)
	var sumOut_i32 [4]int32
	var prodOut_i32 [4]int32
	sum_i32.StoreArray(&sumOut_i32)
	prod_i32.StoreArray(&prodOut_i32)

	expectedSum_i32 := [4]int32{15, 26, 37, 48}
	expectedProd_i32 := [4]int32{50, 120, 210, 320}
	for i := range sumOut_i32 {
		if sumOut_i32[i] != expectedSum_i32[i] {
			fmt.Printf("Int32x4 Add test failed: expected %v, got %v\n", expectedSum_i32, sumOut_i32)
			fail = true
			break
		}
		if prodOut_i32[i] != expectedProd_i32[i] {
			fmt.Printf("Int32x4 Mul test failed: expected %v, got %v\n", expectedProd_i32, prodOut_i32)
			fail = true
			break
		}
	}

	// Test Int64x2 Add (no Mul for 64-bit integers)
	a_i64 := [2]int64{100, 200}
	b_i64 := [2]int64{50, 75}
	va_i64 := archsimd.LoadInt64x2Array(&a_i64)
	vb_i64 := archsimd.LoadInt64x2Array(&b_i64)
	sum_i64 := va_i64.Add(vb_i64)
	var sumOut_i64 [2]int64
	sum_i64.StoreArray(&sumOut_i64)

	expectedSum_i64 := [2]int64{150, 275}
	for i := range sumOut_i64 {
		if sumOut_i64[i] != expectedSum_i64[i] {
			fmt.Printf("Int64x2 Add test failed: expected %v, got %v\n", expectedSum_i64, sumOut_i64)
			fail = true
			break
		}
	}

	// Test Int16x8 Add and Mul
	a_i16 := [8]int16{10, 20, 30, 40, 50, 60, 70, 80}
	b_i16 := [8]int16{2, 3, 4, 5, 6, 7, 8, 9}
	va_i16 := archsimd.LoadInt16x8Array(&a_i16)
	vb_i16 := archsimd.LoadInt16x8Array(&b_i16)
	sum_i16 := va_i16.Add(vb_i16)
	prod_i16 := va_i16.Mul(vb_i16)
	var sumOut_i16 [8]int16
	var prodOut_i16 [8]int16
	sum_i16.StoreArray(&sumOut_i16)
	prod_i16.StoreArray(&prodOut_i16)

	expectedSum_i16 := [8]int16{12, 23, 34, 45, 56, 67, 78, 89}
	expectedProd_i16 := [8]int16{20, 60, 120, 200, 300, 420, 560, 720}
	for i := range sumOut_i16 {
		if sumOut_i16[i] != expectedSum_i16[i] {
			fmt.Printf("Int16x8 Add test failed: expected %v, got %v\n", expectedSum_i16, sumOut_i16)
			fail = true
			break
		}
		if prodOut_i16[i] != expectedProd_i16[i] {
			fmt.Printf("Int16x8 Mul test failed: expected %v, got %v\n", expectedProd_i16, prodOut_i16)
			fail = true
			break
		}
	}

	// Test Int8x16 Add and Mul
	a_i8 := [16]int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b_i8 := [16]int8{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	va_i8 := archsimd.LoadInt8x16Array(&a_i8)
	vb_i8 := archsimd.LoadInt8x16Array(&b_i8)
	sum_i8 := va_i8.Add(vb_i8)
	prod_i8 := va_i8.Mul(vb_i8)
	var sumOut_i8 [16]int8
	var prodOut_i8 [16]int8
	sum_i8.StoreArray(&sumOut_i8)
	prod_i8.StoreArray(&prodOut_i8)

	expectedSum_i8 := [16]int8{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
	expectedProd_i8 := [16]int8{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
	for i := range sumOut_i8 {
		if sumOut_i8[i] != expectedSum_i8[i] {
			fmt.Printf("Int8x16 Add test failed: expected %v, got %v\n", expectedSum_i8, sumOut_i8)
			fail = true
			break
		}
		if prodOut_i8[i] != expectedProd_i8[i] {
			fmt.Printf("Int8x16 Mul test failed: expected %v, got %v\n", expectedProd_i8, prodOut_i8)
			fail = true
			break
		}
	}

	// Int8x16 GetElem/SetElem
	{
		a := [16]int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadInt8x16Array(&a)

		// Test GetElem
		for i := uint8(0); i < 16; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		// Test SetElem
		v = v.SetElem(5, int8(99))
		a[5] = 99
		var out [16]int8
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Int16x8 GetElem/SetElem
	{
		a := [8]int16{10, 20, 30, 40, 50, 60, 70, 80}
		v := archsimd.LoadInt16x8Array(&a)

		for i := uint8(0); i < 8; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(3, int16(123))
		a[3] = 123
		var out [8]int16
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Int32x4 GetElem/SetElem
	{
		a := [4]int32{100, 200, 300, 400}
		v := archsimd.LoadInt32x4Array(&a)

		for i := uint8(0); i < 4; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(2, int32(999))
		a[2] = 999
		var out [4]int32
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Int64x2 GetElem/SetElem
	{
		a := [2]int64{1000, 2000}
		v := archsimd.LoadInt64x2Array(&a)

		for i := uint8(0); i < 2; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(1, int64(5555))
		a[1] = 5555
		var out [2]int64
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Uint8x16 GetElem/SetElem
	{
		a := [16]uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadUint8x16Array(&a)

		for i := uint8(0); i < 16; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(7, uint8(200))
		a[7] = 200
		var out [16]uint8
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Uint16x8 GetElem/SetElem
	{
		a := [8]uint16{100, 200, 300, 400, 500, 600, 700, 800}
		v := archsimd.LoadUint16x8Array(&a)

		for i := uint8(0); i < 8; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(0, uint16(1111))
		a[0] = 1111
		var out [8]uint16
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Uint32x4 GetElem/SetElem
	{
		a := [4]uint32{1000, 2000, 3000, 4000}
		v := archsimd.LoadUint32x4Array(&a)

		for i := uint8(0); i < 4; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(3, uint32(9999))
		a[3] = 9999
		var out [4]uint32
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Uint64x2 GetElem/SetElem
	{
		a := [2]uint64{10000, 20000}
		v := archsimd.LoadUint64x2Array(&a)

		for i := uint8(0); i < 2; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(0, uint64(55555))
		a[0] = 55555
		var out [2]uint64
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Float32x4 GetElem/SetElem
	{
		a := [4]float32{1.0, 2.0, 3.0, 4.0}
		v := archsimd.LoadFloat32x4Array(&a)

		for i := uint8(0); i < 4; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(1, float32(9.5))
		a[1] = 9.5
		var out [4]float32
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	// Float64x2 GetElem/SetElem
	{
		a := [2]float64{10.5, 20.5}
		v := archsimd.LoadFloat64x2Array(&a)

		for i := uint8(0); i < 2; i++ {
			if got := v.GetElem(i); got != a[i] {
				fail = true
			}
		}

		v = v.SetElem(0, float64(99.25))
		a[0] = 99.25
		var out [2]float64
		v.StoreArray(&out)
		for i := range out {
			if out[i] != a[i] {
				fail = true
			}
		}
	}

	if fail {
		os.Exit(1)
	}
}
