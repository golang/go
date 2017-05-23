package main

import "fmt"

//go:noinline
func add_uint64_0_ssa(a uint64) uint64 {
	return a + 0
}

//go:noinline
func add_0_uint64_ssa(a uint64) uint64 {
	return 0 + a
}

//go:noinline
func add_uint64_1_ssa(a uint64) uint64 {
	return a + 1
}

//go:noinline
func add_1_uint64_ssa(a uint64) uint64 {
	return 1 + a
}

//go:noinline
func add_uint64_4294967296_ssa(a uint64) uint64 {
	return a + 4294967296
}

//go:noinline
func add_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 + a
}

//go:noinline
func add_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a + 18446744073709551615
}

//go:noinline
func add_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 + a
}

//go:noinline
func sub_uint64_0_ssa(a uint64) uint64 {
	return a - 0
}

//go:noinline
func sub_0_uint64_ssa(a uint64) uint64 {
	return 0 - a
}

//go:noinline
func sub_uint64_1_ssa(a uint64) uint64 {
	return a - 1
}

//go:noinline
func sub_1_uint64_ssa(a uint64) uint64 {
	return 1 - a
}

//go:noinline
func sub_uint64_4294967296_ssa(a uint64) uint64 {
	return a - 4294967296
}

//go:noinline
func sub_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 - a
}

//go:noinline
func sub_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a - 18446744073709551615
}

//go:noinline
func sub_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 - a
}

//go:noinline
func div_0_uint64_ssa(a uint64) uint64 {
	return 0 / a
}

//go:noinline
func div_uint64_1_ssa(a uint64) uint64 {
	return a / 1
}

//go:noinline
func div_1_uint64_ssa(a uint64) uint64 {
	return 1 / a
}

//go:noinline
func div_uint64_4294967296_ssa(a uint64) uint64 {
	return a / 4294967296
}

//go:noinline
func div_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 / a
}

//go:noinline
func div_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a / 18446744073709551615
}

//go:noinline
func div_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 / a
}

//go:noinline
func mul_uint64_0_ssa(a uint64) uint64 {
	return a * 0
}

//go:noinline
func mul_0_uint64_ssa(a uint64) uint64 {
	return 0 * a
}

//go:noinline
func mul_uint64_1_ssa(a uint64) uint64 {
	return a * 1
}

//go:noinline
func mul_1_uint64_ssa(a uint64) uint64 {
	return 1 * a
}

//go:noinline
func mul_uint64_4294967296_ssa(a uint64) uint64 {
	return a * 4294967296
}

//go:noinline
func mul_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 * a
}

//go:noinline
func mul_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a * 18446744073709551615
}

//go:noinline
func mul_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 * a
}

//go:noinline
func lsh_uint64_0_ssa(a uint64) uint64 {
	return a << 0
}

//go:noinline
func lsh_0_uint64_ssa(a uint64) uint64 {
	return 0 << a
}

//go:noinline
func lsh_uint64_1_ssa(a uint64) uint64 {
	return a << 1
}

//go:noinline
func lsh_1_uint64_ssa(a uint64) uint64 {
	return 1 << a
}

//go:noinline
func lsh_uint64_4294967296_ssa(a uint64) uint64 {
	return a << uint64(4294967296)
}

//go:noinline
func lsh_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 << a
}

//go:noinline
func lsh_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a << uint64(18446744073709551615)
}

//go:noinline
func lsh_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 << a
}

//go:noinline
func rsh_uint64_0_ssa(a uint64) uint64 {
	return a >> 0
}

//go:noinline
func rsh_0_uint64_ssa(a uint64) uint64 {
	return 0 >> a
}

//go:noinline
func rsh_uint64_1_ssa(a uint64) uint64 {
	return a >> 1
}

//go:noinline
func rsh_1_uint64_ssa(a uint64) uint64 {
	return 1 >> a
}

//go:noinline
func rsh_uint64_4294967296_ssa(a uint64) uint64 {
	return a >> uint64(4294967296)
}

//go:noinline
func rsh_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 >> a
}

//go:noinline
func rsh_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a >> uint64(18446744073709551615)
}

//go:noinline
func rsh_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 >> a
}

//go:noinline
func mod_0_uint64_ssa(a uint64) uint64 {
	return 0 % a
}

//go:noinline
func mod_uint64_1_ssa(a uint64) uint64 {
	return a % 1
}

//go:noinline
func mod_1_uint64_ssa(a uint64) uint64 {
	return 1 % a
}

//go:noinline
func mod_uint64_4294967296_ssa(a uint64) uint64 {
	return a % 4294967296
}

//go:noinline
func mod_4294967296_uint64_ssa(a uint64) uint64 {
	return 4294967296 % a
}

//go:noinline
func mod_uint64_18446744073709551615_ssa(a uint64) uint64 {
	return a % 18446744073709551615
}

//go:noinline
func mod_18446744073709551615_uint64_ssa(a uint64) uint64 {
	return 18446744073709551615 % a
}

//go:noinline
func add_int64_Neg9223372036854775808_ssa(a int64) int64 {
	return a + -9223372036854775808
}

//go:noinline
func add_Neg9223372036854775808_int64_ssa(a int64) int64 {
	return -9223372036854775808 + a
}

//go:noinline
func add_int64_Neg9223372036854775807_ssa(a int64) int64 {
	return a + -9223372036854775807
}

//go:noinline
func add_Neg9223372036854775807_int64_ssa(a int64) int64 {
	return -9223372036854775807 + a
}

//go:noinline
func add_int64_Neg4294967296_ssa(a int64) int64 {
	return a + -4294967296
}

//go:noinline
func add_Neg4294967296_int64_ssa(a int64) int64 {
	return -4294967296 + a
}

//go:noinline
func add_int64_Neg1_ssa(a int64) int64 {
	return a + -1
}

//go:noinline
func add_Neg1_int64_ssa(a int64) int64 {
	return -1 + a
}

//go:noinline
func add_int64_0_ssa(a int64) int64 {
	return a + 0
}

//go:noinline
func add_0_int64_ssa(a int64) int64 {
	return 0 + a
}

//go:noinline
func add_int64_1_ssa(a int64) int64 {
	return a + 1
}

//go:noinline
func add_1_int64_ssa(a int64) int64 {
	return 1 + a
}

//go:noinline
func add_int64_4294967296_ssa(a int64) int64 {
	return a + 4294967296
}

//go:noinline
func add_4294967296_int64_ssa(a int64) int64 {
	return 4294967296 + a
}

//go:noinline
func add_int64_9223372036854775806_ssa(a int64) int64 {
	return a + 9223372036854775806
}

//go:noinline
func add_9223372036854775806_int64_ssa(a int64) int64 {
	return 9223372036854775806 + a
}

//go:noinline
func add_int64_9223372036854775807_ssa(a int64) int64 {
	return a + 9223372036854775807
}

//go:noinline
func add_9223372036854775807_int64_ssa(a int64) int64 {
	return 9223372036854775807 + a
}

//go:noinline
func sub_int64_Neg9223372036854775808_ssa(a int64) int64 {
	return a - -9223372036854775808
}

//go:noinline
func sub_Neg9223372036854775808_int64_ssa(a int64) int64 {
	return -9223372036854775808 - a
}

//go:noinline
func sub_int64_Neg9223372036854775807_ssa(a int64) int64 {
	return a - -9223372036854775807
}

//go:noinline
func sub_Neg9223372036854775807_int64_ssa(a int64) int64 {
	return -9223372036854775807 - a
}

//go:noinline
func sub_int64_Neg4294967296_ssa(a int64) int64 {
	return a - -4294967296
}

//go:noinline
func sub_Neg4294967296_int64_ssa(a int64) int64 {
	return -4294967296 - a
}

//go:noinline
func sub_int64_Neg1_ssa(a int64) int64 {
	return a - -1
}

//go:noinline
func sub_Neg1_int64_ssa(a int64) int64 {
	return -1 - a
}

//go:noinline
func sub_int64_0_ssa(a int64) int64 {
	return a - 0
}

//go:noinline
func sub_0_int64_ssa(a int64) int64 {
	return 0 - a
}

//go:noinline
func sub_int64_1_ssa(a int64) int64 {
	return a - 1
}

//go:noinline
func sub_1_int64_ssa(a int64) int64 {
	return 1 - a
}

//go:noinline
func sub_int64_4294967296_ssa(a int64) int64 {
	return a - 4294967296
}

//go:noinline
func sub_4294967296_int64_ssa(a int64) int64 {
	return 4294967296 - a
}

//go:noinline
func sub_int64_9223372036854775806_ssa(a int64) int64 {
	return a - 9223372036854775806
}

//go:noinline
func sub_9223372036854775806_int64_ssa(a int64) int64 {
	return 9223372036854775806 - a
}

//go:noinline
func sub_int64_9223372036854775807_ssa(a int64) int64 {
	return a - 9223372036854775807
}

//go:noinline
func sub_9223372036854775807_int64_ssa(a int64) int64 {
	return 9223372036854775807 - a
}

//go:noinline
func div_int64_Neg9223372036854775808_ssa(a int64) int64 {
	return a / -9223372036854775808
}

//go:noinline
func div_Neg9223372036854775808_int64_ssa(a int64) int64 {
	return -9223372036854775808 / a
}

//go:noinline
func div_int64_Neg9223372036854775807_ssa(a int64) int64 {
	return a / -9223372036854775807
}

//go:noinline
func div_Neg9223372036854775807_int64_ssa(a int64) int64 {
	return -9223372036854775807 / a
}

//go:noinline
func div_int64_Neg4294967296_ssa(a int64) int64 {
	return a / -4294967296
}

//go:noinline
func div_Neg4294967296_int64_ssa(a int64) int64 {
	return -4294967296 / a
}

//go:noinline
func div_int64_Neg1_ssa(a int64) int64 {
	return a / -1
}

//go:noinline
func div_Neg1_int64_ssa(a int64) int64 {
	return -1 / a
}

//go:noinline
func div_0_int64_ssa(a int64) int64 {
	return 0 / a
}

//go:noinline
func div_int64_1_ssa(a int64) int64 {
	return a / 1
}

//go:noinline
func div_1_int64_ssa(a int64) int64 {
	return 1 / a
}

//go:noinline
func div_int64_4294967296_ssa(a int64) int64 {
	return a / 4294967296
}

//go:noinline
func div_4294967296_int64_ssa(a int64) int64 {
	return 4294967296 / a
}

//go:noinline
func div_int64_9223372036854775806_ssa(a int64) int64 {
	return a / 9223372036854775806
}

//go:noinline
func div_9223372036854775806_int64_ssa(a int64) int64 {
	return 9223372036854775806 / a
}

//go:noinline
func div_int64_9223372036854775807_ssa(a int64) int64 {
	return a / 9223372036854775807
}

//go:noinline
func div_9223372036854775807_int64_ssa(a int64) int64 {
	return 9223372036854775807 / a
}

//go:noinline
func mul_int64_Neg9223372036854775808_ssa(a int64) int64 {
	return a * -9223372036854775808
}

//go:noinline
func mul_Neg9223372036854775808_int64_ssa(a int64) int64 {
	return -9223372036854775808 * a
}

//go:noinline
func mul_int64_Neg9223372036854775807_ssa(a int64) int64 {
	return a * -9223372036854775807
}

//go:noinline
func mul_Neg9223372036854775807_int64_ssa(a int64) int64 {
	return -9223372036854775807 * a
}

//go:noinline
func mul_int64_Neg4294967296_ssa(a int64) int64 {
	return a * -4294967296
}

//go:noinline
func mul_Neg4294967296_int64_ssa(a int64) int64 {
	return -4294967296 * a
}

//go:noinline
func mul_int64_Neg1_ssa(a int64) int64 {
	return a * -1
}

//go:noinline
func mul_Neg1_int64_ssa(a int64) int64 {
	return -1 * a
}

//go:noinline
func mul_int64_0_ssa(a int64) int64 {
	return a * 0
}

//go:noinline
func mul_0_int64_ssa(a int64) int64 {
	return 0 * a
}

//go:noinline
func mul_int64_1_ssa(a int64) int64 {
	return a * 1
}

//go:noinline
func mul_1_int64_ssa(a int64) int64 {
	return 1 * a
}

//go:noinline
func mul_int64_4294967296_ssa(a int64) int64 {
	return a * 4294967296
}

//go:noinline
func mul_4294967296_int64_ssa(a int64) int64 {
	return 4294967296 * a
}

//go:noinline
func mul_int64_9223372036854775806_ssa(a int64) int64 {
	return a * 9223372036854775806
}

//go:noinline
func mul_9223372036854775806_int64_ssa(a int64) int64 {
	return 9223372036854775806 * a
}

//go:noinline
func mul_int64_9223372036854775807_ssa(a int64) int64 {
	return a * 9223372036854775807
}

//go:noinline
func mul_9223372036854775807_int64_ssa(a int64) int64 {
	return 9223372036854775807 * a
}

//go:noinline
func mod_int64_Neg9223372036854775808_ssa(a int64) int64 {
	return a % -9223372036854775808
}

//go:noinline
func mod_Neg9223372036854775808_int64_ssa(a int64) int64 {
	return -9223372036854775808 % a
}

//go:noinline
func mod_int64_Neg9223372036854775807_ssa(a int64) int64 {
	return a % -9223372036854775807
}

//go:noinline
func mod_Neg9223372036854775807_int64_ssa(a int64) int64 {
	return -9223372036854775807 % a
}

//go:noinline
func mod_int64_Neg4294967296_ssa(a int64) int64 {
	return a % -4294967296
}

//go:noinline
func mod_Neg4294967296_int64_ssa(a int64) int64 {
	return -4294967296 % a
}

//go:noinline
func mod_int64_Neg1_ssa(a int64) int64 {
	return a % -1
}

//go:noinline
func mod_Neg1_int64_ssa(a int64) int64 {
	return -1 % a
}

//go:noinline
func mod_0_int64_ssa(a int64) int64 {
	return 0 % a
}

//go:noinline
func mod_int64_1_ssa(a int64) int64 {
	return a % 1
}

//go:noinline
func mod_1_int64_ssa(a int64) int64 {
	return 1 % a
}

//go:noinline
func mod_int64_4294967296_ssa(a int64) int64 {
	return a % 4294967296
}

//go:noinline
func mod_4294967296_int64_ssa(a int64) int64 {
	return 4294967296 % a
}

//go:noinline
func mod_int64_9223372036854775806_ssa(a int64) int64 {
	return a % 9223372036854775806
}

//go:noinline
func mod_9223372036854775806_int64_ssa(a int64) int64 {
	return 9223372036854775806 % a
}

//go:noinline
func mod_int64_9223372036854775807_ssa(a int64) int64 {
	return a % 9223372036854775807
}

//go:noinline
func mod_9223372036854775807_int64_ssa(a int64) int64 {
	return 9223372036854775807 % a
}

//go:noinline
func add_uint32_0_ssa(a uint32) uint32 {
	return a + 0
}

//go:noinline
func add_0_uint32_ssa(a uint32) uint32 {
	return 0 + a
}

//go:noinline
func add_uint32_1_ssa(a uint32) uint32 {
	return a + 1
}

//go:noinline
func add_1_uint32_ssa(a uint32) uint32 {
	return 1 + a
}

//go:noinline
func add_uint32_4294967295_ssa(a uint32) uint32 {
	return a + 4294967295
}

//go:noinline
func add_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 + a
}

//go:noinline
func sub_uint32_0_ssa(a uint32) uint32 {
	return a - 0
}

//go:noinline
func sub_0_uint32_ssa(a uint32) uint32 {
	return 0 - a
}

//go:noinline
func sub_uint32_1_ssa(a uint32) uint32 {
	return a - 1
}

//go:noinline
func sub_1_uint32_ssa(a uint32) uint32 {
	return 1 - a
}

//go:noinline
func sub_uint32_4294967295_ssa(a uint32) uint32 {
	return a - 4294967295
}

//go:noinline
func sub_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 - a
}

//go:noinline
func div_0_uint32_ssa(a uint32) uint32 {
	return 0 / a
}

//go:noinline
func div_uint32_1_ssa(a uint32) uint32 {
	return a / 1
}

//go:noinline
func div_1_uint32_ssa(a uint32) uint32 {
	return 1 / a
}

//go:noinline
func div_uint32_4294967295_ssa(a uint32) uint32 {
	return a / 4294967295
}

//go:noinline
func div_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 / a
}

//go:noinline
func mul_uint32_0_ssa(a uint32) uint32 {
	return a * 0
}

//go:noinline
func mul_0_uint32_ssa(a uint32) uint32 {
	return 0 * a
}

//go:noinline
func mul_uint32_1_ssa(a uint32) uint32 {
	return a * 1
}

//go:noinline
func mul_1_uint32_ssa(a uint32) uint32 {
	return 1 * a
}

//go:noinline
func mul_uint32_4294967295_ssa(a uint32) uint32 {
	return a * 4294967295
}

//go:noinline
func mul_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 * a
}

//go:noinline
func lsh_uint32_0_ssa(a uint32) uint32 {
	return a << 0
}

//go:noinline
func lsh_0_uint32_ssa(a uint32) uint32 {
	return 0 << a
}

//go:noinline
func lsh_uint32_1_ssa(a uint32) uint32 {
	return a << 1
}

//go:noinline
func lsh_1_uint32_ssa(a uint32) uint32 {
	return 1 << a
}

//go:noinline
func lsh_uint32_4294967295_ssa(a uint32) uint32 {
	return a << 4294967295
}

//go:noinline
func lsh_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 << a
}

//go:noinline
func rsh_uint32_0_ssa(a uint32) uint32 {
	return a >> 0
}

//go:noinline
func rsh_0_uint32_ssa(a uint32) uint32 {
	return 0 >> a
}

//go:noinline
func rsh_uint32_1_ssa(a uint32) uint32 {
	return a >> 1
}

//go:noinline
func rsh_1_uint32_ssa(a uint32) uint32 {
	return 1 >> a
}

//go:noinline
func rsh_uint32_4294967295_ssa(a uint32) uint32 {
	return a >> 4294967295
}

//go:noinline
func rsh_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 >> a
}

//go:noinline
func mod_0_uint32_ssa(a uint32) uint32 {
	return 0 % a
}

//go:noinline
func mod_uint32_1_ssa(a uint32) uint32 {
	return a % 1
}

//go:noinline
func mod_1_uint32_ssa(a uint32) uint32 {
	return 1 % a
}

//go:noinline
func mod_uint32_4294967295_ssa(a uint32) uint32 {
	return a % 4294967295
}

//go:noinline
func mod_4294967295_uint32_ssa(a uint32) uint32 {
	return 4294967295 % a
}

//go:noinline
func add_int32_Neg2147483648_ssa(a int32) int32 {
	return a + -2147483648
}

//go:noinline
func add_Neg2147483648_int32_ssa(a int32) int32 {
	return -2147483648 + a
}

//go:noinline
func add_int32_Neg2147483647_ssa(a int32) int32 {
	return a + -2147483647
}

//go:noinline
func add_Neg2147483647_int32_ssa(a int32) int32 {
	return -2147483647 + a
}

//go:noinline
func add_int32_Neg1_ssa(a int32) int32 {
	return a + -1
}

//go:noinline
func add_Neg1_int32_ssa(a int32) int32 {
	return -1 + a
}

//go:noinline
func add_int32_0_ssa(a int32) int32 {
	return a + 0
}

//go:noinline
func add_0_int32_ssa(a int32) int32 {
	return 0 + a
}

//go:noinline
func add_int32_1_ssa(a int32) int32 {
	return a + 1
}

//go:noinline
func add_1_int32_ssa(a int32) int32 {
	return 1 + a
}

//go:noinline
func add_int32_2147483647_ssa(a int32) int32 {
	return a + 2147483647
}

//go:noinline
func add_2147483647_int32_ssa(a int32) int32 {
	return 2147483647 + a
}

//go:noinline
func sub_int32_Neg2147483648_ssa(a int32) int32 {
	return a - -2147483648
}

//go:noinline
func sub_Neg2147483648_int32_ssa(a int32) int32 {
	return -2147483648 - a
}

//go:noinline
func sub_int32_Neg2147483647_ssa(a int32) int32 {
	return a - -2147483647
}

//go:noinline
func sub_Neg2147483647_int32_ssa(a int32) int32 {
	return -2147483647 - a
}

//go:noinline
func sub_int32_Neg1_ssa(a int32) int32 {
	return a - -1
}

//go:noinline
func sub_Neg1_int32_ssa(a int32) int32 {
	return -1 - a
}

//go:noinline
func sub_int32_0_ssa(a int32) int32 {
	return a - 0
}

//go:noinline
func sub_0_int32_ssa(a int32) int32 {
	return 0 - a
}

//go:noinline
func sub_int32_1_ssa(a int32) int32 {
	return a - 1
}

//go:noinline
func sub_1_int32_ssa(a int32) int32 {
	return 1 - a
}

//go:noinline
func sub_int32_2147483647_ssa(a int32) int32 {
	return a - 2147483647
}

//go:noinline
func sub_2147483647_int32_ssa(a int32) int32 {
	return 2147483647 - a
}

//go:noinline
func div_int32_Neg2147483648_ssa(a int32) int32 {
	return a / -2147483648
}

//go:noinline
func div_Neg2147483648_int32_ssa(a int32) int32 {
	return -2147483648 / a
}

//go:noinline
func div_int32_Neg2147483647_ssa(a int32) int32 {
	return a / -2147483647
}

//go:noinline
func div_Neg2147483647_int32_ssa(a int32) int32 {
	return -2147483647 / a
}

//go:noinline
func div_int32_Neg1_ssa(a int32) int32 {
	return a / -1
}

//go:noinline
func div_Neg1_int32_ssa(a int32) int32 {
	return -1 / a
}

//go:noinline
func div_0_int32_ssa(a int32) int32 {
	return 0 / a
}

//go:noinline
func div_int32_1_ssa(a int32) int32 {
	return a / 1
}

//go:noinline
func div_1_int32_ssa(a int32) int32 {
	return 1 / a
}

//go:noinline
func div_int32_2147483647_ssa(a int32) int32 {
	return a / 2147483647
}

//go:noinline
func div_2147483647_int32_ssa(a int32) int32 {
	return 2147483647 / a
}

//go:noinline
func mul_int32_Neg2147483648_ssa(a int32) int32 {
	return a * -2147483648
}

//go:noinline
func mul_Neg2147483648_int32_ssa(a int32) int32 {
	return -2147483648 * a
}

//go:noinline
func mul_int32_Neg2147483647_ssa(a int32) int32 {
	return a * -2147483647
}

//go:noinline
func mul_Neg2147483647_int32_ssa(a int32) int32 {
	return -2147483647 * a
}

//go:noinline
func mul_int32_Neg1_ssa(a int32) int32 {
	return a * -1
}

//go:noinline
func mul_Neg1_int32_ssa(a int32) int32 {
	return -1 * a
}

//go:noinline
func mul_int32_0_ssa(a int32) int32 {
	return a * 0
}

//go:noinline
func mul_0_int32_ssa(a int32) int32 {
	return 0 * a
}

//go:noinline
func mul_int32_1_ssa(a int32) int32 {
	return a * 1
}

//go:noinline
func mul_1_int32_ssa(a int32) int32 {
	return 1 * a
}

//go:noinline
func mul_int32_2147483647_ssa(a int32) int32 {
	return a * 2147483647
}

//go:noinline
func mul_2147483647_int32_ssa(a int32) int32 {
	return 2147483647 * a
}

//go:noinline
func mod_int32_Neg2147483648_ssa(a int32) int32 {
	return a % -2147483648
}

//go:noinline
func mod_Neg2147483648_int32_ssa(a int32) int32 {
	return -2147483648 % a
}

//go:noinline
func mod_int32_Neg2147483647_ssa(a int32) int32 {
	return a % -2147483647
}

//go:noinline
func mod_Neg2147483647_int32_ssa(a int32) int32 {
	return -2147483647 % a
}

//go:noinline
func mod_int32_Neg1_ssa(a int32) int32 {
	return a % -1
}

//go:noinline
func mod_Neg1_int32_ssa(a int32) int32 {
	return -1 % a
}

//go:noinline
func mod_0_int32_ssa(a int32) int32 {
	return 0 % a
}

//go:noinline
func mod_int32_1_ssa(a int32) int32 {
	return a % 1
}

//go:noinline
func mod_1_int32_ssa(a int32) int32 {
	return 1 % a
}

//go:noinline
func mod_int32_2147483647_ssa(a int32) int32 {
	return a % 2147483647
}

//go:noinline
func mod_2147483647_int32_ssa(a int32) int32 {
	return 2147483647 % a
}

//go:noinline
func add_uint16_0_ssa(a uint16) uint16 {
	return a + 0
}

//go:noinline
func add_0_uint16_ssa(a uint16) uint16 {
	return 0 + a
}

//go:noinline
func add_uint16_1_ssa(a uint16) uint16 {
	return a + 1
}

//go:noinline
func add_1_uint16_ssa(a uint16) uint16 {
	return 1 + a
}

//go:noinline
func add_uint16_65535_ssa(a uint16) uint16 {
	return a + 65535
}

//go:noinline
func add_65535_uint16_ssa(a uint16) uint16 {
	return 65535 + a
}

//go:noinline
func sub_uint16_0_ssa(a uint16) uint16 {
	return a - 0
}

//go:noinline
func sub_0_uint16_ssa(a uint16) uint16 {
	return 0 - a
}

//go:noinline
func sub_uint16_1_ssa(a uint16) uint16 {
	return a - 1
}

//go:noinline
func sub_1_uint16_ssa(a uint16) uint16 {
	return 1 - a
}

//go:noinline
func sub_uint16_65535_ssa(a uint16) uint16 {
	return a - 65535
}

//go:noinline
func sub_65535_uint16_ssa(a uint16) uint16 {
	return 65535 - a
}

//go:noinline
func div_0_uint16_ssa(a uint16) uint16 {
	return 0 / a
}

//go:noinline
func div_uint16_1_ssa(a uint16) uint16 {
	return a / 1
}

//go:noinline
func div_1_uint16_ssa(a uint16) uint16 {
	return 1 / a
}

//go:noinline
func div_uint16_65535_ssa(a uint16) uint16 {
	return a / 65535
}

//go:noinline
func div_65535_uint16_ssa(a uint16) uint16 {
	return 65535 / a
}

//go:noinline
func mul_uint16_0_ssa(a uint16) uint16 {
	return a * 0
}

//go:noinline
func mul_0_uint16_ssa(a uint16) uint16 {
	return 0 * a
}

//go:noinline
func mul_uint16_1_ssa(a uint16) uint16 {
	return a * 1
}

//go:noinline
func mul_1_uint16_ssa(a uint16) uint16 {
	return 1 * a
}

//go:noinline
func mul_uint16_65535_ssa(a uint16) uint16 {
	return a * 65535
}

//go:noinline
func mul_65535_uint16_ssa(a uint16) uint16 {
	return 65535 * a
}

//go:noinline
func lsh_uint16_0_ssa(a uint16) uint16 {
	return a << 0
}

//go:noinline
func lsh_0_uint16_ssa(a uint16) uint16 {
	return 0 << a
}

//go:noinline
func lsh_uint16_1_ssa(a uint16) uint16 {
	return a << 1
}

//go:noinline
func lsh_1_uint16_ssa(a uint16) uint16 {
	return 1 << a
}

//go:noinline
func lsh_uint16_65535_ssa(a uint16) uint16 {
	return a << 65535
}

//go:noinline
func lsh_65535_uint16_ssa(a uint16) uint16 {
	return 65535 << a
}

//go:noinline
func rsh_uint16_0_ssa(a uint16) uint16 {
	return a >> 0
}

//go:noinline
func rsh_0_uint16_ssa(a uint16) uint16 {
	return 0 >> a
}

//go:noinline
func rsh_uint16_1_ssa(a uint16) uint16 {
	return a >> 1
}

//go:noinline
func rsh_1_uint16_ssa(a uint16) uint16 {
	return 1 >> a
}

//go:noinline
func rsh_uint16_65535_ssa(a uint16) uint16 {
	return a >> 65535
}

//go:noinline
func rsh_65535_uint16_ssa(a uint16) uint16 {
	return 65535 >> a
}

//go:noinline
func mod_0_uint16_ssa(a uint16) uint16 {
	return 0 % a
}

//go:noinline
func mod_uint16_1_ssa(a uint16) uint16 {
	return a % 1
}

//go:noinline
func mod_1_uint16_ssa(a uint16) uint16 {
	return 1 % a
}

//go:noinline
func mod_uint16_65535_ssa(a uint16) uint16 {
	return a % 65535
}

//go:noinline
func mod_65535_uint16_ssa(a uint16) uint16 {
	return 65535 % a
}

//go:noinline
func add_int16_Neg32768_ssa(a int16) int16 {
	return a + -32768
}

//go:noinline
func add_Neg32768_int16_ssa(a int16) int16 {
	return -32768 + a
}

//go:noinline
func add_int16_Neg32767_ssa(a int16) int16 {
	return a + -32767
}

//go:noinline
func add_Neg32767_int16_ssa(a int16) int16 {
	return -32767 + a
}

//go:noinline
func add_int16_Neg1_ssa(a int16) int16 {
	return a + -1
}

//go:noinline
func add_Neg1_int16_ssa(a int16) int16 {
	return -1 + a
}

//go:noinline
func add_int16_0_ssa(a int16) int16 {
	return a + 0
}

//go:noinline
func add_0_int16_ssa(a int16) int16 {
	return 0 + a
}

//go:noinline
func add_int16_1_ssa(a int16) int16 {
	return a + 1
}

//go:noinline
func add_1_int16_ssa(a int16) int16 {
	return 1 + a
}

//go:noinline
func add_int16_32766_ssa(a int16) int16 {
	return a + 32766
}

//go:noinline
func add_32766_int16_ssa(a int16) int16 {
	return 32766 + a
}

//go:noinline
func add_int16_32767_ssa(a int16) int16 {
	return a + 32767
}

//go:noinline
func add_32767_int16_ssa(a int16) int16 {
	return 32767 + a
}

//go:noinline
func sub_int16_Neg32768_ssa(a int16) int16 {
	return a - -32768
}

//go:noinline
func sub_Neg32768_int16_ssa(a int16) int16 {
	return -32768 - a
}

//go:noinline
func sub_int16_Neg32767_ssa(a int16) int16 {
	return a - -32767
}

//go:noinline
func sub_Neg32767_int16_ssa(a int16) int16 {
	return -32767 - a
}

//go:noinline
func sub_int16_Neg1_ssa(a int16) int16 {
	return a - -1
}

//go:noinline
func sub_Neg1_int16_ssa(a int16) int16 {
	return -1 - a
}

//go:noinline
func sub_int16_0_ssa(a int16) int16 {
	return a - 0
}

//go:noinline
func sub_0_int16_ssa(a int16) int16 {
	return 0 - a
}

//go:noinline
func sub_int16_1_ssa(a int16) int16 {
	return a - 1
}

//go:noinline
func sub_1_int16_ssa(a int16) int16 {
	return 1 - a
}

//go:noinline
func sub_int16_32766_ssa(a int16) int16 {
	return a - 32766
}

//go:noinline
func sub_32766_int16_ssa(a int16) int16 {
	return 32766 - a
}

//go:noinline
func sub_int16_32767_ssa(a int16) int16 {
	return a - 32767
}

//go:noinline
func sub_32767_int16_ssa(a int16) int16 {
	return 32767 - a
}

//go:noinline
func div_int16_Neg32768_ssa(a int16) int16 {
	return a / -32768
}

//go:noinline
func div_Neg32768_int16_ssa(a int16) int16 {
	return -32768 / a
}

//go:noinline
func div_int16_Neg32767_ssa(a int16) int16 {
	return a / -32767
}

//go:noinline
func div_Neg32767_int16_ssa(a int16) int16 {
	return -32767 / a
}

//go:noinline
func div_int16_Neg1_ssa(a int16) int16 {
	return a / -1
}

//go:noinline
func div_Neg1_int16_ssa(a int16) int16 {
	return -1 / a
}

//go:noinline
func div_0_int16_ssa(a int16) int16 {
	return 0 / a
}

//go:noinline
func div_int16_1_ssa(a int16) int16 {
	return a / 1
}

//go:noinline
func div_1_int16_ssa(a int16) int16 {
	return 1 / a
}

//go:noinline
func div_int16_32766_ssa(a int16) int16 {
	return a / 32766
}

//go:noinline
func div_32766_int16_ssa(a int16) int16 {
	return 32766 / a
}

//go:noinline
func div_int16_32767_ssa(a int16) int16 {
	return a / 32767
}

//go:noinline
func div_32767_int16_ssa(a int16) int16 {
	return 32767 / a
}

//go:noinline
func mul_int16_Neg32768_ssa(a int16) int16 {
	return a * -32768
}

//go:noinline
func mul_Neg32768_int16_ssa(a int16) int16 {
	return -32768 * a
}

//go:noinline
func mul_int16_Neg32767_ssa(a int16) int16 {
	return a * -32767
}

//go:noinline
func mul_Neg32767_int16_ssa(a int16) int16 {
	return -32767 * a
}

//go:noinline
func mul_int16_Neg1_ssa(a int16) int16 {
	return a * -1
}

//go:noinline
func mul_Neg1_int16_ssa(a int16) int16 {
	return -1 * a
}

//go:noinline
func mul_int16_0_ssa(a int16) int16 {
	return a * 0
}

//go:noinline
func mul_0_int16_ssa(a int16) int16 {
	return 0 * a
}

//go:noinline
func mul_int16_1_ssa(a int16) int16 {
	return a * 1
}

//go:noinline
func mul_1_int16_ssa(a int16) int16 {
	return 1 * a
}

//go:noinline
func mul_int16_32766_ssa(a int16) int16 {
	return a * 32766
}

//go:noinline
func mul_32766_int16_ssa(a int16) int16 {
	return 32766 * a
}

//go:noinline
func mul_int16_32767_ssa(a int16) int16 {
	return a * 32767
}

//go:noinline
func mul_32767_int16_ssa(a int16) int16 {
	return 32767 * a
}

//go:noinline
func mod_int16_Neg32768_ssa(a int16) int16 {
	return a % -32768
}

//go:noinline
func mod_Neg32768_int16_ssa(a int16) int16 {
	return -32768 % a
}

//go:noinline
func mod_int16_Neg32767_ssa(a int16) int16 {
	return a % -32767
}

//go:noinline
func mod_Neg32767_int16_ssa(a int16) int16 {
	return -32767 % a
}

//go:noinline
func mod_int16_Neg1_ssa(a int16) int16 {
	return a % -1
}

//go:noinline
func mod_Neg1_int16_ssa(a int16) int16 {
	return -1 % a
}

//go:noinline
func mod_0_int16_ssa(a int16) int16 {
	return 0 % a
}

//go:noinline
func mod_int16_1_ssa(a int16) int16 {
	return a % 1
}

//go:noinline
func mod_1_int16_ssa(a int16) int16 {
	return 1 % a
}

//go:noinline
func mod_int16_32766_ssa(a int16) int16 {
	return a % 32766
}

//go:noinline
func mod_32766_int16_ssa(a int16) int16 {
	return 32766 % a
}

//go:noinline
func mod_int16_32767_ssa(a int16) int16 {
	return a % 32767
}

//go:noinline
func mod_32767_int16_ssa(a int16) int16 {
	return 32767 % a
}

//go:noinline
func add_uint8_0_ssa(a uint8) uint8 {
	return a + 0
}

//go:noinline
func add_0_uint8_ssa(a uint8) uint8 {
	return 0 + a
}

//go:noinline
func add_uint8_1_ssa(a uint8) uint8 {
	return a + 1
}

//go:noinline
func add_1_uint8_ssa(a uint8) uint8 {
	return 1 + a
}

//go:noinline
func add_uint8_255_ssa(a uint8) uint8 {
	return a + 255
}

//go:noinline
func add_255_uint8_ssa(a uint8) uint8 {
	return 255 + a
}

//go:noinline
func sub_uint8_0_ssa(a uint8) uint8 {
	return a - 0
}

//go:noinline
func sub_0_uint8_ssa(a uint8) uint8 {
	return 0 - a
}

//go:noinline
func sub_uint8_1_ssa(a uint8) uint8 {
	return a - 1
}

//go:noinline
func sub_1_uint8_ssa(a uint8) uint8 {
	return 1 - a
}

//go:noinline
func sub_uint8_255_ssa(a uint8) uint8 {
	return a - 255
}

//go:noinline
func sub_255_uint8_ssa(a uint8) uint8 {
	return 255 - a
}

//go:noinline
func div_0_uint8_ssa(a uint8) uint8 {
	return 0 / a
}

//go:noinline
func div_uint8_1_ssa(a uint8) uint8 {
	return a / 1
}

//go:noinline
func div_1_uint8_ssa(a uint8) uint8 {
	return 1 / a
}

//go:noinline
func div_uint8_255_ssa(a uint8) uint8 {
	return a / 255
}

//go:noinline
func div_255_uint8_ssa(a uint8) uint8 {
	return 255 / a
}

//go:noinline
func mul_uint8_0_ssa(a uint8) uint8 {
	return a * 0
}

//go:noinline
func mul_0_uint8_ssa(a uint8) uint8 {
	return 0 * a
}

//go:noinline
func mul_uint8_1_ssa(a uint8) uint8 {
	return a * 1
}

//go:noinline
func mul_1_uint8_ssa(a uint8) uint8 {
	return 1 * a
}

//go:noinline
func mul_uint8_255_ssa(a uint8) uint8 {
	return a * 255
}

//go:noinline
func mul_255_uint8_ssa(a uint8) uint8 {
	return 255 * a
}

//go:noinline
func lsh_uint8_0_ssa(a uint8) uint8 {
	return a << 0
}

//go:noinline
func lsh_0_uint8_ssa(a uint8) uint8 {
	return 0 << a
}

//go:noinline
func lsh_uint8_1_ssa(a uint8) uint8 {
	return a << 1
}

//go:noinline
func lsh_1_uint8_ssa(a uint8) uint8 {
	return 1 << a
}

//go:noinline
func lsh_uint8_255_ssa(a uint8) uint8 {
	return a << 255
}

//go:noinline
func lsh_255_uint8_ssa(a uint8) uint8 {
	return 255 << a
}

//go:noinline
func rsh_uint8_0_ssa(a uint8) uint8 {
	return a >> 0
}

//go:noinline
func rsh_0_uint8_ssa(a uint8) uint8 {
	return 0 >> a
}

//go:noinline
func rsh_uint8_1_ssa(a uint8) uint8 {
	return a >> 1
}

//go:noinline
func rsh_1_uint8_ssa(a uint8) uint8 {
	return 1 >> a
}

//go:noinline
func rsh_uint8_255_ssa(a uint8) uint8 {
	return a >> 255
}

//go:noinline
func rsh_255_uint8_ssa(a uint8) uint8 {
	return 255 >> a
}

//go:noinline
func mod_0_uint8_ssa(a uint8) uint8 {
	return 0 % a
}

//go:noinline
func mod_uint8_1_ssa(a uint8) uint8 {
	return a % 1
}

//go:noinline
func mod_1_uint8_ssa(a uint8) uint8 {
	return 1 % a
}

//go:noinline
func mod_uint8_255_ssa(a uint8) uint8 {
	return a % 255
}

//go:noinline
func mod_255_uint8_ssa(a uint8) uint8 {
	return 255 % a
}

//go:noinline
func add_int8_Neg128_ssa(a int8) int8 {
	return a + -128
}

//go:noinline
func add_Neg128_int8_ssa(a int8) int8 {
	return -128 + a
}

//go:noinline
func add_int8_Neg127_ssa(a int8) int8 {
	return a + -127
}

//go:noinline
func add_Neg127_int8_ssa(a int8) int8 {
	return -127 + a
}

//go:noinline
func add_int8_Neg1_ssa(a int8) int8 {
	return a + -1
}

//go:noinline
func add_Neg1_int8_ssa(a int8) int8 {
	return -1 + a
}

//go:noinline
func add_int8_0_ssa(a int8) int8 {
	return a + 0
}

//go:noinline
func add_0_int8_ssa(a int8) int8 {
	return 0 + a
}

//go:noinline
func add_int8_1_ssa(a int8) int8 {
	return a + 1
}

//go:noinline
func add_1_int8_ssa(a int8) int8 {
	return 1 + a
}

//go:noinline
func add_int8_126_ssa(a int8) int8 {
	return a + 126
}

//go:noinline
func add_126_int8_ssa(a int8) int8 {
	return 126 + a
}

//go:noinline
func add_int8_127_ssa(a int8) int8 {
	return a + 127
}

//go:noinline
func add_127_int8_ssa(a int8) int8 {
	return 127 + a
}

//go:noinline
func sub_int8_Neg128_ssa(a int8) int8 {
	return a - -128
}

//go:noinline
func sub_Neg128_int8_ssa(a int8) int8 {
	return -128 - a
}

//go:noinline
func sub_int8_Neg127_ssa(a int8) int8 {
	return a - -127
}

//go:noinline
func sub_Neg127_int8_ssa(a int8) int8 {
	return -127 - a
}

//go:noinline
func sub_int8_Neg1_ssa(a int8) int8 {
	return a - -1
}

//go:noinline
func sub_Neg1_int8_ssa(a int8) int8 {
	return -1 - a
}

//go:noinline
func sub_int8_0_ssa(a int8) int8 {
	return a - 0
}

//go:noinline
func sub_0_int8_ssa(a int8) int8 {
	return 0 - a
}

//go:noinline
func sub_int8_1_ssa(a int8) int8 {
	return a - 1
}

//go:noinline
func sub_1_int8_ssa(a int8) int8 {
	return 1 - a
}

//go:noinline
func sub_int8_126_ssa(a int8) int8 {
	return a - 126
}

//go:noinline
func sub_126_int8_ssa(a int8) int8 {
	return 126 - a
}

//go:noinline
func sub_int8_127_ssa(a int8) int8 {
	return a - 127
}

//go:noinline
func sub_127_int8_ssa(a int8) int8 {
	return 127 - a
}

//go:noinline
func div_int8_Neg128_ssa(a int8) int8 {
	return a / -128
}

//go:noinline
func div_Neg128_int8_ssa(a int8) int8 {
	return -128 / a
}

//go:noinline
func div_int8_Neg127_ssa(a int8) int8 {
	return a / -127
}

//go:noinline
func div_Neg127_int8_ssa(a int8) int8 {
	return -127 / a
}

//go:noinline
func div_int8_Neg1_ssa(a int8) int8 {
	return a / -1
}

//go:noinline
func div_Neg1_int8_ssa(a int8) int8 {
	return -1 / a
}

//go:noinline
func div_0_int8_ssa(a int8) int8 {
	return 0 / a
}

//go:noinline
func div_int8_1_ssa(a int8) int8 {
	return a / 1
}

//go:noinline
func div_1_int8_ssa(a int8) int8 {
	return 1 / a
}

//go:noinline
func div_int8_126_ssa(a int8) int8 {
	return a / 126
}

//go:noinline
func div_126_int8_ssa(a int8) int8 {
	return 126 / a
}

//go:noinline
func div_int8_127_ssa(a int8) int8 {
	return a / 127
}

//go:noinline
func div_127_int8_ssa(a int8) int8 {
	return 127 / a
}

//go:noinline
func mul_int8_Neg128_ssa(a int8) int8 {
	return a * -128
}

//go:noinline
func mul_Neg128_int8_ssa(a int8) int8 {
	return -128 * a
}

//go:noinline
func mul_int8_Neg127_ssa(a int8) int8 {
	return a * -127
}

//go:noinline
func mul_Neg127_int8_ssa(a int8) int8 {
	return -127 * a
}

//go:noinline
func mul_int8_Neg1_ssa(a int8) int8 {
	return a * -1
}

//go:noinline
func mul_Neg1_int8_ssa(a int8) int8 {
	return -1 * a
}

//go:noinline
func mul_int8_0_ssa(a int8) int8 {
	return a * 0
}

//go:noinline
func mul_0_int8_ssa(a int8) int8 {
	return 0 * a
}

//go:noinline
func mul_int8_1_ssa(a int8) int8 {
	return a * 1
}

//go:noinline
func mul_1_int8_ssa(a int8) int8 {
	return 1 * a
}

//go:noinline
func mul_int8_126_ssa(a int8) int8 {
	return a * 126
}

//go:noinline
func mul_126_int8_ssa(a int8) int8 {
	return 126 * a
}

//go:noinline
func mul_int8_127_ssa(a int8) int8 {
	return a * 127
}

//go:noinline
func mul_127_int8_ssa(a int8) int8 {
	return 127 * a
}

//go:noinline
func mod_int8_Neg128_ssa(a int8) int8 {
	return a % -128
}

//go:noinline
func mod_Neg128_int8_ssa(a int8) int8 {
	return -128 % a
}

//go:noinline
func mod_int8_Neg127_ssa(a int8) int8 {
	return a % -127
}

//go:noinline
func mod_Neg127_int8_ssa(a int8) int8 {
	return -127 % a
}

//go:noinline
func mod_int8_Neg1_ssa(a int8) int8 {
	return a % -1
}

//go:noinline
func mod_Neg1_int8_ssa(a int8) int8 {
	return -1 % a
}

//go:noinline
func mod_0_int8_ssa(a int8) int8 {
	return 0 % a
}

//go:noinline
func mod_int8_1_ssa(a int8) int8 {
	return a % 1
}

//go:noinline
func mod_1_int8_ssa(a int8) int8 {
	return 1 % a
}

//go:noinline
func mod_int8_126_ssa(a int8) int8 {
	return a % 126
}

//go:noinline
func mod_126_int8_ssa(a int8) int8 {
	return 126 % a
}

//go:noinline
func mod_int8_127_ssa(a int8) int8 {
	return a % 127
}

//go:noinline
func mod_127_int8_ssa(a int8) int8 {
	return 127 % a
}

var failed bool

func main() {

	if got := add_0_uint64_ssa(0); got != 0 {
		fmt.Printf("add_uint64 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint64_0_ssa(0); got != 0 {
		fmt.Printf("add_uint64 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_uint64_ssa(1); got != 1 {
		fmt.Printf("add_uint64 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint64_0_ssa(1); got != 1 {
		fmt.Printf("add_uint64 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_uint64_ssa(4294967296); got != 4294967296 {
		fmt.Printf("add_uint64 0%s4294967296 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_uint64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("add_uint64 4294967296%s0 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_0_uint64_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("add_uint64 0%s18446744073709551615 = %d, wanted 18446744073709551615\n", `+`, got)
		failed = true
	}

	if got := add_uint64_0_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("add_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `+`, got)
		failed = true
	}

	if got := add_1_uint64_ssa(0); got != 1 {
		fmt.Printf("add_uint64 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint64_1_ssa(0); got != 1 {
		fmt.Printf("add_uint64 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_uint64_ssa(1); got != 2 {
		fmt.Printf("add_uint64 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_uint64_1_ssa(1); got != 2 {
		fmt.Printf("add_uint64 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_uint64_ssa(4294967296); got != 4294967297 {
		fmt.Printf("add_uint64 1%s4294967296 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_uint64_1_ssa(4294967296); got != 4294967297 {
		fmt.Printf("add_uint64 4294967296%s1 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_1_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("add_uint64 1%s18446744073709551615 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint64_1_ssa(18446744073709551615); got != 0 {
		fmt.Printf("add_uint64 18446744073709551615%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_uint64_ssa(0); got != 4294967296 {
		fmt.Printf("add_uint64 4294967296%s0 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_uint64_4294967296_ssa(0); got != 4294967296 {
		fmt.Printf("add_uint64 0%s4294967296 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_uint64_ssa(1); got != 4294967297 {
		fmt.Printf("add_uint64 4294967296%s1 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_uint64_4294967296_ssa(1); got != 4294967297 {
		fmt.Printf("add_uint64 1%s4294967296 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_uint64_ssa(4294967296); got != 8589934592 {
		fmt.Printf("add_uint64 4294967296%s4294967296 = %d, wanted 8589934592\n", `+`, got)
		failed = true
	}

	if got := add_uint64_4294967296_ssa(4294967296); got != 8589934592 {
		fmt.Printf("add_uint64 4294967296%s4294967296 = %d, wanted 8589934592\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_uint64_ssa(18446744073709551615); got != 4294967295 {
		fmt.Printf("add_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_uint64_4294967296_ssa(18446744073709551615); got != 4294967295 {
		fmt.Printf("add_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_18446744073709551615_uint64_ssa(0); got != 18446744073709551615 {
		fmt.Printf("add_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `+`, got)
		failed = true
	}

	if got := add_uint64_18446744073709551615_ssa(0); got != 18446744073709551615 {
		fmt.Printf("add_uint64 0%s18446744073709551615 = %d, wanted 18446744073709551615\n", `+`, got)
		failed = true
	}

	if got := add_18446744073709551615_uint64_ssa(1); got != 0 {
		fmt.Printf("add_uint64 18446744073709551615%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint64_18446744073709551615_ssa(1); got != 0 {
		fmt.Printf("add_uint64 1%s18446744073709551615 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_18446744073709551615_uint64_ssa(4294967296); got != 4294967295 {
		fmt.Printf("add_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_uint64_18446744073709551615_ssa(4294967296); got != 4294967295 {
		fmt.Printf("add_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_18446744073709551615_uint64_ssa(18446744073709551615); got != 18446744073709551614 {
		fmt.Printf("add_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 18446744073709551614\n", `+`, got)
		failed = true
	}

	if got := add_uint64_18446744073709551615_ssa(18446744073709551615); got != 18446744073709551614 {
		fmt.Printf("add_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 18446744073709551614\n", `+`, got)
		failed = true
	}

	if got := sub_0_uint64_ssa(0); got != 0 {
		fmt.Printf("sub_uint64 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_0_ssa(0); got != 0 {
		fmt.Printf("sub_uint64 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint64_ssa(1); got != 18446744073709551615 {
		fmt.Printf("sub_uint64 0%s1 = %d, wanted 18446744073709551615\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_0_ssa(1); got != 1 {
		fmt.Printf("sub_uint64 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint64_ssa(4294967296); got != 18446744069414584320 {
		fmt.Printf("sub_uint64 0%s4294967296 = %d, wanted 18446744069414584320\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("sub_uint64 4294967296%s0 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint64_ssa(18446744073709551615); got != 1 {
		fmt.Printf("sub_uint64 0%s18446744073709551615 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_0_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("sub_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint64_ssa(0); got != 1 {
		fmt.Printf("sub_uint64 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_1_ssa(0); got != 18446744073709551615 {
		fmt.Printf("sub_uint64 0%s1 = %d, wanted 18446744073709551615\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint64_ssa(1); got != 0 {
		fmt.Printf("sub_uint64 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_1_ssa(1); got != 0 {
		fmt.Printf("sub_uint64 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint64_ssa(4294967296); got != 18446744069414584321 {
		fmt.Printf("sub_uint64 1%s4294967296 = %d, wanted 18446744069414584321\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_1_ssa(4294967296); got != 4294967295 {
		fmt.Printf("sub_uint64 4294967296%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint64_ssa(18446744073709551615); got != 2 {
		fmt.Printf("sub_uint64 1%s18446744073709551615 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_1_ssa(18446744073709551615); got != 18446744073709551614 {
		fmt.Printf("sub_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551614\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_uint64_ssa(0); got != 4294967296 {
		fmt.Printf("sub_uint64 4294967296%s0 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_4294967296_ssa(0); got != 18446744069414584320 {
		fmt.Printf("sub_uint64 0%s4294967296 = %d, wanted 18446744069414584320\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_uint64_ssa(1); got != 4294967295 {
		fmt.Printf("sub_uint64 4294967296%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_4294967296_ssa(1); got != 18446744069414584321 {
		fmt.Printf("sub_uint64 1%s4294967296 = %d, wanted 18446744069414584321\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("sub_uint64 4294967296%s4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("sub_uint64 4294967296%s4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_uint64_ssa(18446744073709551615); got != 4294967297 {
		fmt.Printf("sub_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_4294967296_ssa(18446744073709551615); got != 18446744069414584319 {
		fmt.Printf("sub_uint64 18446744073709551615%s4294967296 = %d, wanted 18446744069414584319\n", `-`, got)
		failed = true
	}

	if got := sub_18446744073709551615_uint64_ssa(0); got != 18446744073709551615 {
		fmt.Printf("sub_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_18446744073709551615_ssa(0); got != 1 {
		fmt.Printf("sub_uint64 0%s18446744073709551615 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_18446744073709551615_uint64_ssa(1); got != 18446744073709551614 {
		fmt.Printf("sub_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551614\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_18446744073709551615_ssa(1); got != 2 {
		fmt.Printf("sub_uint64 1%s18446744073709551615 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_18446744073709551615_uint64_ssa(4294967296); got != 18446744069414584319 {
		fmt.Printf("sub_uint64 18446744073709551615%s4294967296 = %d, wanted 18446744069414584319\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_18446744073709551615_ssa(4294967296); got != 4294967297 {
		fmt.Printf("sub_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_18446744073709551615_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("sub_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint64_18446744073709551615_ssa(18446744073709551615); got != 0 {
		fmt.Printf("sub_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_0_uint64_ssa(1); got != 0 {
		fmt.Printf("div_uint64 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("div_uint64 0%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("div_uint64 0%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint64_1_ssa(0); got != 0 {
		fmt.Printf("div_uint64 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_uint64_ssa(1); got != 1 {
		fmt.Printf("div_uint64 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint64_1_ssa(1); got != 1 {
		fmt.Printf("div_uint64 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("div_uint64 1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint64_1_ssa(4294967296); got != 4294967296 {
		fmt.Printf("div_uint64 4294967296%s1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_1_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("div_uint64 1%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint64_1_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("div_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551615\n", `/`, got)
		failed = true
	}

	if got := div_uint64_4294967296_ssa(0); got != 0 {
		fmt.Printf("div_uint64 0%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_uint64_ssa(1); got != 4294967296 {
		fmt.Printf("div_uint64 4294967296%s1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_uint64_4294967296_ssa(1); got != 0 {
		fmt.Printf("div_uint64 1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_uint64_ssa(4294967296); got != 1 {
		fmt.Printf("div_uint64 4294967296%s4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint64_4294967296_ssa(4294967296); got != 1 {
		fmt.Printf("div_uint64 4294967296%s4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("div_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint64_4294967296_ssa(18446744073709551615); got != 4294967295 {
		fmt.Printf("div_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `/`, got)
		failed = true
	}

	if got := div_uint64_18446744073709551615_ssa(0); got != 0 {
		fmt.Printf("div_uint64 0%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_18446744073709551615_uint64_ssa(1); got != 18446744073709551615 {
		fmt.Printf("div_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551615\n", `/`, got)
		failed = true
	}

	if got := div_uint64_18446744073709551615_ssa(1); got != 0 {
		fmt.Printf("div_uint64 1%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_18446744073709551615_uint64_ssa(4294967296); got != 4294967295 {
		fmt.Printf("div_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `/`, got)
		failed = true
	}

	if got := div_uint64_18446744073709551615_ssa(4294967296); got != 0 {
		fmt.Printf("div_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_18446744073709551615_uint64_ssa(18446744073709551615); got != 1 {
		fmt.Printf("div_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint64_18446744073709551615_ssa(18446744073709551615); got != 1 {
		fmt.Printf("div_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_0_uint64_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_0_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint64_ssa(1); got != 0 {
		fmt.Printf("mul_uint64 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_0_ssa(1); got != 0 {
		fmt.Printf("mul_uint64 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_uint64 0%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_0_ssa(4294967296); got != 0 {
		fmt.Printf("mul_uint64 4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mul_uint64 0%s18446744073709551615 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_0_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mul_uint64 18446744073709551615%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint64_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_1_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint64_ssa(1); got != 1 {
		fmt.Printf("mul_uint64 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_1_ssa(1); got != 1 {
		fmt.Printf("mul_uint64 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint64_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_uint64 1%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_1_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_uint64 4294967296%s1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint64_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("mul_uint64 1%s18446744073709551615 = %d, wanted 18446744073709551615\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_1_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("mul_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551615\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_uint64_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_4294967296_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 0%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_uint64_ssa(1); got != 4294967296 {
		fmt.Printf("mul_uint64 4294967296%s1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_4294967296_ssa(1); got != 4294967296 {
		fmt.Printf("mul_uint64 1%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_uint64 4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mul_uint64 4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_uint64_ssa(18446744073709551615); got != 18446744069414584320 {
		fmt.Printf("mul_uint64 4294967296%s18446744073709551615 = %d, wanted 18446744069414584320\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_4294967296_ssa(18446744073709551615); got != 18446744069414584320 {
		fmt.Printf("mul_uint64 18446744073709551615%s4294967296 = %d, wanted 18446744069414584320\n", `*`, got)
		failed = true
	}

	if got := mul_18446744073709551615_uint64_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 18446744073709551615%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_18446744073709551615_ssa(0); got != 0 {
		fmt.Printf("mul_uint64 0%s18446744073709551615 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_18446744073709551615_uint64_ssa(1); got != 18446744073709551615 {
		fmt.Printf("mul_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551615\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_18446744073709551615_ssa(1); got != 18446744073709551615 {
		fmt.Printf("mul_uint64 1%s18446744073709551615 = %d, wanted 18446744073709551615\n", `*`, got)
		failed = true
	}

	if got := mul_18446744073709551615_uint64_ssa(4294967296); got != 18446744069414584320 {
		fmt.Printf("mul_uint64 18446744073709551615%s4294967296 = %d, wanted 18446744069414584320\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_18446744073709551615_ssa(4294967296); got != 18446744069414584320 {
		fmt.Printf("mul_uint64 4294967296%s18446744073709551615 = %d, wanted 18446744069414584320\n", `*`, got)
		failed = true
	}

	if got := mul_18446744073709551615_uint64_ssa(18446744073709551615); got != 1 {
		fmt.Printf("mul_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint64_18446744073709551615_ssa(18446744073709551615); got != 1 {
		fmt.Printf("mul_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := lsh_0_uint64_ssa(0); got != 0 {
		fmt.Printf("lsh_uint64 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_0_ssa(0); got != 0 {
		fmt.Printf("lsh_uint64 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint64_ssa(1); got != 0 {
		fmt.Printf("lsh_uint64 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_0_ssa(1); got != 1 {
		fmt.Printf("lsh_uint64 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 0%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("lsh_uint64 4294967296%s0 = %d, wanted 4294967296\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 0%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_0_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("lsh_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint64_ssa(0); got != 1 {
		fmt.Printf("lsh_uint64 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_1_ssa(0); got != 0 {
		fmt.Printf("lsh_uint64 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint64_ssa(1); got != 2 {
		fmt.Printf("lsh_uint64 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_1_ssa(1); got != 2 {
		fmt.Printf("lsh_uint64 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 1%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_1_ssa(4294967296); got != 8589934592 {
		fmt.Printf("lsh_uint64 4294967296%s1 = %d, wanted 8589934592\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 1%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_1_ssa(18446744073709551615); got != 18446744073709551614 {
		fmt.Printf("lsh_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551614\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967296_uint64_ssa(0); got != 4294967296 {
		fmt.Printf("lsh_uint64 4294967296%s0 = %d, wanted 4294967296\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_4294967296_ssa(0); got != 0 {
		fmt.Printf("lsh_uint64 0%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967296_uint64_ssa(1); got != 8589934592 {
		fmt.Printf("lsh_uint64 4294967296%s1 = %d, wanted 8589934592\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_4294967296_ssa(1); got != 0 {
		fmt.Printf("lsh_uint64 1%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967296_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 4294967296%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 4294967296%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967296_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_4294967296_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 18446744073709551615%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_18446744073709551615_uint64_ssa(0); got != 18446744073709551615 {
		fmt.Printf("lsh_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_18446744073709551615_ssa(0); got != 0 {
		fmt.Printf("lsh_uint64 0%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_18446744073709551615_uint64_ssa(1); got != 18446744073709551614 {
		fmt.Printf("lsh_uint64 18446744073709551615%s1 = %d, wanted 18446744073709551614\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_18446744073709551615_ssa(1); got != 0 {
		fmt.Printf("lsh_uint64 1%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_18446744073709551615_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 18446744073709551615%s4294967296 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_18446744073709551615_ssa(4294967296); got != 0 {
		fmt.Printf("lsh_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_18446744073709551615_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint64_18446744073709551615_ssa(18446744073709551615); got != 0 {
		fmt.Printf("lsh_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := rsh_0_uint64_ssa(0); got != 0 {
		fmt.Printf("rsh_uint64 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_0_ssa(0); got != 0 {
		fmt.Printf("rsh_uint64 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint64_ssa(1); got != 0 {
		fmt.Printf("rsh_uint64 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_0_ssa(1); got != 1 {
		fmt.Printf("rsh_uint64 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 0%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("rsh_uint64 4294967296%s0 = %d, wanted 4294967296\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 0%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_0_ssa(18446744073709551615); got != 18446744073709551615 {
		fmt.Printf("rsh_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint64_ssa(0); got != 1 {
		fmt.Printf("rsh_uint64 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_1_ssa(0); got != 0 {
		fmt.Printf("rsh_uint64 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint64_ssa(1); got != 0 {
		fmt.Printf("rsh_uint64 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_1_ssa(1); got != 0 {
		fmt.Printf("rsh_uint64 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 1%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_1_ssa(4294967296); got != 2147483648 {
		fmt.Printf("rsh_uint64 4294967296%s1 = %d, wanted 2147483648\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 1%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_1_ssa(18446744073709551615); got != 9223372036854775807 {
		fmt.Printf("rsh_uint64 18446744073709551615%s1 = %d, wanted 9223372036854775807\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967296_uint64_ssa(0); got != 4294967296 {
		fmt.Printf("rsh_uint64 4294967296%s0 = %d, wanted 4294967296\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_4294967296_ssa(0); got != 0 {
		fmt.Printf("rsh_uint64 0%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967296_uint64_ssa(1); got != 2147483648 {
		fmt.Printf("rsh_uint64 4294967296%s1 = %d, wanted 2147483648\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_4294967296_ssa(1); got != 0 {
		fmt.Printf("rsh_uint64 1%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967296_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 4294967296%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 4294967296%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967296_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_4294967296_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 18446744073709551615%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_18446744073709551615_uint64_ssa(0); got != 18446744073709551615 {
		fmt.Printf("rsh_uint64 18446744073709551615%s0 = %d, wanted 18446744073709551615\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_18446744073709551615_ssa(0); got != 0 {
		fmt.Printf("rsh_uint64 0%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_18446744073709551615_uint64_ssa(1); got != 9223372036854775807 {
		fmt.Printf("rsh_uint64 18446744073709551615%s1 = %d, wanted 9223372036854775807\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_18446744073709551615_ssa(1); got != 0 {
		fmt.Printf("rsh_uint64 1%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_18446744073709551615_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 18446744073709551615%s4294967296 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_18446744073709551615_ssa(4294967296); got != 0 {
		fmt.Printf("rsh_uint64 4294967296%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_18446744073709551615_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint64_18446744073709551615_ssa(18446744073709551615); got != 0 {
		fmt.Printf("rsh_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := mod_0_uint64_ssa(1); got != 0 {
		fmt.Printf("mod_uint64 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_uint64 0%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mod_uint64 0%s18446744073709551615 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_1_ssa(0); got != 0 {
		fmt.Printf("mod_uint64 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint64_ssa(1); got != 0 {
		fmt.Printf("mod_uint64 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_1_ssa(1); got != 0 {
		fmt.Printf("mod_uint64 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint64_ssa(4294967296); got != 1 {
		fmt.Printf("mod_uint64 1%s4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_1_ssa(4294967296); got != 0 {
		fmt.Printf("mod_uint64 4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint64_ssa(18446744073709551615); got != 1 {
		fmt.Printf("mod_uint64 1%s18446744073709551615 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_1_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mod_uint64 18446744073709551615%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_4294967296_ssa(0); got != 0 {
		fmt.Printf("mod_uint64 0%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_uint64_ssa(1); got != 0 {
		fmt.Printf("mod_uint64 4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_4294967296_ssa(1); got != 1 {
		fmt.Printf("mod_uint64 1%s4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_uint64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_uint64 4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mod_uint64 4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_uint64_ssa(18446744073709551615); got != 4294967296 {
		fmt.Printf("mod_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_4294967296_ssa(18446744073709551615); got != 4294967295 {
		fmt.Printf("mod_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_18446744073709551615_ssa(0); got != 0 {
		fmt.Printf("mod_uint64 0%s18446744073709551615 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_18446744073709551615_uint64_ssa(1); got != 0 {
		fmt.Printf("mod_uint64 18446744073709551615%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_18446744073709551615_ssa(1); got != 1 {
		fmt.Printf("mod_uint64 1%s18446744073709551615 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_18446744073709551615_uint64_ssa(4294967296); got != 4294967295 {
		fmt.Printf("mod_uint64 18446744073709551615%s4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_18446744073709551615_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mod_uint64 4294967296%s18446744073709551615 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_18446744073709551615_uint64_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mod_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint64_18446744073709551615_ssa(18446744073709551615); got != 0 {
		fmt.Printf("mod_uint64 18446744073709551615%s18446744073709551615 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("add_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("add_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("add_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("add_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(-4294967296); got != 9223372032559808512 {
		fmt.Printf("add_int64 -9223372036854775808%s-4294967296 = %d, wanted 9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(-4294967296); got != 9223372032559808512 {
		fmt.Printf("add_int64 -4294967296%s-9223372036854775808 = %d, wanted 9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775808%s-1 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("add_int64 -1%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(0); got != -9223372036854775808 {
		fmt.Printf("add_int64 -9223372036854775808%s0 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(0); got != -9223372036854775808 {
		fmt.Printf("add_int64 0%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(1); got != -9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(1); got != -9223372036854775807 {
		fmt.Printf("add_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(4294967296); got != -9223372032559808512 {
		fmt.Printf("add_int64 -9223372036854775808%s4294967296 = %d, wanted -9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(4294967296); got != -9223372032559808512 {
		fmt.Printf("add_int64 4294967296%s-9223372036854775808 = %d, wanted -9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(9223372036854775806); got != -2 {
		fmt.Printf("add_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(9223372036854775806); got != -2 {
		fmt.Printf("add_int64 9223372036854775806%s-9223372036854775808 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775808_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("add_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775808_ssa(9223372036854775807); got != -1 {
		fmt.Printf("add_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("add_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("add_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(-9223372036854775807); got != 2 {
		fmt.Printf("add_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(-9223372036854775807); got != 2 {
		fmt.Printf("add_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(-4294967296); got != 9223372032559808513 {
		fmt.Printf("add_int64 -9223372036854775807%s-4294967296 = %d, wanted 9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(-4294967296); got != 9223372032559808513 {
		fmt.Printf("add_int64 -4294967296%s-9223372036854775807 = %d, wanted 9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("add_int64 -9223372036854775807%s-1 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("add_int64 -1%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(0); got != -9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775807%s0 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(0); got != -9223372036854775807 {
		fmt.Printf("add_int64 0%s-9223372036854775807 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(1); got != -9223372036854775806 {
		fmt.Printf("add_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(1); got != -9223372036854775806 {
		fmt.Printf("add_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(4294967296); got != -9223372032559808511 {
		fmt.Printf("add_int64 -9223372036854775807%s4294967296 = %d, wanted -9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(4294967296); got != -9223372032559808511 {
		fmt.Printf("add_int64 4294967296%s-9223372036854775807 = %d, wanted -9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(9223372036854775806); got != -1 {
		fmt.Printf("add_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(9223372036854775806); got != -1 {
		fmt.Printf("add_int64 9223372036854775806%s-9223372036854775807 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg9223372036854775807_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("add_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg9223372036854775807_ssa(9223372036854775807); got != 0 {
		fmt.Printf("add_int64 9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(-9223372036854775808); got != 9223372032559808512 {
		fmt.Printf("add_int64 -4294967296%s-9223372036854775808 = %d, wanted 9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(-9223372036854775808); got != 9223372032559808512 {
		fmt.Printf("add_int64 -9223372036854775808%s-4294967296 = %d, wanted 9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(-9223372036854775807); got != 9223372032559808513 {
		fmt.Printf("add_int64 -4294967296%s-9223372036854775807 = %d, wanted 9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(-9223372036854775807); got != 9223372032559808513 {
		fmt.Printf("add_int64 -9223372036854775807%s-4294967296 = %d, wanted 9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(-4294967296); got != -8589934592 {
		fmt.Printf("add_int64 -4294967296%s-4294967296 = %d, wanted -8589934592\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(-4294967296); got != -8589934592 {
		fmt.Printf("add_int64 -4294967296%s-4294967296 = %d, wanted -8589934592\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(-1); got != -4294967297 {
		fmt.Printf("add_int64 -4294967296%s-1 = %d, wanted -4294967297\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(-1); got != -4294967297 {
		fmt.Printf("add_int64 -1%s-4294967296 = %d, wanted -4294967297\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(0); got != -4294967296 {
		fmt.Printf("add_int64 -4294967296%s0 = %d, wanted -4294967296\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(0); got != -4294967296 {
		fmt.Printf("add_int64 0%s-4294967296 = %d, wanted -4294967296\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(1); got != -4294967295 {
		fmt.Printf("add_int64 -4294967296%s1 = %d, wanted -4294967295\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(1); got != -4294967295 {
		fmt.Printf("add_int64 1%s-4294967296 = %d, wanted -4294967295\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("add_int64 -4294967296%s4294967296 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("add_int64 4294967296%s-4294967296 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(9223372036854775806); got != 9223372032559808510 {
		fmt.Printf("add_int64 -4294967296%s9223372036854775806 = %d, wanted 9223372032559808510\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(9223372036854775806); got != 9223372032559808510 {
		fmt.Printf("add_int64 9223372036854775806%s-4294967296 = %d, wanted 9223372032559808510\n", `+`, got)
		failed = true
	}

	if got := add_Neg4294967296_int64_ssa(9223372036854775807); got != 9223372032559808511 {
		fmt.Printf("add_int64 -4294967296%s9223372036854775807 = %d, wanted 9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg4294967296_ssa(9223372036854775807); got != 9223372032559808511 {
		fmt.Printf("add_int64 9223372036854775807%s-4294967296 = %d, wanted 9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(-9223372036854775808); got != 9223372036854775807 {
		fmt.Printf("add_int64 -1%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(-9223372036854775808); got != 9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775808%s-1 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("add_int64 -1%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("add_int64 -9223372036854775807%s-1 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(-4294967296); got != -4294967297 {
		fmt.Printf("add_int64 -1%s-4294967296 = %d, wanted -4294967297\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(-4294967296); got != -4294967297 {
		fmt.Printf("add_int64 -4294967296%s-1 = %d, wanted -4294967297\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(-1); got != -2 {
		fmt.Printf("add_int64 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(-1); got != -2 {
		fmt.Printf("add_int64 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(0); got != -1 {
		fmt.Printf("add_int64 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(0); got != -1 {
		fmt.Printf("add_int64 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(1); got != 0 {
		fmt.Printf("add_int64 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(1); got != 0 {
		fmt.Printf("add_int64 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(4294967296); got != 4294967295 {
		fmt.Printf("add_int64 -1%s4294967296 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(4294967296); got != 4294967295 {
		fmt.Printf("add_int64 4294967296%s-1 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(9223372036854775806); got != 9223372036854775805 {
		fmt.Printf("add_int64 -1%s9223372036854775806 = %d, wanted 9223372036854775805\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(9223372036854775806); got != 9223372036854775805 {
		fmt.Printf("add_int64 9223372036854775806%s-1 = %d, wanted 9223372036854775805\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int64_ssa(9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("add_int64 -1%s9223372036854775807 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_Neg1_ssa(9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("add_int64 9223372036854775807%s-1 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("add_int64 0%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("add_int64 -9223372036854775808%s0 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("add_int64 0%s-9223372036854775807 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775807%s0 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("add_int64 0%s-4294967296 = %d, wanted -4294967296\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("add_int64 -4294967296%s0 = %d, wanted -4294967296\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(-1); got != -1 {
		fmt.Printf("add_int64 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(-1); got != -1 {
		fmt.Printf("add_int64 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(0); got != 0 {
		fmt.Printf("add_int64 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(0); got != 0 {
		fmt.Printf("add_int64 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(1); got != 1 {
		fmt.Printf("add_int64 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(1); got != 1 {
		fmt.Printf("add_int64 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(4294967296); got != 4294967296 {
		fmt.Printf("add_int64 0%s4294967296 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("add_int64 4294967296%s0 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("add_int64 0%s9223372036854775806 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("add_int64 9223372036854775806%s0 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_0_int64_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("add_int64 0%s9223372036854775807 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_0_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("add_int64 9223372036854775807%s0 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(-9223372036854775808); got != -9223372036854775807 {
		fmt.Printf("add_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(-9223372036854775808); got != -9223372036854775807 {
		fmt.Printf("add_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(-9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("add_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(-9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("add_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(-4294967296); got != -4294967295 {
		fmt.Printf("add_int64 1%s-4294967296 = %d, wanted -4294967295\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(-4294967296); got != -4294967295 {
		fmt.Printf("add_int64 -4294967296%s1 = %d, wanted -4294967295\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(-1); got != 0 {
		fmt.Printf("add_int64 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(-1); got != 0 {
		fmt.Printf("add_int64 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(0); got != 1 {
		fmt.Printf("add_int64 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(0); got != 1 {
		fmt.Printf("add_int64 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(1); got != 2 {
		fmt.Printf("add_int64 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(1); got != 2 {
		fmt.Printf("add_int64 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(4294967296); got != 4294967297 {
		fmt.Printf("add_int64 1%s4294967296 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(4294967296); got != 4294967297 {
		fmt.Printf("add_int64 4294967296%s1 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(9223372036854775806); got != 9223372036854775807 {
		fmt.Printf("add_int64 1%s9223372036854775806 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(9223372036854775806); got != 9223372036854775807 {
		fmt.Printf("add_int64 9223372036854775806%s1 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_1_int64_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("add_int64 1%s9223372036854775807 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_1_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("add_int64 9223372036854775807%s1 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(-9223372036854775808); got != -9223372032559808512 {
		fmt.Printf("add_int64 4294967296%s-9223372036854775808 = %d, wanted -9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(-9223372036854775808); got != -9223372032559808512 {
		fmt.Printf("add_int64 -9223372036854775808%s4294967296 = %d, wanted -9223372032559808512\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(-9223372036854775807); got != -9223372032559808511 {
		fmt.Printf("add_int64 4294967296%s-9223372036854775807 = %d, wanted -9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(-9223372036854775807); got != -9223372032559808511 {
		fmt.Printf("add_int64 -9223372036854775807%s4294967296 = %d, wanted -9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("add_int64 4294967296%s-4294967296 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("add_int64 -4294967296%s4294967296 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(-1); got != 4294967295 {
		fmt.Printf("add_int64 4294967296%s-1 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(-1); got != 4294967295 {
		fmt.Printf("add_int64 -1%s4294967296 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(0); got != 4294967296 {
		fmt.Printf("add_int64 4294967296%s0 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(0); got != 4294967296 {
		fmt.Printf("add_int64 0%s4294967296 = %d, wanted 4294967296\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(1); got != 4294967297 {
		fmt.Printf("add_int64 4294967296%s1 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(1); got != 4294967297 {
		fmt.Printf("add_int64 1%s4294967296 = %d, wanted 4294967297\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(4294967296); got != 8589934592 {
		fmt.Printf("add_int64 4294967296%s4294967296 = %d, wanted 8589934592\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(4294967296); got != 8589934592 {
		fmt.Printf("add_int64 4294967296%s4294967296 = %d, wanted 8589934592\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(9223372036854775806); got != -9223372032559808514 {
		fmt.Printf("add_int64 4294967296%s9223372036854775806 = %d, wanted -9223372032559808514\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(9223372036854775806); got != -9223372032559808514 {
		fmt.Printf("add_int64 9223372036854775806%s4294967296 = %d, wanted -9223372032559808514\n", `+`, got)
		failed = true
	}

	if got := add_4294967296_int64_ssa(9223372036854775807); got != -9223372032559808513 {
		fmt.Printf("add_int64 4294967296%s9223372036854775807 = %d, wanted -9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_int64_4294967296_ssa(9223372036854775807); got != -9223372032559808513 {
		fmt.Printf("add_int64 9223372036854775807%s4294967296 = %d, wanted -9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(-9223372036854775808); got != -2 {
		fmt.Printf("add_int64 9223372036854775806%s-9223372036854775808 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(-9223372036854775808); got != -2 {
		fmt.Printf("add_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("add_int64 9223372036854775806%s-9223372036854775807 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("add_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(-4294967296); got != 9223372032559808510 {
		fmt.Printf("add_int64 9223372036854775806%s-4294967296 = %d, wanted 9223372032559808510\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(-4294967296); got != 9223372032559808510 {
		fmt.Printf("add_int64 -4294967296%s9223372036854775806 = %d, wanted 9223372032559808510\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(-1); got != 9223372036854775805 {
		fmt.Printf("add_int64 9223372036854775806%s-1 = %d, wanted 9223372036854775805\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(-1); got != 9223372036854775805 {
		fmt.Printf("add_int64 -1%s9223372036854775806 = %d, wanted 9223372036854775805\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(0); got != 9223372036854775806 {
		fmt.Printf("add_int64 9223372036854775806%s0 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(0); got != 9223372036854775806 {
		fmt.Printf("add_int64 0%s9223372036854775806 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(1); got != 9223372036854775807 {
		fmt.Printf("add_int64 9223372036854775806%s1 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(1); got != 9223372036854775807 {
		fmt.Printf("add_int64 1%s9223372036854775806 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(4294967296); got != -9223372032559808514 {
		fmt.Printf("add_int64 9223372036854775806%s4294967296 = %d, wanted -9223372032559808514\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(4294967296); got != -9223372032559808514 {
		fmt.Printf("add_int64 4294967296%s9223372036854775806 = %d, wanted -9223372032559808514\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(9223372036854775806); got != -4 {
		fmt.Printf("add_int64 9223372036854775806%s9223372036854775806 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(9223372036854775806); got != -4 {
		fmt.Printf("add_int64 9223372036854775806%s9223372036854775806 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775806_int64_ssa(9223372036854775807); got != -3 {
		fmt.Printf("add_int64 9223372036854775806%s9223372036854775807 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775806_ssa(9223372036854775807); got != -3 {
		fmt.Printf("add_int64 9223372036854775807%s9223372036854775806 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("add_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("add_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("add_int64 9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("add_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(-4294967296); got != 9223372032559808511 {
		fmt.Printf("add_int64 9223372036854775807%s-4294967296 = %d, wanted 9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(-4294967296); got != 9223372032559808511 {
		fmt.Printf("add_int64 -4294967296%s9223372036854775807 = %d, wanted 9223372032559808511\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(-1); got != 9223372036854775806 {
		fmt.Printf("add_int64 9223372036854775807%s-1 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(-1); got != 9223372036854775806 {
		fmt.Printf("add_int64 -1%s9223372036854775807 = %d, wanted 9223372036854775806\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(0); got != 9223372036854775807 {
		fmt.Printf("add_int64 9223372036854775807%s0 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(0); got != 9223372036854775807 {
		fmt.Printf("add_int64 0%s9223372036854775807 = %d, wanted 9223372036854775807\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(1); got != -9223372036854775808 {
		fmt.Printf("add_int64 9223372036854775807%s1 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(1); got != -9223372036854775808 {
		fmt.Printf("add_int64 1%s9223372036854775807 = %d, wanted -9223372036854775808\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(4294967296); got != -9223372032559808513 {
		fmt.Printf("add_int64 9223372036854775807%s4294967296 = %d, wanted -9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(4294967296); got != -9223372032559808513 {
		fmt.Printf("add_int64 4294967296%s9223372036854775807 = %d, wanted -9223372032559808513\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(9223372036854775806); got != -3 {
		fmt.Printf("add_int64 9223372036854775807%s9223372036854775806 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(9223372036854775806); got != -3 {
		fmt.Printf("add_int64 9223372036854775806%s9223372036854775807 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_9223372036854775807_int64_ssa(9223372036854775807); got != -2 {
		fmt.Printf("add_int64 9223372036854775807%s9223372036854775807 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int64_9223372036854775807_ssa(9223372036854775807); got != -2 {
		fmt.Printf("add_int64 9223372036854775807%s9223372036854775807 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("sub_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("sub_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("sub_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("sub_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(-4294967296); got != -9223372032559808512 {
		fmt.Printf("sub_int64 -9223372036854775808%s-4294967296 = %d, wanted -9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(-4294967296); got != 9223372032559808512 {
		fmt.Printf("sub_int64 -4294967296%s-9223372036854775808 = %d, wanted 9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(-1); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("sub_int64 -1%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(0); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -9223372036854775808%s0 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(0); got != -9223372036854775808 {
		fmt.Printf("sub_int64 0%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(1); got != 9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775808%s1 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(1); got != -9223372036854775807 {
		fmt.Printf("sub_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(4294967296); got != 9223372032559808512 {
		fmt.Printf("sub_int64 -9223372036854775808%s4294967296 = %d, wanted 9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(4294967296); got != -9223372032559808512 {
		fmt.Printf("sub_int64 4294967296%s-9223372036854775808 = %d, wanted -9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(9223372036854775806); got != 2 {
		fmt.Printf("sub_int64 -9223372036854775808%s9223372036854775806 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(9223372036854775806); got != -2 {
		fmt.Printf("sub_int64 9223372036854775806%s-9223372036854775808 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775808_int64_ssa(9223372036854775807); got != 1 {
		fmt.Printf("sub_int64 -9223372036854775808%s9223372036854775807 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775808_ssa(9223372036854775807); got != -1 {
		fmt.Printf("sub_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("sub_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("sub_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("sub_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("sub_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(-4294967296); got != -9223372032559808511 {
		fmt.Printf("sub_int64 -9223372036854775807%s-4294967296 = %d, wanted -9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(-4294967296); got != 9223372032559808511 {
		fmt.Printf("sub_int64 -4294967296%s-9223372036854775807 = %d, wanted 9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(-1); got != -9223372036854775806 {
		fmt.Printf("sub_int64 -9223372036854775807%s-1 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(-1); got != 9223372036854775806 {
		fmt.Printf("sub_int64 -1%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(0); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775807%s0 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(0); got != 9223372036854775807 {
		fmt.Printf("sub_int64 0%s-9223372036854775807 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(1); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(1); got != -9223372036854775808 {
		fmt.Printf("sub_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(4294967296); got != 9223372032559808513 {
		fmt.Printf("sub_int64 -9223372036854775807%s4294967296 = %d, wanted 9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(4294967296); got != -9223372032559808513 {
		fmt.Printf("sub_int64 4294967296%s-9223372036854775807 = %d, wanted -9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(9223372036854775806); got != 3 {
		fmt.Printf("sub_int64 -9223372036854775807%s9223372036854775806 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(9223372036854775806); got != -3 {
		fmt.Printf("sub_int64 9223372036854775806%s-9223372036854775807 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_Neg9223372036854775807_int64_ssa(9223372036854775807); got != 2 {
		fmt.Printf("sub_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg9223372036854775807_ssa(9223372036854775807); got != -2 {
		fmt.Printf("sub_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(-9223372036854775808); got != 9223372032559808512 {
		fmt.Printf("sub_int64 -4294967296%s-9223372036854775808 = %d, wanted 9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(-9223372036854775808); got != -9223372032559808512 {
		fmt.Printf("sub_int64 -9223372036854775808%s-4294967296 = %d, wanted -9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(-9223372036854775807); got != 9223372032559808511 {
		fmt.Printf("sub_int64 -4294967296%s-9223372036854775807 = %d, wanted 9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(-9223372036854775807); got != -9223372032559808511 {
		fmt.Printf("sub_int64 -9223372036854775807%s-4294967296 = %d, wanted -9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("sub_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("sub_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(-1); got != -4294967295 {
		fmt.Printf("sub_int64 -4294967296%s-1 = %d, wanted -4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(-1); got != 4294967295 {
		fmt.Printf("sub_int64 -1%s-4294967296 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(0); got != -4294967296 {
		fmt.Printf("sub_int64 -4294967296%s0 = %d, wanted -4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(0); got != 4294967296 {
		fmt.Printf("sub_int64 0%s-4294967296 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(1); got != -4294967297 {
		fmt.Printf("sub_int64 -4294967296%s1 = %d, wanted -4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(1); got != 4294967297 {
		fmt.Printf("sub_int64 1%s-4294967296 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(4294967296); got != -8589934592 {
		fmt.Printf("sub_int64 -4294967296%s4294967296 = %d, wanted -8589934592\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(4294967296); got != 8589934592 {
		fmt.Printf("sub_int64 4294967296%s-4294967296 = %d, wanted 8589934592\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(9223372036854775806); got != 9223372032559808514 {
		fmt.Printf("sub_int64 -4294967296%s9223372036854775806 = %d, wanted 9223372032559808514\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(9223372036854775806); got != -9223372032559808514 {
		fmt.Printf("sub_int64 9223372036854775806%s-4294967296 = %d, wanted -9223372032559808514\n", `-`, got)
		failed = true
	}

	if got := sub_Neg4294967296_int64_ssa(9223372036854775807); got != 9223372032559808513 {
		fmt.Printf("sub_int64 -4294967296%s9223372036854775807 = %d, wanted 9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg4294967296_ssa(9223372036854775807); got != -9223372032559808513 {
		fmt.Printf("sub_int64 9223372036854775807%s-4294967296 = %d, wanted -9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(-9223372036854775808); got != 9223372036854775807 {
		fmt.Printf("sub_int64 -1%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(-9223372036854775808); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(-9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("sub_int64 -1%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(-9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("sub_int64 -9223372036854775807%s-1 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(-4294967296); got != 4294967295 {
		fmt.Printf("sub_int64 -1%s-4294967296 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(-4294967296); got != -4294967295 {
		fmt.Printf("sub_int64 -4294967296%s-1 = %d, wanted -4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(-1); got != 0 {
		fmt.Printf("sub_int64 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(-1); got != 0 {
		fmt.Printf("sub_int64 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(0); got != -1 {
		fmt.Printf("sub_int64 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(0); got != 1 {
		fmt.Printf("sub_int64 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(1); got != -2 {
		fmt.Printf("sub_int64 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(1); got != 2 {
		fmt.Printf("sub_int64 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(4294967296); got != -4294967297 {
		fmt.Printf("sub_int64 -1%s4294967296 = %d, wanted -4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(4294967296); got != 4294967297 {
		fmt.Printf("sub_int64 4294967296%s-1 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(9223372036854775806); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -1%s9223372036854775806 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(9223372036854775806); got != 9223372036854775807 {
		fmt.Printf("sub_int64 9223372036854775806%s-1 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int64_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -1%s9223372036854775807 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_Neg1_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("sub_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("sub_int64 0%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -9223372036854775808%s0 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(-9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("sub_int64 0%s-9223372036854775807 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775807%s0 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("sub_int64 0%s-4294967296 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("sub_int64 -4294967296%s0 = %d, wanted -4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(-1); got != 1 {
		fmt.Printf("sub_int64 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(-1); got != -1 {
		fmt.Printf("sub_int64 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(0); got != 0 {
		fmt.Printf("sub_int64 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(0); got != 0 {
		fmt.Printf("sub_int64 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(1); got != -1 {
		fmt.Printf("sub_int64 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(1); got != 1 {
		fmt.Printf("sub_int64 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(4294967296); got != -4294967296 {
		fmt.Printf("sub_int64 0%s4294967296 = %d, wanted -4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(4294967296); got != 4294967296 {
		fmt.Printf("sub_int64 4294967296%s0 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("sub_int64 0%s9223372036854775806 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("sub_int64 9223372036854775806%s0 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_0_int64_ssa(9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("sub_int64 0%s9223372036854775807 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_0_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("sub_int64 9223372036854775807%s0 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(-9223372036854775808); got != -9223372036854775807 {
		fmt.Printf("sub_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(-9223372036854775808); got != 9223372036854775807 {
		fmt.Printf("sub_int64 -9223372036854775808%s1 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("sub_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(-4294967296); got != 4294967297 {
		fmt.Printf("sub_int64 1%s-4294967296 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(-4294967296); got != -4294967297 {
		fmt.Printf("sub_int64 -4294967296%s1 = %d, wanted -4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(-1); got != 2 {
		fmt.Printf("sub_int64 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(-1); got != -2 {
		fmt.Printf("sub_int64 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(0); got != 1 {
		fmt.Printf("sub_int64 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(0); got != -1 {
		fmt.Printf("sub_int64 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(1); got != 0 {
		fmt.Printf("sub_int64 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(1); got != 0 {
		fmt.Printf("sub_int64 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(4294967296); got != -4294967295 {
		fmt.Printf("sub_int64 1%s4294967296 = %d, wanted -4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(4294967296); got != 4294967295 {
		fmt.Printf("sub_int64 4294967296%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(9223372036854775806); got != -9223372036854775805 {
		fmt.Printf("sub_int64 1%s9223372036854775806 = %d, wanted -9223372036854775805\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(9223372036854775806); got != 9223372036854775805 {
		fmt.Printf("sub_int64 9223372036854775806%s1 = %d, wanted 9223372036854775805\n", `-`, got)
		failed = true
	}

	if got := sub_1_int64_ssa(9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("sub_int64 1%s9223372036854775807 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_1_ssa(9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("sub_int64 9223372036854775807%s1 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(-9223372036854775808); got != -9223372032559808512 {
		fmt.Printf("sub_int64 4294967296%s-9223372036854775808 = %d, wanted -9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(-9223372036854775808); got != 9223372032559808512 {
		fmt.Printf("sub_int64 -9223372036854775808%s4294967296 = %d, wanted 9223372032559808512\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(-9223372036854775807); got != -9223372032559808513 {
		fmt.Printf("sub_int64 4294967296%s-9223372036854775807 = %d, wanted -9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(-9223372036854775807); got != 9223372032559808513 {
		fmt.Printf("sub_int64 -9223372036854775807%s4294967296 = %d, wanted 9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(-4294967296); got != 8589934592 {
		fmt.Printf("sub_int64 4294967296%s-4294967296 = %d, wanted 8589934592\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(-4294967296); got != -8589934592 {
		fmt.Printf("sub_int64 -4294967296%s4294967296 = %d, wanted -8589934592\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(-1); got != 4294967297 {
		fmt.Printf("sub_int64 4294967296%s-1 = %d, wanted 4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(-1); got != -4294967297 {
		fmt.Printf("sub_int64 -1%s4294967296 = %d, wanted -4294967297\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(0); got != 4294967296 {
		fmt.Printf("sub_int64 4294967296%s0 = %d, wanted 4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(0); got != -4294967296 {
		fmt.Printf("sub_int64 0%s4294967296 = %d, wanted -4294967296\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(1); got != 4294967295 {
		fmt.Printf("sub_int64 4294967296%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(1); got != -4294967295 {
		fmt.Printf("sub_int64 1%s4294967296 = %d, wanted -4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("sub_int64 4294967296%s4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("sub_int64 4294967296%s4294967296 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(9223372036854775806); got != -9223372032559808510 {
		fmt.Printf("sub_int64 4294967296%s9223372036854775806 = %d, wanted -9223372032559808510\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(9223372036854775806); got != 9223372032559808510 {
		fmt.Printf("sub_int64 9223372036854775806%s4294967296 = %d, wanted 9223372032559808510\n", `-`, got)
		failed = true
	}

	if got := sub_4294967296_int64_ssa(9223372036854775807); got != -9223372032559808511 {
		fmt.Printf("sub_int64 4294967296%s9223372036854775807 = %d, wanted -9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_int64_4294967296_ssa(9223372036854775807); got != 9223372032559808511 {
		fmt.Printf("sub_int64 9223372036854775807%s4294967296 = %d, wanted 9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(-9223372036854775808); got != -2 {
		fmt.Printf("sub_int64 9223372036854775806%s-9223372036854775808 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(-9223372036854775808); got != 2 {
		fmt.Printf("sub_int64 -9223372036854775808%s9223372036854775806 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(-9223372036854775807); got != -3 {
		fmt.Printf("sub_int64 9223372036854775806%s-9223372036854775807 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(-9223372036854775807); got != 3 {
		fmt.Printf("sub_int64 -9223372036854775807%s9223372036854775806 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(-4294967296); got != -9223372032559808514 {
		fmt.Printf("sub_int64 9223372036854775806%s-4294967296 = %d, wanted -9223372032559808514\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(-4294967296); got != 9223372032559808514 {
		fmt.Printf("sub_int64 -4294967296%s9223372036854775806 = %d, wanted 9223372032559808514\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("sub_int64 9223372036854775806%s-1 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(-1); got != -9223372036854775807 {
		fmt.Printf("sub_int64 -1%s9223372036854775806 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(0); got != 9223372036854775806 {
		fmt.Printf("sub_int64 9223372036854775806%s0 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(0); got != -9223372036854775806 {
		fmt.Printf("sub_int64 0%s9223372036854775806 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(1); got != 9223372036854775805 {
		fmt.Printf("sub_int64 9223372036854775806%s1 = %d, wanted 9223372036854775805\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(1); got != -9223372036854775805 {
		fmt.Printf("sub_int64 1%s9223372036854775806 = %d, wanted -9223372036854775805\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(4294967296); got != 9223372032559808510 {
		fmt.Printf("sub_int64 9223372036854775806%s4294967296 = %d, wanted 9223372032559808510\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(4294967296); got != -9223372032559808510 {
		fmt.Printf("sub_int64 4294967296%s9223372036854775806 = %d, wanted -9223372032559808510\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("sub_int64 9223372036854775806%s9223372036854775806 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(9223372036854775806); got != 0 {
		fmt.Printf("sub_int64 9223372036854775806%s9223372036854775806 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775806_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("sub_int64 9223372036854775806%s9223372036854775807 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775806_ssa(9223372036854775807); got != 1 {
		fmt.Printf("sub_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("sub_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("sub_int64 -9223372036854775808%s9223372036854775807 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(-9223372036854775807); got != -2 {
		fmt.Printf("sub_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(-9223372036854775807); got != 2 {
		fmt.Printf("sub_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(-4294967296); got != -9223372032559808513 {
		fmt.Printf("sub_int64 9223372036854775807%s-4294967296 = %d, wanted -9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(-4294967296); got != 9223372032559808513 {
		fmt.Printf("sub_int64 -4294967296%s9223372036854775807 = %d, wanted 9223372032559808513\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("sub_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("sub_int64 -1%s9223372036854775807 = %d, wanted -9223372036854775808\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(0); got != 9223372036854775807 {
		fmt.Printf("sub_int64 9223372036854775807%s0 = %d, wanted 9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(0); got != -9223372036854775807 {
		fmt.Printf("sub_int64 0%s9223372036854775807 = %d, wanted -9223372036854775807\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(1); got != 9223372036854775806 {
		fmt.Printf("sub_int64 9223372036854775807%s1 = %d, wanted 9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(1); got != -9223372036854775806 {
		fmt.Printf("sub_int64 1%s9223372036854775807 = %d, wanted -9223372036854775806\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(4294967296); got != 9223372032559808511 {
		fmt.Printf("sub_int64 9223372036854775807%s4294967296 = %d, wanted 9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(4294967296); got != -9223372032559808511 {
		fmt.Printf("sub_int64 4294967296%s9223372036854775807 = %d, wanted -9223372032559808511\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(9223372036854775806); got != 1 {
		fmt.Printf("sub_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(9223372036854775806); got != -1 {
		fmt.Printf("sub_int64 9223372036854775806%s9223372036854775807 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_9223372036854775807_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("sub_int64 9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int64_9223372036854775807_ssa(9223372036854775807); got != 0 {
		fmt.Printf("sub_int64 9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("div_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("div_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("div_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(-4294967296); got != 2147483648 {
		fmt.Printf("div_int64 -9223372036854775808%s-4294967296 = %d, wanted 2147483648\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 -4294967296%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("div_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775808\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(1); got != -9223372036854775808 {
		fmt.Printf("div_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775808\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(4294967296); got != -2147483648 {
		fmt.Printf("div_int64 -9223372036854775808%s4294967296 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 4294967296%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(9223372036854775806); got != -1 {
		fmt.Printf("div_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775808_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("div_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775808_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 9223372036854775807%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("div_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("div_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("div_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(-4294967296); got != 2147483647 {
		fmt.Printf("div_int64 -9223372036854775807%s-4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 -4294967296%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("div_int64 -9223372036854775807%s-1 = %d, wanted 9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(1); got != -9223372036854775807 {
		fmt.Printf("div_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(4294967296); got != -2147483647 {
		fmt.Printf("div_int64 -9223372036854775807%s4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 4294967296%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(9223372036854775806); got != -1 {
		fmt.Printf("div_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg9223372036854775807_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("div_int64 -9223372036854775807%s9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg9223372036854775807_ssa(9223372036854775807); got != -1 {
		fmt.Printf("div_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 -4294967296%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(-9223372036854775808); got != 2147483648 {
		fmt.Printf("div_int64 -9223372036854775808%s-4294967296 = %d, wanted 2147483648\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 -4294967296%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(-9223372036854775807); got != 2147483647 {
		fmt.Printf("div_int64 -9223372036854775807%s-4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(-4294967296); got != 1 {
		fmt.Printf("div_int64 -4294967296%s-4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(-4294967296); got != 1 {
		fmt.Printf("div_int64 -4294967296%s-4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(-1); got != 4294967296 {
		fmt.Printf("div_int64 -4294967296%s-1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(1); got != -4294967296 {
		fmt.Printf("div_int64 -4294967296%s1 = %d, wanted -4294967296\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(4294967296); got != -1 {
		fmt.Printf("div_int64 -4294967296%s4294967296 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(4294967296); got != -1 {
		fmt.Printf("div_int64 4294967296%s-4294967296 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 -4294967296%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(9223372036854775806); got != -2147483647 {
		fmt.Printf("div_int64 9223372036854775806%s-4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_Neg4294967296_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 -4294967296%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg4294967296_ssa(9223372036854775807); got != -2147483647 {
		fmt.Printf("div_int64 9223372036854775807%s-4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 -1%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("div_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775808\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 -1%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(-9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("div_int64 -9223372036854775807%s-1 = %d, wanted 9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 -1%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("div_int64 -4294967296%s-1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(-1); got != 1 {
		fmt.Printf("div_int64 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(-1); got != 1 {
		fmt.Printf("div_int64 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(1); got != -1 {
		fmt.Printf("div_int64 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(1); got != -1 {
		fmt.Printf("div_int64 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 -1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(4294967296); got != -4294967296 {
		fmt.Printf("div_int64 4294967296%s-1 = %d, wanted -4294967296\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 -1%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("div_int64 9223372036854775806%s-1 = %d, wanted -9223372036854775806\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 -1%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_Neg1_ssa(9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("div_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 0%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 0%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 0%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(-1); got != 0 {
		fmt.Printf("div_int64 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(1); got != 0 {
		fmt.Printf("div_int64 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 0%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 0%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 0%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 1%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("div_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775808\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 1%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("div_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 1%s-4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("div_int64 -4294967296%s1 = %d, wanted -4294967296\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(-1); got != -1 {
		fmt.Printf("div_int64 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(-1); got != -1 {
		fmt.Printf("div_int64 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(1); got != 1 {
		fmt.Printf("div_int64 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(1); got != 1 {
		fmt.Printf("div_int64 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(4294967296); got != 4294967296 {
		fmt.Printf("div_int64 4294967296%s1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 1%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("div_int64 9223372036854775806%s1 = %d, wanted 9223372036854775806\n", `/`, got)
		failed = true
	}

	if got := div_1_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 1%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_1_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("div_int64 9223372036854775807%s1 = %d, wanted 9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 4294967296%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(-9223372036854775808); got != -2147483648 {
		fmt.Printf("div_int64 -9223372036854775808%s4294967296 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 4294967296%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(-9223372036854775807); got != -2147483647 {
		fmt.Printf("div_int64 -9223372036854775807%s4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(-4294967296); got != -1 {
		fmt.Printf("div_int64 4294967296%s-4294967296 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(-4294967296); got != -1 {
		fmt.Printf("div_int64 -4294967296%s4294967296 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(-1); got != -4294967296 {
		fmt.Printf("div_int64 4294967296%s-1 = %d, wanted -4294967296\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(1); got != 4294967296 {
		fmt.Printf("div_int64 4294967296%s1 = %d, wanted 4294967296\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s4294967296 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(4294967296); got != 1 {
		fmt.Printf("div_int64 4294967296%s4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(4294967296); got != 1 {
		fmt.Printf("div_int64 4294967296%s4294967296 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 4294967296%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(9223372036854775806); got != 2147483647 {
		fmt.Printf("div_int64 9223372036854775806%s4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_4294967296_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 4294967296%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_4294967296_ssa(9223372036854775807); got != 2147483647 {
		fmt.Printf("div_int64 9223372036854775807%s4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("div_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("div_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(-4294967296); got != -2147483647 {
		fmt.Printf("div_int64 9223372036854775806%s-4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 -4294967296%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(-1); got != -9223372036854775806 {
		fmt.Printf("div_int64 9223372036854775806%s-1 = %d, wanted -9223372036854775806\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(1); got != 9223372036854775806 {
		fmt.Printf("div_int64 9223372036854775806%s1 = %d, wanted 9223372036854775806\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(4294967296); got != 2147483647 {
		fmt.Printf("div_int64 9223372036854775806%s4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 4294967296%s9223372036854775806 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(9223372036854775806); got != 1 {
		fmt.Printf("div_int64 9223372036854775806%s9223372036854775806 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(9223372036854775806); got != 1 {
		fmt.Printf("div_int64 9223372036854775806%s9223372036854775806 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775806_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775806_ssa(9223372036854775807); got != 1 {
		fmt.Printf("div_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("div_int64 9223372036854775807%s-9223372036854775808 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("div_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("div_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("div_int64 -9223372036854775807%s9223372036854775807 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(-4294967296); got != -2147483647 {
		fmt.Printf("div_int64 9223372036854775807%s-4294967296 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(-4294967296); got != 0 {
		fmt.Printf("div_int64 -4294967296%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(-1); got != -9223372036854775807 {
		fmt.Printf("div_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(-1); got != 0 {
		fmt.Printf("div_int64 -1%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("div_int64 0%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(1); got != 9223372036854775807 {
		fmt.Printf("div_int64 9223372036854775807%s1 = %d, wanted 9223372036854775807\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(1); got != 0 {
		fmt.Printf("div_int64 1%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(4294967296); got != 2147483647 {
		fmt.Printf("div_int64 9223372036854775807%s4294967296 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(4294967296); got != 0 {
		fmt.Printf("div_int64 4294967296%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(9223372036854775806); got != 1 {
		fmt.Printf("div_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(9223372036854775806); got != 0 {
		fmt.Printf("div_int64 9223372036854775806%s9223372036854775807 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_9223372036854775807_int64_ssa(9223372036854775807); got != 1 {
		fmt.Printf("div_int64 9223372036854775807%s9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int64_9223372036854775807_ssa(9223372036854775807); got != 1 {
		fmt.Printf("div_int64 9223372036854775807%s9223372036854775807 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(-9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(-1); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -1%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(1); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(1); got != -9223372036854775808 {
		fmt.Printf("mul_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s9223372036854775806 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mul_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775808_int64_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775808_ssa(9223372036854775807); got != -9223372036854775808 {
		fmt.Printf("mul_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("mul_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("mul_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 -9223372036854775807%s-4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 -4294967296%s-9223372036854775807 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("mul_int64 -9223372036854775807%s-1 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(-1); got != 9223372036854775807 {
		fmt.Printf("mul_int64 -1%s-9223372036854775807 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775807%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s-9223372036854775807 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(1); got != -9223372036854775807 {
		fmt.Printf("mul_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(1); got != -9223372036854775807 {
		fmt.Printf("mul_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 -9223372036854775807%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 4294967296%s-9223372036854775807 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mul_int64 -9223372036854775807%s9223372036854775806 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_Neg9223372036854775807_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("mul_int64 -9223372036854775807%s9223372036854775807 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg9223372036854775807_ssa(9223372036854775807); got != -1 {
		fmt.Printf("mul_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(-9223372036854775807); got != -4294967296 {
		fmt.Printf("mul_int64 -4294967296%s-9223372036854775807 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(-9223372036854775807); got != -4294967296 {
		fmt.Printf("mul_int64 -9223372036854775807%s-4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(-1); got != 4294967296 {
		fmt.Printf("mul_int64 -4294967296%s-1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(-1); got != 4294967296 {
		fmt.Printf("mul_int64 -1%s-4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(1); got != -4294967296 {
		fmt.Printf("mul_int64 -4294967296%s1 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(1); got != -4294967296 {
		fmt.Printf("mul_int64 1%s-4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(9223372036854775806); got != 8589934592 {
		fmt.Printf("mul_int64 -4294967296%s9223372036854775806 = %d, wanted 8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(9223372036854775806); got != 8589934592 {
		fmt.Printf("mul_int64 9223372036854775806%s-4294967296 = %d, wanted 8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_Neg4294967296_int64_ssa(9223372036854775807); got != 4294967296 {
		fmt.Printf("mul_int64 -4294967296%s9223372036854775807 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg4294967296_ssa(9223372036854775807); got != 4294967296 {
		fmt.Printf("mul_int64 9223372036854775807%s-4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -1%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s-1 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(-9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("mul_int64 -1%s-9223372036854775807 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(-9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("mul_int64 -9223372036854775807%s-1 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 -1%s-4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 -4294967296%s-1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(-1); got != 1 {
		fmt.Printf("mul_int64 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(-1); got != 1 {
		fmt.Printf("mul_int64 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(1); got != -1 {
		fmt.Printf("mul_int64 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(1); got != -1 {
		fmt.Printf("mul_int64 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 -1%s4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 4294967296%s-1 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("mul_int64 -1%s9223372036854775806 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s-1 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int64_ssa(9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("mul_int64 -1%s9223372036854775807 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_Neg1_ssa(9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("mul_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 0%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mul_int64 0%s-9223372036854775807 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775807%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 0%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(-1); got != 0 {
		fmt.Printf("mul_int64 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(-1); got != 0 {
		fmt.Printf("mul_int64 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(1); got != 0 {
		fmt.Printf("mul_int64 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(1); got != 0 {
		fmt.Printf("mul_int64 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 0%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mul_int64 0%s9223372036854775806 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mul_int64 9223372036854775806%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mul_int64 0%s9223372036854775807 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_0_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mul_int64 9223372036854775807%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 1%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s1 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("mul_int64 1%s-9223372036854775807 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("mul_int64 -9223372036854775807%s1 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 1%s-4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 -4294967296%s1 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(-1); got != -1 {
		fmt.Printf("mul_int64 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(-1); got != -1 {
		fmt.Printf("mul_int64 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(1); got != 1 {
		fmt.Printf("mul_int64 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(1); got != 1 {
		fmt.Printf("mul_int64 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 1%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 4294967296%s1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mul_int64 1%s9223372036854775806 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s1 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_1_int64_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("mul_int64 1%s9223372036854775807 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_1_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("mul_int64 9223372036854775807%s1 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 4294967296%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(-9223372036854775807); got != 4294967296 {
		fmt.Printf("mul_int64 4294967296%s-9223372036854775807 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(-9223372036854775807); got != 4294967296 {
		fmt.Printf("mul_int64 -9223372036854775807%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s-4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("mul_int64 -4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(-1); got != -4294967296 {
		fmt.Printf("mul_int64 4294967296%s-1 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(-1); got != -4294967296 {
		fmt.Printf("mul_int64 -1%s4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 4294967296%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(1); got != 4294967296 {
		fmt.Printf("mul_int64 4294967296%s1 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(1); got != 4294967296 {
		fmt.Printf("mul_int64 1%s4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mul_int64 4294967296%s4294967296 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(9223372036854775806); got != -8589934592 {
		fmt.Printf("mul_int64 4294967296%s9223372036854775806 = %d, wanted -8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(9223372036854775806); got != -8589934592 {
		fmt.Printf("mul_int64 9223372036854775806%s4294967296 = %d, wanted -8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_4294967296_int64_ssa(9223372036854775807); got != -4294967296 {
		fmt.Printf("mul_int64 4294967296%s9223372036854775807 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_4294967296_ssa(9223372036854775807); got != -4294967296 {
		fmt.Printf("mul_int64 9223372036854775807%s4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mul_int64 -9223372036854775808%s9223372036854775806 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(-9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(-9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("mul_int64 -9223372036854775807%s9223372036854775806 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(-4294967296); got != 8589934592 {
		fmt.Printf("mul_int64 9223372036854775806%s-4294967296 = %d, wanted 8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(-4294967296); got != 8589934592 {
		fmt.Printf("mul_int64 -4294967296%s9223372036854775806 = %d, wanted 8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(-1); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s-1 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(-1); got != -9223372036854775806 {
		fmt.Printf("mul_int64 -1%s9223372036854775806 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 9223372036854775806%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s9223372036854775806 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(1); got != 9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s1 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(1); got != 9223372036854775806 {
		fmt.Printf("mul_int64 1%s9223372036854775806 = %d, wanted 9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(4294967296); got != -8589934592 {
		fmt.Printf("mul_int64 9223372036854775806%s4294967296 = %d, wanted -8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(4294967296); got != -8589934592 {
		fmt.Printf("mul_int64 4294967296%s9223372036854775806 = %d, wanted -8589934592\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(9223372036854775806); got != 4 {
		fmt.Printf("mul_int64 9223372036854775806%s9223372036854775806 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(9223372036854775806); got != 4 {
		fmt.Printf("mul_int64 9223372036854775806%s9223372036854775806 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775806_int64_ssa(9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s9223372036854775807 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775806_ssa(9223372036854775807); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775807%s9223372036854775806 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(-9223372036854775808); got != -9223372036854775808 {
		fmt.Printf("mul_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -9223372036854775808\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("mul_int64 9223372036854775807%s-9223372036854775807 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("mul_int64 -9223372036854775807%s9223372036854775807 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 9223372036854775807%s-4294967296 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(-4294967296); got != 4294967296 {
		fmt.Printf("mul_int64 -4294967296%s9223372036854775807 = %d, wanted 4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(-1); got != -9223372036854775807 {
		fmt.Printf("mul_int64 9223372036854775807%s-1 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(-1); got != -9223372036854775807 {
		fmt.Printf("mul_int64 -1%s9223372036854775807 = %d, wanted -9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(0); got != 0 {
		fmt.Printf("mul_int64 9223372036854775807%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("mul_int64 0%s9223372036854775807 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(1); got != 9223372036854775807 {
		fmt.Printf("mul_int64 9223372036854775807%s1 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(1); got != 9223372036854775807 {
		fmt.Printf("mul_int64 1%s9223372036854775807 = %d, wanted 9223372036854775807\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 9223372036854775807%s4294967296 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(4294967296); got != -4294967296 {
		fmt.Printf("mul_int64 4294967296%s9223372036854775807 = %d, wanted -4294967296\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775807%s9223372036854775806 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(9223372036854775806); got != -9223372036854775806 {
		fmt.Printf("mul_int64 9223372036854775806%s9223372036854775807 = %d, wanted -9223372036854775806\n", `*`, got)
		failed = true
	}

	if got := mul_9223372036854775807_int64_ssa(9223372036854775807); got != 1 {
		fmt.Printf("mul_int64 9223372036854775807%s9223372036854775807 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int64_9223372036854775807_ssa(9223372036854775807); got != 1 {
		fmt.Printf("mul_int64 9223372036854775807%s9223372036854775807 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-9223372036854775808 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(-9223372036854775807); got != -9223372036854775807 {
		fmt.Printf("mod_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s-9223372036854775808 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s-9223372036854775808 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s-9223372036854775808 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s-9223372036854775808 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s-9223372036854775808 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(9223372036854775806); got != -2 {
		fmt.Printf("mod_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775808_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775808_ssa(9223372036854775807); got != 9223372036854775807 {
		fmt.Printf("mod_int64 9223372036854775807%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(-9223372036854775808); got != -9223372036854775807 {
		fmt.Printf("mod_int64 -9223372036854775807%s-9223372036854775808 = %d, wanted -9223372036854775807\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775808%s-9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(-4294967296); got != -4294967295 {
		fmt.Printf("mod_int64 -9223372036854775807%s-4294967296 = %d, wanted -4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s-9223372036854775807 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s-9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s-9223372036854775807 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(4294967296); got != -4294967295 {
		fmt.Printf("mod_int64 -9223372036854775807%s4294967296 = %d, wanted -4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s-9223372036854775807 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(9223372036854775806); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_Neg9223372036854775807_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg9223372036854775807_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(-9223372036854775808); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s-9223372036854775808 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(-9223372036854775807); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s-9223372036854775807 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(-9223372036854775807); got != -4294967295 {
		fmt.Printf("mod_int64 -9223372036854775807%s-4294967296 = %d, wanted -4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s-4294967296 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s-4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(9223372036854775806); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s9223372036854775806 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(9223372036854775806); got != 4294967294 {
		fmt.Printf("mod_int64 9223372036854775806%s-4294967296 = %d, wanted 4294967294\n", `%`, got)
		failed = true
	}

	if got := mod_Neg4294967296_int64_ssa(9223372036854775807); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s9223372036854775807 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg4294967296_ssa(9223372036854775807); got != 4294967295 {
		fmt.Printf("mod_int64 9223372036854775807%s-4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("mod_int64 -1%s-9223372036854775808 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("mod_int64 -1%s-9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(-4294967296); got != -1 {
		fmt.Printf("mod_int64 -1%s-4294967296 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(1); got != 0 {
		fmt.Printf("mod_int64 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(4294967296); got != -1 {
		fmt.Printf("mod_int64 -1%s4294967296 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(9223372036854775806); got != -1 {
		fmt.Printf("mod_int64 -1%s9223372036854775806 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int64_ssa(9223372036854775807); got != -1 {
		fmt.Printf("mod_int64 -1%s9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_Neg1_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 0%s-9223372036854775808 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 0%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 0%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 0%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mod_int64 0%s9223372036854775806 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 0%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(-9223372036854775808); got != 1 {
		fmt.Printf("mod_int64 1%s-9223372036854775808 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(-9223372036854775807); got != 1 {
		fmt.Printf("mod_int64 1%s-9223372036854775807 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(-4294967296); got != 1 {
		fmt.Printf("mod_int64 1%s-4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(1); got != 0 {
		fmt.Printf("mod_int64 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(4294967296); got != 1 {
		fmt.Printf("mod_int64 1%s4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(9223372036854775806); got != 1 {
		fmt.Printf("mod_int64 1%s9223372036854775806 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int64_ssa(9223372036854775807); got != 1 {
		fmt.Printf("mod_int64 1%s9223372036854775807 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_1_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(-9223372036854775808); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s-9223372036854775808 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(-9223372036854775808); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775808%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(-9223372036854775807); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s-9223372036854775807 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(-9223372036854775807); got != -4294967295 {
		fmt.Printf("mod_int64 -9223372036854775807%s4294967296 = %d, wanted -4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s-4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(-4294967296); got != 0 {
		fmt.Printf("mod_int64 -4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 4294967296%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s4294967296 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 4294967296%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s4294967296 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(4294967296); got != 0 {
		fmt.Printf("mod_int64 4294967296%s4294967296 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(9223372036854775806); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s9223372036854775806 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(9223372036854775806); got != 4294967294 {
		fmt.Printf("mod_int64 9223372036854775806%s4294967296 = %d, wanted 4294967294\n", `%`, got)
		failed = true
	}

	if got := mod_4294967296_int64_ssa(9223372036854775807); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s9223372036854775807 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_int64_4294967296_ssa(9223372036854775807); got != 4294967295 {
		fmt.Printf("mod_int64 9223372036854775807%s4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(-9223372036854775808); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s-9223372036854775808 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(-9223372036854775808); got != -2 {
		fmt.Printf("mod_int64 -9223372036854775808%s9223372036854775806 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(-9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s-9223372036854775807 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(-9223372036854775807); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775807%s9223372036854775806 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(-4294967296); got != 4294967294 {
		fmt.Printf("mod_int64 9223372036854775806%s-4294967296 = %d, wanted 4294967294\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s9223372036854775806 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s9223372036854775806 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s9223372036854775806 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s9223372036854775806 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(4294967296); got != 4294967294 {
		fmt.Printf("mod_int64 9223372036854775806%s4294967296 = %d, wanted 4294967294\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s9223372036854775806 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s9223372036854775806 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(9223372036854775806); got != 0 {
		fmt.Printf("mod_int64 9223372036854775806%s9223372036854775806 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775806_int64_ssa(9223372036854775807); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s9223372036854775807 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775806_ssa(9223372036854775807); got != 1 {
		fmt.Printf("mod_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(-9223372036854775808); got != 9223372036854775807 {
		fmt.Printf("mod_int64 9223372036854775807%s-9223372036854775808 = %d, wanted 9223372036854775807\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(-9223372036854775808); got != -1 {
		fmt.Printf("mod_int64 -9223372036854775808%s9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s-9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(-9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 -9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(-4294967296); got != 4294967295 {
		fmt.Printf("mod_int64 9223372036854775807%s-4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(-4294967296); got != -4294967296 {
		fmt.Printf("mod_int64 -4294967296%s9223372036854775807 = %d, wanted -4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(-1); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(-1); got != -1 {
		fmt.Printf("mod_int64 -1%s9223372036854775807 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(0); got != 0 {
		fmt.Printf("mod_int64 0%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(1); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(1); got != 1 {
		fmt.Printf("mod_int64 1%s9223372036854775807 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(4294967296); got != 4294967295 {
		fmt.Printf("mod_int64 9223372036854775807%s4294967296 = %d, wanted 4294967295\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(4294967296); got != 4294967296 {
		fmt.Printf("mod_int64 4294967296%s9223372036854775807 = %d, wanted 4294967296\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(9223372036854775806); got != 1 {
		fmt.Printf("mod_int64 9223372036854775807%s9223372036854775806 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(9223372036854775806); got != 9223372036854775806 {
		fmt.Printf("mod_int64 9223372036854775806%s9223372036854775807 = %d, wanted 9223372036854775806\n", `%`, got)
		failed = true
	}

	if got := mod_9223372036854775807_int64_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int64_9223372036854775807_ssa(9223372036854775807); got != 0 {
		fmt.Printf("mod_int64 9223372036854775807%s9223372036854775807 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_0_uint32_ssa(0); got != 0 {
		fmt.Printf("add_uint32 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint32_0_ssa(0); got != 0 {
		fmt.Printf("add_uint32 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_uint32_ssa(1); got != 1 {
		fmt.Printf("add_uint32 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint32_0_ssa(1); got != 1 {
		fmt.Printf("add_uint32 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_uint32_ssa(4294967295); got != 4294967295 {
		fmt.Printf("add_uint32 0%s4294967295 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_uint32_0_ssa(4294967295); got != 4294967295 {
		fmt.Printf("add_uint32 4294967295%s0 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_1_uint32_ssa(0); got != 1 {
		fmt.Printf("add_uint32 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint32_1_ssa(0); got != 1 {
		fmt.Printf("add_uint32 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_uint32_ssa(1); got != 2 {
		fmt.Printf("add_uint32 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_uint32_1_ssa(1); got != 2 {
		fmt.Printf("add_uint32 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("add_uint32 1%s4294967295 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint32_1_ssa(4294967295); got != 0 {
		fmt.Printf("add_uint32 4294967295%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_4294967295_uint32_ssa(0); got != 4294967295 {
		fmt.Printf("add_uint32 4294967295%s0 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_uint32_4294967295_ssa(0); got != 4294967295 {
		fmt.Printf("add_uint32 0%s4294967295 = %d, wanted 4294967295\n", `+`, got)
		failed = true
	}

	if got := add_4294967295_uint32_ssa(1); got != 0 {
		fmt.Printf("add_uint32 4294967295%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint32_4294967295_ssa(1); got != 0 {
		fmt.Printf("add_uint32 1%s4294967295 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_4294967295_uint32_ssa(4294967295); got != 4294967294 {
		fmt.Printf("add_uint32 4294967295%s4294967295 = %d, wanted 4294967294\n", `+`, got)
		failed = true
	}

	if got := add_uint32_4294967295_ssa(4294967295); got != 4294967294 {
		fmt.Printf("add_uint32 4294967295%s4294967295 = %d, wanted 4294967294\n", `+`, got)
		failed = true
	}

	if got := sub_0_uint32_ssa(0); got != 0 {
		fmt.Printf("sub_uint32 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_0_ssa(0); got != 0 {
		fmt.Printf("sub_uint32 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint32_ssa(1); got != 4294967295 {
		fmt.Printf("sub_uint32 0%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_0_ssa(1); got != 1 {
		fmt.Printf("sub_uint32 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint32_ssa(4294967295); got != 1 {
		fmt.Printf("sub_uint32 0%s4294967295 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_0_ssa(4294967295); got != 4294967295 {
		fmt.Printf("sub_uint32 4294967295%s0 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint32_ssa(0); got != 1 {
		fmt.Printf("sub_uint32 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_1_ssa(0); got != 4294967295 {
		fmt.Printf("sub_uint32 0%s1 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint32_ssa(1); got != 0 {
		fmt.Printf("sub_uint32 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_1_ssa(1); got != 0 {
		fmt.Printf("sub_uint32 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint32_ssa(4294967295); got != 2 {
		fmt.Printf("sub_uint32 1%s4294967295 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_1_ssa(4294967295); got != 4294967294 {
		fmt.Printf("sub_uint32 4294967295%s1 = %d, wanted 4294967294\n", `-`, got)
		failed = true
	}

	if got := sub_4294967295_uint32_ssa(0); got != 4294967295 {
		fmt.Printf("sub_uint32 4294967295%s0 = %d, wanted 4294967295\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_4294967295_ssa(0); got != 1 {
		fmt.Printf("sub_uint32 0%s4294967295 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_4294967295_uint32_ssa(1); got != 4294967294 {
		fmt.Printf("sub_uint32 4294967295%s1 = %d, wanted 4294967294\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_4294967295_ssa(1); got != 2 {
		fmt.Printf("sub_uint32 1%s4294967295 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_4294967295_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("sub_uint32 4294967295%s4294967295 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint32_4294967295_ssa(4294967295); got != 0 {
		fmt.Printf("sub_uint32 4294967295%s4294967295 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_0_uint32_ssa(1); got != 0 {
		fmt.Printf("div_uint32 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("div_uint32 0%s4294967295 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint32_1_ssa(0); got != 0 {
		fmt.Printf("div_uint32 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_uint32_ssa(1); got != 1 {
		fmt.Printf("div_uint32 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint32_1_ssa(1); got != 1 {
		fmt.Printf("div_uint32 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("div_uint32 1%s4294967295 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint32_1_ssa(4294967295); got != 4294967295 {
		fmt.Printf("div_uint32 4294967295%s1 = %d, wanted 4294967295\n", `/`, got)
		failed = true
	}

	if got := div_uint32_4294967295_ssa(0); got != 0 {
		fmt.Printf("div_uint32 0%s4294967295 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967295_uint32_ssa(1); got != 4294967295 {
		fmt.Printf("div_uint32 4294967295%s1 = %d, wanted 4294967295\n", `/`, got)
		failed = true
	}

	if got := div_uint32_4294967295_ssa(1); got != 0 {
		fmt.Printf("div_uint32 1%s4294967295 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_4294967295_uint32_ssa(4294967295); got != 1 {
		fmt.Printf("div_uint32 4294967295%s4294967295 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint32_4294967295_ssa(4294967295); got != 1 {
		fmt.Printf("div_uint32 4294967295%s4294967295 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_0_uint32_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_0_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint32_ssa(1); got != 0 {
		fmt.Printf("mul_uint32 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_0_ssa(1); got != 0 {
		fmt.Printf("mul_uint32 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("mul_uint32 0%s4294967295 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_0_ssa(4294967295); got != 0 {
		fmt.Printf("mul_uint32 4294967295%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint32_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_1_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint32_ssa(1); got != 1 {
		fmt.Printf("mul_uint32 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_1_ssa(1); got != 1 {
		fmt.Printf("mul_uint32 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint32_ssa(4294967295); got != 4294967295 {
		fmt.Printf("mul_uint32 1%s4294967295 = %d, wanted 4294967295\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_1_ssa(4294967295); got != 4294967295 {
		fmt.Printf("mul_uint32 4294967295%s1 = %d, wanted 4294967295\n", `*`, got)
		failed = true
	}

	if got := mul_4294967295_uint32_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 4294967295%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_4294967295_ssa(0); got != 0 {
		fmt.Printf("mul_uint32 0%s4294967295 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_4294967295_uint32_ssa(1); got != 4294967295 {
		fmt.Printf("mul_uint32 4294967295%s1 = %d, wanted 4294967295\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_4294967295_ssa(1); got != 4294967295 {
		fmt.Printf("mul_uint32 1%s4294967295 = %d, wanted 4294967295\n", `*`, got)
		failed = true
	}

	if got := mul_4294967295_uint32_ssa(4294967295); got != 1 {
		fmt.Printf("mul_uint32 4294967295%s4294967295 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint32_4294967295_ssa(4294967295); got != 1 {
		fmt.Printf("mul_uint32 4294967295%s4294967295 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := lsh_0_uint32_ssa(0); got != 0 {
		fmt.Printf("lsh_uint32 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_0_ssa(0); got != 0 {
		fmt.Printf("lsh_uint32 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint32_ssa(1); got != 0 {
		fmt.Printf("lsh_uint32 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_0_ssa(1); got != 1 {
		fmt.Printf("lsh_uint32 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("lsh_uint32 0%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_0_ssa(4294967295); got != 4294967295 {
		fmt.Printf("lsh_uint32 4294967295%s0 = %d, wanted 4294967295\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint32_ssa(0); got != 1 {
		fmt.Printf("lsh_uint32 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_1_ssa(0); got != 0 {
		fmt.Printf("lsh_uint32 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint32_ssa(1); got != 2 {
		fmt.Printf("lsh_uint32 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_1_ssa(1); got != 2 {
		fmt.Printf("lsh_uint32 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("lsh_uint32 1%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_1_ssa(4294967295); got != 4294967294 {
		fmt.Printf("lsh_uint32 4294967295%s1 = %d, wanted 4294967294\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967295_uint32_ssa(0); got != 4294967295 {
		fmt.Printf("lsh_uint32 4294967295%s0 = %d, wanted 4294967295\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_4294967295_ssa(0); got != 0 {
		fmt.Printf("lsh_uint32 0%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967295_uint32_ssa(1); got != 4294967294 {
		fmt.Printf("lsh_uint32 4294967295%s1 = %d, wanted 4294967294\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_4294967295_ssa(1); got != 0 {
		fmt.Printf("lsh_uint32 1%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_4294967295_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("lsh_uint32 4294967295%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint32_4294967295_ssa(4294967295); got != 0 {
		fmt.Printf("lsh_uint32 4294967295%s4294967295 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := rsh_0_uint32_ssa(0); got != 0 {
		fmt.Printf("rsh_uint32 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_0_ssa(0); got != 0 {
		fmt.Printf("rsh_uint32 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint32_ssa(1); got != 0 {
		fmt.Printf("rsh_uint32 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_0_ssa(1); got != 1 {
		fmt.Printf("rsh_uint32 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("rsh_uint32 0%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_0_ssa(4294967295); got != 4294967295 {
		fmt.Printf("rsh_uint32 4294967295%s0 = %d, wanted 4294967295\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint32_ssa(0); got != 1 {
		fmt.Printf("rsh_uint32 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_1_ssa(0); got != 0 {
		fmt.Printf("rsh_uint32 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint32_ssa(1); got != 0 {
		fmt.Printf("rsh_uint32 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_1_ssa(1); got != 0 {
		fmt.Printf("rsh_uint32 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("rsh_uint32 1%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_1_ssa(4294967295); got != 2147483647 {
		fmt.Printf("rsh_uint32 4294967295%s1 = %d, wanted 2147483647\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967295_uint32_ssa(0); got != 4294967295 {
		fmt.Printf("rsh_uint32 4294967295%s0 = %d, wanted 4294967295\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_4294967295_ssa(0); got != 0 {
		fmt.Printf("rsh_uint32 0%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967295_uint32_ssa(1); got != 2147483647 {
		fmt.Printf("rsh_uint32 4294967295%s1 = %d, wanted 2147483647\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_4294967295_ssa(1); got != 0 {
		fmt.Printf("rsh_uint32 1%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_4294967295_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("rsh_uint32 4294967295%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint32_4294967295_ssa(4294967295); got != 0 {
		fmt.Printf("rsh_uint32 4294967295%s4294967295 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := mod_0_uint32_ssa(1); got != 0 {
		fmt.Printf("mod_uint32 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("mod_uint32 0%s4294967295 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_1_ssa(0); got != 0 {
		fmt.Printf("mod_uint32 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint32_ssa(1); got != 0 {
		fmt.Printf("mod_uint32 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_1_ssa(1); got != 0 {
		fmt.Printf("mod_uint32 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint32_ssa(4294967295); got != 1 {
		fmt.Printf("mod_uint32 1%s4294967295 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_1_ssa(4294967295); got != 0 {
		fmt.Printf("mod_uint32 4294967295%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_4294967295_ssa(0); got != 0 {
		fmt.Printf("mod_uint32 0%s4294967295 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_4294967295_uint32_ssa(1); got != 0 {
		fmt.Printf("mod_uint32 4294967295%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_4294967295_ssa(1); got != 1 {
		fmt.Printf("mod_uint32 1%s4294967295 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_4294967295_uint32_ssa(4294967295); got != 0 {
		fmt.Printf("mod_uint32 4294967295%s4294967295 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint32_4294967295_ssa(4294967295); got != 0 {
		fmt.Printf("mod_uint32 4294967295%s4294967295 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("add_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(-2147483648); got != 0 {
		fmt.Printf("add_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(-2147483647); got != 1 {
		fmt.Printf("add_int32 -2147483648%s-2147483647 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(-2147483647); got != 1 {
		fmt.Printf("add_int32 -2147483647%s-2147483648 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(-1); got != 2147483647 {
		fmt.Printf("add_int32 -2147483648%s-1 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(-1); got != 2147483647 {
		fmt.Printf("add_int32 -1%s-2147483648 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(0); got != -2147483648 {
		fmt.Printf("add_int32 -2147483648%s0 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(0); got != -2147483648 {
		fmt.Printf("add_int32 0%s-2147483648 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(1); got != -2147483647 {
		fmt.Printf("add_int32 -2147483648%s1 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(1); got != -2147483647 {
		fmt.Printf("add_int32 1%s-2147483648 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483648_int32_ssa(2147483647); got != -1 {
		fmt.Printf("add_int32 -2147483648%s2147483647 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483648_ssa(2147483647); got != -1 {
		fmt.Printf("add_int32 2147483647%s-2147483648 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(-2147483648); got != 1 {
		fmt.Printf("add_int32 -2147483647%s-2147483648 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(-2147483648); got != 1 {
		fmt.Printf("add_int32 -2147483648%s-2147483647 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(-2147483647); got != 2 {
		fmt.Printf("add_int32 -2147483647%s-2147483647 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(-2147483647); got != 2 {
		fmt.Printf("add_int32 -2147483647%s-2147483647 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(-1); got != -2147483648 {
		fmt.Printf("add_int32 -2147483647%s-1 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(-1); got != -2147483648 {
		fmt.Printf("add_int32 -1%s-2147483647 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(0); got != -2147483647 {
		fmt.Printf("add_int32 -2147483647%s0 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(0); got != -2147483647 {
		fmt.Printf("add_int32 0%s-2147483647 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(1); got != -2147483646 {
		fmt.Printf("add_int32 -2147483647%s1 = %d, wanted -2147483646\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(1); got != -2147483646 {
		fmt.Printf("add_int32 1%s-2147483647 = %d, wanted -2147483646\n", `+`, got)
		failed = true
	}

	if got := add_Neg2147483647_int32_ssa(2147483647); got != 0 {
		fmt.Printf("add_int32 -2147483647%s2147483647 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg2147483647_ssa(2147483647); got != 0 {
		fmt.Printf("add_int32 2147483647%s-2147483647 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(-2147483648); got != 2147483647 {
		fmt.Printf("add_int32 -1%s-2147483648 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(-2147483648); got != 2147483647 {
		fmt.Printf("add_int32 -2147483648%s-1 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("add_int32 -1%s-2147483647 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("add_int32 -2147483647%s-1 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(-1); got != -2 {
		fmt.Printf("add_int32 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(-1); got != -2 {
		fmt.Printf("add_int32 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(0); got != -1 {
		fmt.Printf("add_int32 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(0); got != -1 {
		fmt.Printf("add_int32 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(1); got != 0 {
		fmt.Printf("add_int32 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(1); got != 0 {
		fmt.Printf("add_int32 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int32_ssa(2147483647); got != 2147483646 {
		fmt.Printf("add_int32 -1%s2147483647 = %d, wanted 2147483646\n", `+`, got)
		failed = true
	}

	if got := add_int32_Neg1_ssa(2147483647); got != 2147483646 {
		fmt.Printf("add_int32 2147483647%s-1 = %d, wanted 2147483646\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("add_int32 0%s-2147483648 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("add_int32 -2147483648%s0 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("add_int32 0%s-2147483647 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("add_int32 -2147483647%s0 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(-1); got != -1 {
		fmt.Printf("add_int32 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(-1); got != -1 {
		fmt.Printf("add_int32 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(0); got != 0 {
		fmt.Printf("add_int32 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(0); got != 0 {
		fmt.Printf("add_int32 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(1); got != 1 {
		fmt.Printf("add_int32 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(1); got != 1 {
		fmt.Printf("add_int32 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_int32_ssa(2147483647); got != 2147483647 {
		fmt.Printf("add_int32 0%s2147483647 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_0_ssa(2147483647); got != 2147483647 {
		fmt.Printf("add_int32 2147483647%s0 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(-2147483648); got != -2147483647 {
		fmt.Printf("add_int32 1%s-2147483648 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(-2147483648); got != -2147483647 {
		fmt.Printf("add_int32 -2147483648%s1 = %d, wanted -2147483647\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(-2147483647); got != -2147483646 {
		fmt.Printf("add_int32 1%s-2147483647 = %d, wanted -2147483646\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(-2147483647); got != -2147483646 {
		fmt.Printf("add_int32 -2147483647%s1 = %d, wanted -2147483646\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(-1); got != 0 {
		fmt.Printf("add_int32 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(-1); got != 0 {
		fmt.Printf("add_int32 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(0); got != 1 {
		fmt.Printf("add_int32 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(0); got != 1 {
		fmt.Printf("add_int32 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(1); got != 2 {
		fmt.Printf("add_int32 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(1); got != 2 {
		fmt.Printf("add_int32 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_int32_ssa(2147483647); got != -2147483648 {
		fmt.Printf("add_int32 1%s2147483647 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_1_ssa(2147483647); got != -2147483648 {
		fmt.Printf("add_int32 2147483647%s1 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(-2147483648); got != -1 {
		fmt.Printf("add_int32 2147483647%s-2147483648 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(-2147483648); got != -1 {
		fmt.Printf("add_int32 -2147483648%s2147483647 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("add_int32 2147483647%s-2147483647 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(-2147483647); got != 0 {
		fmt.Printf("add_int32 -2147483647%s2147483647 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(-1); got != 2147483646 {
		fmt.Printf("add_int32 2147483647%s-1 = %d, wanted 2147483646\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(-1); got != 2147483646 {
		fmt.Printf("add_int32 -1%s2147483647 = %d, wanted 2147483646\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(0); got != 2147483647 {
		fmt.Printf("add_int32 2147483647%s0 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(0); got != 2147483647 {
		fmt.Printf("add_int32 0%s2147483647 = %d, wanted 2147483647\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(1); got != -2147483648 {
		fmt.Printf("add_int32 2147483647%s1 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(1); got != -2147483648 {
		fmt.Printf("add_int32 1%s2147483647 = %d, wanted -2147483648\n", `+`, got)
		failed = true
	}

	if got := add_2147483647_int32_ssa(2147483647); got != -2 {
		fmt.Printf("add_int32 2147483647%s2147483647 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int32_2147483647_ssa(2147483647); got != -2 {
		fmt.Printf("add_int32 2147483647%s2147483647 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("sub_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(-2147483648); got != 0 {
		fmt.Printf("sub_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(-2147483647); got != -1 {
		fmt.Printf("sub_int32 -2147483648%s-2147483647 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(-2147483647); got != 1 {
		fmt.Printf("sub_int32 -2147483647%s-2147483648 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(-1); got != -2147483647 {
		fmt.Printf("sub_int32 -2147483648%s-1 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(-1); got != 2147483647 {
		fmt.Printf("sub_int32 -1%s-2147483648 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(0); got != -2147483648 {
		fmt.Printf("sub_int32 -2147483648%s0 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(0); got != -2147483648 {
		fmt.Printf("sub_int32 0%s-2147483648 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(1); got != 2147483647 {
		fmt.Printf("sub_int32 -2147483648%s1 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(1); got != -2147483647 {
		fmt.Printf("sub_int32 1%s-2147483648 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483648_int32_ssa(2147483647); got != 1 {
		fmt.Printf("sub_int32 -2147483648%s2147483647 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483648_ssa(2147483647); got != -1 {
		fmt.Printf("sub_int32 2147483647%s-2147483648 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(-2147483648); got != 1 {
		fmt.Printf("sub_int32 -2147483647%s-2147483648 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(-2147483648); got != -1 {
		fmt.Printf("sub_int32 -2147483648%s-2147483647 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("sub_int32 -2147483647%s-2147483647 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(-2147483647); got != 0 {
		fmt.Printf("sub_int32 -2147483647%s-2147483647 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(-1); got != -2147483646 {
		fmt.Printf("sub_int32 -2147483647%s-1 = %d, wanted -2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(-1); got != 2147483646 {
		fmt.Printf("sub_int32 -1%s-2147483647 = %d, wanted 2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(0); got != -2147483647 {
		fmt.Printf("sub_int32 -2147483647%s0 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(0); got != 2147483647 {
		fmt.Printf("sub_int32 0%s-2147483647 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(1); got != -2147483648 {
		fmt.Printf("sub_int32 -2147483647%s1 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(1); got != -2147483648 {
		fmt.Printf("sub_int32 1%s-2147483647 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_Neg2147483647_int32_ssa(2147483647); got != 2 {
		fmt.Printf("sub_int32 -2147483647%s2147483647 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg2147483647_ssa(2147483647); got != -2 {
		fmt.Printf("sub_int32 2147483647%s-2147483647 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(-2147483648); got != 2147483647 {
		fmt.Printf("sub_int32 -1%s-2147483648 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(-2147483648); got != -2147483647 {
		fmt.Printf("sub_int32 -2147483648%s-1 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(-2147483647); got != 2147483646 {
		fmt.Printf("sub_int32 -1%s-2147483647 = %d, wanted 2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(-2147483647); got != -2147483646 {
		fmt.Printf("sub_int32 -2147483647%s-1 = %d, wanted -2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(-1); got != 0 {
		fmt.Printf("sub_int32 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(-1); got != 0 {
		fmt.Printf("sub_int32 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(0); got != -1 {
		fmt.Printf("sub_int32 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(0); got != 1 {
		fmt.Printf("sub_int32 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(1); got != -2 {
		fmt.Printf("sub_int32 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(1); got != 2 {
		fmt.Printf("sub_int32 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int32_ssa(2147483647); got != -2147483648 {
		fmt.Printf("sub_int32 -1%s2147483647 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_Neg1_ssa(2147483647); got != -2147483648 {
		fmt.Printf("sub_int32 2147483647%s-1 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("sub_int32 0%s-2147483648 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("sub_int32 -2147483648%s0 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(-2147483647); got != 2147483647 {
		fmt.Printf("sub_int32 0%s-2147483647 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("sub_int32 -2147483647%s0 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(-1); got != 1 {
		fmt.Printf("sub_int32 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(-1); got != -1 {
		fmt.Printf("sub_int32 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(0); got != 0 {
		fmt.Printf("sub_int32 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(0); got != 0 {
		fmt.Printf("sub_int32 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(1); got != -1 {
		fmt.Printf("sub_int32 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(1); got != 1 {
		fmt.Printf("sub_int32 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int32_ssa(2147483647); got != -2147483647 {
		fmt.Printf("sub_int32 0%s2147483647 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_0_ssa(2147483647); got != 2147483647 {
		fmt.Printf("sub_int32 2147483647%s0 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(-2147483648); got != -2147483647 {
		fmt.Printf("sub_int32 1%s-2147483648 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(-2147483648); got != 2147483647 {
		fmt.Printf("sub_int32 -2147483648%s1 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("sub_int32 1%s-2147483647 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("sub_int32 -2147483647%s1 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(-1); got != 2 {
		fmt.Printf("sub_int32 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(-1); got != -2 {
		fmt.Printf("sub_int32 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(0); got != 1 {
		fmt.Printf("sub_int32 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(0); got != -1 {
		fmt.Printf("sub_int32 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(1); got != 0 {
		fmt.Printf("sub_int32 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(1); got != 0 {
		fmt.Printf("sub_int32 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_int32_ssa(2147483647); got != -2147483646 {
		fmt.Printf("sub_int32 1%s2147483647 = %d, wanted -2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_int32_1_ssa(2147483647); got != 2147483646 {
		fmt.Printf("sub_int32 2147483647%s1 = %d, wanted 2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(-2147483648); got != -1 {
		fmt.Printf("sub_int32 2147483647%s-2147483648 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(-2147483648); got != 1 {
		fmt.Printf("sub_int32 -2147483648%s2147483647 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(-2147483647); got != -2 {
		fmt.Printf("sub_int32 2147483647%s-2147483647 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(-2147483647); got != 2 {
		fmt.Printf("sub_int32 -2147483647%s2147483647 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(-1); got != -2147483648 {
		fmt.Printf("sub_int32 2147483647%s-1 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(-1); got != -2147483648 {
		fmt.Printf("sub_int32 -1%s2147483647 = %d, wanted -2147483648\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(0); got != 2147483647 {
		fmt.Printf("sub_int32 2147483647%s0 = %d, wanted 2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(0); got != -2147483647 {
		fmt.Printf("sub_int32 0%s2147483647 = %d, wanted -2147483647\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(1); got != 2147483646 {
		fmt.Printf("sub_int32 2147483647%s1 = %d, wanted 2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(1); got != -2147483646 {
		fmt.Printf("sub_int32 1%s2147483647 = %d, wanted -2147483646\n", `-`, got)
		failed = true
	}

	if got := sub_2147483647_int32_ssa(2147483647); got != 0 {
		fmt.Printf("sub_int32 2147483647%s2147483647 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int32_2147483647_ssa(2147483647); got != 0 {
		fmt.Printf("sub_int32 2147483647%s2147483647 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_Neg2147483648_int32_ssa(-2147483648); got != 1 {
		fmt.Printf("div_int32 -2147483648%s-2147483648 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(-2147483648); got != 1 {
		fmt.Printf("div_int32 -2147483648%s-2147483648 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483648_int32_ssa(-2147483647); got != 1 {
		fmt.Printf("div_int32 -2147483648%s-2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(-2147483647); got != 0 {
		fmt.Printf("div_int32 -2147483647%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483648_int32_ssa(-1); got != -2147483648 {
		fmt.Printf("div_int32 -2147483648%s-1 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(-1); got != 0 {
		fmt.Printf("div_int32 -1%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(0); got != 0 {
		fmt.Printf("div_int32 0%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483648_int32_ssa(1); got != -2147483648 {
		fmt.Printf("div_int32 -2147483648%s1 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(1); got != 0 {
		fmt.Printf("div_int32 1%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483648_int32_ssa(2147483647); got != -1 {
		fmt.Printf("div_int32 -2147483648%s2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483648_ssa(2147483647); got != 0 {
		fmt.Printf("div_int32 2147483647%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483647_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("div_int32 -2147483647%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(-2147483648); got != 1 {
		fmt.Printf("div_int32 -2147483648%s-2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483647_int32_ssa(-2147483647); got != 1 {
		fmt.Printf("div_int32 -2147483647%s-2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(-2147483647); got != 1 {
		fmt.Printf("div_int32 -2147483647%s-2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483647_int32_ssa(-1); got != 2147483647 {
		fmt.Printf("div_int32 -2147483647%s-1 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(-1); got != 0 {
		fmt.Printf("div_int32 -1%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(0); got != 0 {
		fmt.Printf("div_int32 0%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483647_int32_ssa(1); got != -2147483647 {
		fmt.Printf("div_int32 -2147483647%s1 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(1); got != 0 {
		fmt.Printf("div_int32 1%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg2147483647_int32_ssa(2147483647); got != -1 {
		fmt.Printf("div_int32 -2147483647%s2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg2147483647_ssa(2147483647); got != -1 {
		fmt.Printf("div_int32 2147483647%s-2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("div_int32 -1%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("div_int32 -2147483648%s-1 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("div_int32 -1%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(-2147483647); got != 2147483647 {
		fmt.Printf("div_int32 -2147483647%s-1 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int32_ssa(-1); got != 1 {
		fmt.Printf("div_int32 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(-1); got != 1 {
		fmt.Printf("div_int32 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(0); got != 0 {
		fmt.Printf("div_int32 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int32_ssa(1); got != -1 {
		fmt.Printf("div_int32 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(1); got != -1 {
		fmt.Printf("div_int32 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int32_ssa(2147483647); got != 0 {
		fmt.Printf("div_int32 -1%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_Neg1_ssa(2147483647); got != -2147483647 {
		fmt.Printf("div_int32 2147483647%s-1 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_0_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("div_int32 0%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("div_int32 0%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int32_ssa(-1); got != 0 {
		fmt.Printf("div_int32 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int32_ssa(1); got != 0 {
		fmt.Printf("div_int32 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int32_ssa(2147483647); got != 0 {
		fmt.Printf("div_int32 0%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("div_int32 1%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("div_int32 -2147483648%s1 = %d, wanted -2147483648\n", `/`, got)
		failed = true
	}

	if got := div_1_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("div_int32 1%s-2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("div_int32 -2147483647%s1 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_1_int32_ssa(-1); got != -1 {
		fmt.Printf("div_int32 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(-1); got != -1 {
		fmt.Printf("div_int32 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(0); got != 0 {
		fmt.Printf("div_int32 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int32_ssa(1); got != 1 {
		fmt.Printf("div_int32 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(1); got != 1 {
		fmt.Printf("div_int32 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_int32_ssa(2147483647); got != 0 {
		fmt.Printf("div_int32 1%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_1_ssa(2147483647); got != 2147483647 {
		fmt.Printf("div_int32 2147483647%s1 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_2147483647_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("div_int32 2147483647%s-2147483648 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(-2147483648); got != -1 {
		fmt.Printf("div_int32 -2147483648%s2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_2147483647_int32_ssa(-2147483647); got != -1 {
		fmt.Printf("div_int32 2147483647%s-2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(-2147483647); got != -1 {
		fmt.Printf("div_int32 -2147483647%s2147483647 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_2147483647_int32_ssa(-1); got != -2147483647 {
		fmt.Printf("div_int32 2147483647%s-1 = %d, wanted -2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(-1); got != 0 {
		fmt.Printf("div_int32 -1%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(0); got != 0 {
		fmt.Printf("div_int32 0%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_2147483647_int32_ssa(1); got != 2147483647 {
		fmt.Printf("div_int32 2147483647%s1 = %d, wanted 2147483647\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(1); got != 0 {
		fmt.Printf("div_int32 1%s2147483647 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_2147483647_int32_ssa(2147483647); got != 1 {
		fmt.Printf("div_int32 2147483647%s2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int32_2147483647_ssa(2147483647); got != 1 {
		fmt.Printf("div_int32 2147483647%s2147483647 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("mul_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(-2147483648); got != 0 {
		fmt.Printf("mul_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s-2147483647 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(-2147483647); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483647%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(-1); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s-1 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(-1); got != -2147483648 {
		fmt.Printf("mul_int32 -1%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 -2147483648%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s-2147483648 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(1); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s1 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(1); got != -2147483648 {
		fmt.Printf("mul_int32 1%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483648_int32_ssa(2147483647); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s2147483647 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483648_ssa(2147483647); got != -2147483648 {
		fmt.Printf("mul_int32 2147483647%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483647%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s-2147483647 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(-2147483647); got != 1 {
		fmt.Printf("mul_int32 -2147483647%s-2147483647 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(-2147483647); got != 1 {
		fmt.Printf("mul_int32 -2147483647%s-2147483647 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(-1); got != 2147483647 {
		fmt.Printf("mul_int32 -2147483647%s-1 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(-1); got != 2147483647 {
		fmt.Printf("mul_int32 -1%s-2147483647 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 -2147483647%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s-2147483647 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(1); got != -2147483647 {
		fmt.Printf("mul_int32 -2147483647%s1 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(1); got != -2147483647 {
		fmt.Printf("mul_int32 1%s-2147483647 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_Neg2147483647_int32_ssa(2147483647); got != -1 {
		fmt.Printf("mul_int32 -2147483647%s2147483647 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg2147483647_ssa(2147483647); got != -1 {
		fmt.Printf("mul_int32 2147483647%s-2147483647 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -1%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s-1 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(-2147483647); got != 2147483647 {
		fmt.Printf("mul_int32 -1%s-2147483647 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(-2147483647); got != 2147483647 {
		fmt.Printf("mul_int32 -2147483647%s-1 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(-1); got != 1 {
		fmt.Printf("mul_int32 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(-1); got != 1 {
		fmt.Printf("mul_int32 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(1); got != -1 {
		fmt.Printf("mul_int32 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(1); got != -1 {
		fmt.Printf("mul_int32 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int32_ssa(2147483647); got != -2147483647 {
		fmt.Printf("mul_int32 -1%s2147483647 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_Neg1_ssa(2147483647); got != -2147483647 {
		fmt.Printf("mul_int32 2147483647%s-1 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("mul_int32 0%s-2147483648 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(-2147483648); got != 0 {
		fmt.Printf("mul_int32 -2147483648%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("mul_int32 0%s-2147483647 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(-2147483647); got != 0 {
		fmt.Printf("mul_int32 -2147483647%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(-1); got != 0 {
		fmt.Printf("mul_int32 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(-1); got != 0 {
		fmt.Printf("mul_int32 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(1); got != 0 {
		fmt.Printf("mul_int32 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(1); got != 0 {
		fmt.Printf("mul_int32 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int32_ssa(2147483647); got != 0 {
		fmt.Printf("mul_int32 0%s2147483647 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_0_ssa(2147483647); got != 0 {
		fmt.Printf("mul_int32 2147483647%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 1%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s1 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("mul_int32 1%s-2147483647 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("mul_int32 -2147483647%s1 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(-1); got != -1 {
		fmt.Printf("mul_int32 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(-1); got != -1 {
		fmt.Printf("mul_int32 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(1); got != 1 {
		fmt.Printf("mul_int32 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(1); got != 1 {
		fmt.Printf("mul_int32 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int32_ssa(2147483647); got != 2147483647 {
		fmt.Printf("mul_int32 1%s2147483647 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_1_ssa(2147483647); got != 2147483647 {
		fmt.Printf("mul_int32 2147483647%s1 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 2147483647%s-2147483648 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(-2147483648); got != -2147483648 {
		fmt.Printf("mul_int32 -2147483648%s2147483647 = %d, wanted -2147483648\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(-2147483647); got != -1 {
		fmt.Printf("mul_int32 2147483647%s-2147483647 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(-2147483647); got != -1 {
		fmt.Printf("mul_int32 -2147483647%s2147483647 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(-1); got != -2147483647 {
		fmt.Printf("mul_int32 2147483647%s-1 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(-1); got != -2147483647 {
		fmt.Printf("mul_int32 -1%s2147483647 = %d, wanted -2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(0); got != 0 {
		fmt.Printf("mul_int32 2147483647%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(0); got != 0 {
		fmt.Printf("mul_int32 0%s2147483647 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(1); got != 2147483647 {
		fmt.Printf("mul_int32 2147483647%s1 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(1); got != 2147483647 {
		fmt.Printf("mul_int32 1%s2147483647 = %d, wanted 2147483647\n", `*`, got)
		failed = true
	}

	if got := mul_2147483647_int32_ssa(2147483647); got != 1 {
		fmt.Printf("mul_int32 2147483647%s2147483647 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int32_2147483647_ssa(2147483647); got != 1 {
		fmt.Printf("mul_int32 2147483647%s2147483647 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mod_Neg2147483648_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(-2147483648); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s-2147483648 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483648_int32_ssa(-2147483647); got != -1 {
		fmt.Printf("mod_int32 -2147483648%s-2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(-2147483647); got != -2147483647 {
		fmt.Printf("mod_int32 -2147483647%s-2147483648 = %d, wanted -2147483647\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483648_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(-1); got != -1 {
		fmt.Printf("mod_int32 -1%s-2147483648 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(0); got != 0 {
		fmt.Printf("mod_int32 0%s-2147483648 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483648_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(1); got != 1 {
		fmt.Printf("mod_int32 1%s-2147483648 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483648_int32_ssa(2147483647); got != -1 {
		fmt.Printf("mod_int32 -2147483648%s2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483648_ssa(2147483647); got != 2147483647 {
		fmt.Printf("mod_int32 2147483647%s-2147483648 = %d, wanted 2147483647\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483647_int32_ssa(-2147483648); got != -2147483647 {
		fmt.Printf("mod_int32 -2147483647%s-2147483648 = %d, wanted -2147483647\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(-2147483648); got != -1 {
		fmt.Printf("mod_int32 -2147483648%s-2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483647_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483647_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(-1); got != -1 {
		fmt.Printf("mod_int32 -1%s-2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(0); got != 0 {
		fmt.Printf("mod_int32 0%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483647_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(1); got != 1 {
		fmt.Printf("mod_int32 1%s-2147483647 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg2147483647_int32_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg2147483647_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int32_ssa(-2147483648); got != -1 {
		fmt.Printf("mod_int32 -1%s-2147483648 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(-2147483648); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int32_ssa(-2147483647); got != -1 {
		fmt.Printf("mod_int32 -1%s-2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(0); got != 0 {
		fmt.Printf("mod_int32 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(1); got != 0 {
		fmt.Printf("mod_int32 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int32_ssa(2147483647); got != -1 {
		fmt.Printf("mod_int32 -1%s2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_Neg1_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int32_ssa(-2147483648); got != 0 {
		fmt.Printf("mod_int32 0%s-2147483648 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 0%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int32_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 0%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int32_ssa(-2147483648); got != 1 {
		fmt.Printf("mod_int32 1%s-2147483648 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(-2147483648); got != 0 {
		fmt.Printf("mod_int32 -2147483648%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int32_ssa(-2147483647); got != 1 {
		fmt.Printf("mod_int32 1%s-2147483647 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(0); got != 0 {
		fmt.Printf("mod_int32 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(1); got != 0 {
		fmt.Printf("mod_int32 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int32_ssa(2147483647); got != 1 {
		fmt.Printf("mod_int32 1%s2147483647 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_1_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_2147483647_int32_ssa(-2147483648); got != 2147483647 {
		fmt.Printf("mod_int32 2147483647%s-2147483648 = %d, wanted 2147483647\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(-2147483648); got != -1 {
		fmt.Printf("mod_int32 -2147483648%s2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_2147483647_int32_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s-2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(-2147483647); got != 0 {
		fmt.Printf("mod_int32 -2147483647%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_2147483647_int32_ssa(-1); got != 0 {
		fmt.Printf("mod_int32 2147483647%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(-1); got != -1 {
		fmt.Printf("mod_int32 -1%s2147483647 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(0); got != 0 {
		fmt.Printf("mod_int32 0%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_2147483647_int32_ssa(1); got != 0 {
		fmt.Printf("mod_int32 2147483647%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(1); got != 1 {
		fmt.Printf("mod_int32 1%s2147483647 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_2147483647_int32_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int32_2147483647_ssa(2147483647); got != 0 {
		fmt.Printf("mod_int32 2147483647%s2147483647 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_0_uint16_ssa(0); got != 0 {
		fmt.Printf("add_uint16 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint16_0_ssa(0); got != 0 {
		fmt.Printf("add_uint16 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_uint16_ssa(1); got != 1 {
		fmt.Printf("add_uint16 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint16_0_ssa(1); got != 1 {
		fmt.Printf("add_uint16 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_uint16_ssa(65535); got != 65535 {
		fmt.Printf("add_uint16 0%s65535 = %d, wanted 65535\n", `+`, got)
		failed = true
	}

	if got := add_uint16_0_ssa(65535); got != 65535 {
		fmt.Printf("add_uint16 65535%s0 = %d, wanted 65535\n", `+`, got)
		failed = true
	}

	if got := add_1_uint16_ssa(0); got != 1 {
		fmt.Printf("add_uint16 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint16_1_ssa(0); got != 1 {
		fmt.Printf("add_uint16 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_uint16_ssa(1); got != 2 {
		fmt.Printf("add_uint16 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_uint16_1_ssa(1); got != 2 {
		fmt.Printf("add_uint16 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_uint16_ssa(65535); got != 0 {
		fmt.Printf("add_uint16 1%s65535 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint16_1_ssa(65535); got != 0 {
		fmt.Printf("add_uint16 65535%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_65535_uint16_ssa(0); got != 65535 {
		fmt.Printf("add_uint16 65535%s0 = %d, wanted 65535\n", `+`, got)
		failed = true
	}

	if got := add_uint16_65535_ssa(0); got != 65535 {
		fmt.Printf("add_uint16 0%s65535 = %d, wanted 65535\n", `+`, got)
		failed = true
	}

	if got := add_65535_uint16_ssa(1); got != 0 {
		fmt.Printf("add_uint16 65535%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint16_65535_ssa(1); got != 0 {
		fmt.Printf("add_uint16 1%s65535 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_65535_uint16_ssa(65535); got != 65534 {
		fmt.Printf("add_uint16 65535%s65535 = %d, wanted 65534\n", `+`, got)
		failed = true
	}

	if got := add_uint16_65535_ssa(65535); got != 65534 {
		fmt.Printf("add_uint16 65535%s65535 = %d, wanted 65534\n", `+`, got)
		failed = true
	}

	if got := sub_0_uint16_ssa(0); got != 0 {
		fmt.Printf("sub_uint16 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_0_ssa(0); got != 0 {
		fmt.Printf("sub_uint16 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint16_ssa(1); got != 65535 {
		fmt.Printf("sub_uint16 0%s1 = %d, wanted 65535\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_0_ssa(1); got != 1 {
		fmt.Printf("sub_uint16 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint16_ssa(65535); got != 1 {
		fmt.Printf("sub_uint16 0%s65535 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_0_ssa(65535); got != 65535 {
		fmt.Printf("sub_uint16 65535%s0 = %d, wanted 65535\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint16_ssa(0); got != 1 {
		fmt.Printf("sub_uint16 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_1_ssa(0); got != 65535 {
		fmt.Printf("sub_uint16 0%s1 = %d, wanted 65535\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint16_ssa(1); got != 0 {
		fmt.Printf("sub_uint16 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_1_ssa(1); got != 0 {
		fmt.Printf("sub_uint16 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint16_ssa(65535); got != 2 {
		fmt.Printf("sub_uint16 1%s65535 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_1_ssa(65535); got != 65534 {
		fmt.Printf("sub_uint16 65535%s1 = %d, wanted 65534\n", `-`, got)
		failed = true
	}

	if got := sub_65535_uint16_ssa(0); got != 65535 {
		fmt.Printf("sub_uint16 65535%s0 = %d, wanted 65535\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_65535_ssa(0); got != 1 {
		fmt.Printf("sub_uint16 0%s65535 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_65535_uint16_ssa(1); got != 65534 {
		fmt.Printf("sub_uint16 65535%s1 = %d, wanted 65534\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_65535_ssa(1); got != 2 {
		fmt.Printf("sub_uint16 1%s65535 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_65535_uint16_ssa(65535); got != 0 {
		fmt.Printf("sub_uint16 65535%s65535 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint16_65535_ssa(65535); got != 0 {
		fmt.Printf("sub_uint16 65535%s65535 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_0_uint16_ssa(1); got != 0 {
		fmt.Printf("div_uint16 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_uint16_ssa(65535); got != 0 {
		fmt.Printf("div_uint16 0%s65535 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint16_1_ssa(0); got != 0 {
		fmt.Printf("div_uint16 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_uint16_ssa(1); got != 1 {
		fmt.Printf("div_uint16 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint16_1_ssa(1); got != 1 {
		fmt.Printf("div_uint16 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_uint16_ssa(65535); got != 0 {
		fmt.Printf("div_uint16 1%s65535 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint16_1_ssa(65535); got != 65535 {
		fmt.Printf("div_uint16 65535%s1 = %d, wanted 65535\n", `/`, got)
		failed = true
	}

	if got := div_uint16_65535_ssa(0); got != 0 {
		fmt.Printf("div_uint16 0%s65535 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_65535_uint16_ssa(1); got != 65535 {
		fmt.Printf("div_uint16 65535%s1 = %d, wanted 65535\n", `/`, got)
		failed = true
	}

	if got := div_uint16_65535_ssa(1); got != 0 {
		fmt.Printf("div_uint16 1%s65535 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_65535_uint16_ssa(65535); got != 1 {
		fmt.Printf("div_uint16 65535%s65535 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint16_65535_ssa(65535); got != 1 {
		fmt.Printf("div_uint16 65535%s65535 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_0_uint16_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_0_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint16_ssa(1); got != 0 {
		fmt.Printf("mul_uint16 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_0_ssa(1); got != 0 {
		fmt.Printf("mul_uint16 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint16_ssa(65535); got != 0 {
		fmt.Printf("mul_uint16 0%s65535 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_0_ssa(65535); got != 0 {
		fmt.Printf("mul_uint16 65535%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint16_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_1_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint16_ssa(1); got != 1 {
		fmt.Printf("mul_uint16 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_1_ssa(1); got != 1 {
		fmt.Printf("mul_uint16 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint16_ssa(65535); got != 65535 {
		fmt.Printf("mul_uint16 1%s65535 = %d, wanted 65535\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_1_ssa(65535); got != 65535 {
		fmt.Printf("mul_uint16 65535%s1 = %d, wanted 65535\n", `*`, got)
		failed = true
	}

	if got := mul_65535_uint16_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 65535%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_65535_ssa(0); got != 0 {
		fmt.Printf("mul_uint16 0%s65535 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_65535_uint16_ssa(1); got != 65535 {
		fmt.Printf("mul_uint16 65535%s1 = %d, wanted 65535\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_65535_ssa(1); got != 65535 {
		fmt.Printf("mul_uint16 1%s65535 = %d, wanted 65535\n", `*`, got)
		failed = true
	}

	if got := mul_65535_uint16_ssa(65535); got != 1 {
		fmt.Printf("mul_uint16 65535%s65535 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint16_65535_ssa(65535); got != 1 {
		fmt.Printf("mul_uint16 65535%s65535 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := lsh_0_uint16_ssa(0); got != 0 {
		fmt.Printf("lsh_uint16 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_0_ssa(0); got != 0 {
		fmt.Printf("lsh_uint16 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint16_ssa(1); got != 0 {
		fmt.Printf("lsh_uint16 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_0_ssa(1); got != 1 {
		fmt.Printf("lsh_uint16 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint16_ssa(65535); got != 0 {
		fmt.Printf("lsh_uint16 0%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_0_ssa(65535); got != 65535 {
		fmt.Printf("lsh_uint16 65535%s0 = %d, wanted 65535\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint16_ssa(0); got != 1 {
		fmt.Printf("lsh_uint16 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_1_ssa(0); got != 0 {
		fmt.Printf("lsh_uint16 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint16_ssa(1); got != 2 {
		fmt.Printf("lsh_uint16 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_1_ssa(1); got != 2 {
		fmt.Printf("lsh_uint16 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint16_ssa(65535); got != 0 {
		fmt.Printf("lsh_uint16 1%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_1_ssa(65535); got != 65534 {
		fmt.Printf("lsh_uint16 65535%s1 = %d, wanted 65534\n", `<<`, got)
		failed = true
	}

	if got := lsh_65535_uint16_ssa(0); got != 65535 {
		fmt.Printf("lsh_uint16 65535%s0 = %d, wanted 65535\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_65535_ssa(0); got != 0 {
		fmt.Printf("lsh_uint16 0%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_65535_uint16_ssa(1); got != 65534 {
		fmt.Printf("lsh_uint16 65535%s1 = %d, wanted 65534\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_65535_ssa(1); got != 0 {
		fmt.Printf("lsh_uint16 1%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_65535_uint16_ssa(65535); got != 0 {
		fmt.Printf("lsh_uint16 65535%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint16_65535_ssa(65535); got != 0 {
		fmt.Printf("lsh_uint16 65535%s65535 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := rsh_0_uint16_ssa(0); got != 0 {
		fmt.Printf("rsh_uint16 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_0_ssa(0); got != 0 {
		fmt.Printf("rsh_uint16 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint16_ssa(1); got != 0 {
		fmt.Printf("rsh_uint16 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_0_ssa(1); got != 1 {
		fmt.Printf("rsh_uint16 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint16_ssa(65535); got != 0 {
		fmt.Printf("rsh_uint16 0%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_0_ssa(65535); got != 65535 {
		fmt.Printf("rsh_uint16 65535%s0 = %d, wanted 65535\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint16_ssa(0); got != 1 {
		fmt.Printf("rsh_uint16 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_1_ssa(0); got != 0 {
		fmt.Printf("rsh_uint16 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint16_ssa(1); got != 0 {
		fmt.Printf("rsh_uint16 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_1_ssa(1); got != 0 {
		fmt.Printf("rsh_uint16 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint16_ssa(65535); got != 0 {
		fmt.Printf("rsh_uint16 1%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_1_ssa(65535); got != 32767 {
		fmt.Printf("rsh_uint16 65535%s1 = %d, wanted 32767\n", `>>`, got)
		failed = true
	}

	if got := rsh_65535_uint16_ssa(0); got != 65535 {
		fmt.Printf("rsh_uint16 65535%s0 = %d, wanted 65535\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_65535_ssa(0); got != 0 {
		fmt.Printf("rsh_uint16 0%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_65535_uint16_ssa(1); got != 32767 {
		fmt.Printf("rsh_uint16 65535%s1 = %d, wanted 32767\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_65535_ssa(1); got != 0 {
		fmt.Printf("rsh_uint16 1%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_65535_uint16_ssa(65535); got != 0 {
		fmt.Printf("rsh_uint16 65535%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint16_65535_ssa(65535); got != 0 {
		fmt.Printf("rsh_uint16 65535%s65535 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := mod_0_uint16_ssa(1); got != 0 {
		fmt.Printf("mod_uint16 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_uint16_ssa(65535); got != 0 {
		fmt.Printf("mod_uint16 0%s65535 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_1_ssa(0); got != 0 {
		fmt.Printf("mod_uint16 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint16_ssa(1); got != 0 {
		fmt.Printf("mod_uint16 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_1_ssa(1); got != 0 {
		fmt.Printf("mod_uint16 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint16_ssa(65535); got != 1 {
		fmt.Printf("mod_uint16 1%s65535 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_1_ssa(65535); got != 0 {
		fmt.Printf("mod_uint16 65535%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_65535_ssa(0); got != 0 {
		fmt.Printf("mod_uint16 0%s65535 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_65535_uint16_ssa(1); got != 0 {
		fmt.Printf("mod_uint16 65535%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_65535_ssa(1); got != 1 {
		fmt.Printf("mod_uint16 1%s65535 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_65535_uint16_ssa(65535); got != 0 {
		fmt.Printf("mod_uint16 65535%s65535 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint16_65535_ssa(65535); got != 0 {
		fmt.Printf("mod_uint16 65535%s65535 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(-32768); got != 0 {
		fmt.Printf("add_int16 -32768%s-32768 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(-32768); got != 0 {
		fmt.Printf("add_int16 -32768%s-32768 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(-32767); got != 1 {
		fmt.Printf("add_int16 -32768%s-32767 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(-32767); got != 1 {
		fmt.Printf("add_int16 -32767%s-32768 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(-1); got != 32767 {
		fmt.Printf("add_int16 -32768%s-1 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(-1); got != 32767 {
		fmt.Printf("add_int16 -1%s-32768 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(0); got != -32768 {
		fmt.Printf("add_int16 -32768%s0 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(0); got != -32768 {
		fmt.Printf("add_int16 0%s-32768 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(1); got != -32767 {
		fmt.Printf("add_int16 -32768%s1 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(1); got != -32767 {
		fmt.Printf("add_int16 1%s-32768 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(32766); got != -2 {
		fmt.Printf("add_int16 -32768%s32766 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(32766); got != -2 {
		fmt.Printf("add_int16 32766%s-32768 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg32768_int16_ssa(32767); got != -1 {
		fmt.Printf("add_int16 -32768%s32767 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32768_ssa(32767); got != -1 {
		fmt.Printf("add_int16 32767%s-32768 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(-32768); got != 1 {
		fmt.Printf("add_int16 -32767%s-32768 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(-32768); got != 1 {
		fmt.Printf("add_int16 -32768%s-32767 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(-32767); got != 2 {
		fmt.Printf("add_int16 -32767%s-32767 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(-32767); got != 2 {
		fmt.Printf("add_int16 -32767%s-32767 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(-1); got != -32768 {
		fmt.Printf("add_int16 -32767%s-1 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(-1); got != -32768 {
		fmt.Printf("add_int16 -1%s-32767 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(0); got != -32767 {
		fmt.Printf("add_int16 -32767%s0 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(0); got != -32767 {
		fmt.Printf("add_int16 0%s-32767 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(1); got != -32766 {
		fmt.Printf("add_int16 -32767%s1 = %d, wanted -32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(1); got != -32766 {
		fmt.Printf("add_int16 1%s-32767 = %d, wanted -32766\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(32766); got != -1 {
		fmt.Printf("add_int16 -32767%s32766 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(32766); got != -1 {
		fmt.Printf("add_int16 32766%s-32767 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg32767_int16_ssa(32767); got != 0 {
		fmt.Printf("add_int16 -32767%s32767 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg32767_ssa(32767); got != 0 {
		fmt.Printf("add_int16 32767%s-32767 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(-32768); got != 32767 {
		fmt.Printf("add_int16 -1%s-32768 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(-32768); got != 32767 {
		fmt.Printf("add_int16 -32768%s-1 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(-32767); got != -32768 {
		fmt.Printf("add_int16 -1%s-32767 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(-32767); got != -32768 {
		fmt.Printf("add_int16 -32767%s-1 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(-1); got != -2 {
		fmt.Printf("add_int16 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(-1); got != -2 {
		fmt.Printf("add_int16 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(0); got != -1 {
		fmt.Printf("add_int16 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(0); got != -1 {
		fmt.Printf("add_int16 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(1); got != 0 {
		fmt.Printf("add_int16 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(1); got != 0 {
		fmt.Printf("add_int16 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(32766); got != 32765 {
		fmt.Printf("add_int16 -1%s32766 = %d, wanted 32765\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(32766); got != 32765 {
		fmt.Printf("add_int16 32766%s-1 = %d, wanted 32765\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int16_ssa(32767); got != 32766 {
		fmt.Printf("add_int16 -1%s32767 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_Neg1_ssa(32767); got != 32766 {
		fmt.Printf("add_int16 32767%s-1 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(-32768); got != -32768 {
		fmt.Printf("add_int16 0%s-32768 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(-32768); got != -32768 {
		fmt.Printf("add_int16 -32768%s0 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(-32767); got != -32767 {
		fmt.Printf("add_int16 0%s-32767 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(-32767); got != -32767 {
		fmt.Printf("add_int16 -32767%s0 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(-1); got != -1 {
		fmt.Printf("add_int16 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(-1); got != -1 {
		fmt.Printf("add_int16 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(0); got != 0 {
		fmt.Printf("add_int16 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(0); got != 0 {
		fmt.Printf("add_int16 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(1); got != 1 {
		fmt.Printf("add_int16 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(1); got != 1 {
		fmt.Printf("add_int16 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(32766); got != 32766 {
		fmt.Printf("add_int16 0%s32766 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(32766); got != 32766 {
		fmt.Printf("add_int16 32766%s0 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_0_int16_ssa(32767); got != 32767 {
		fmt.Printf("add_int16 0%s32767 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_0_ssa(32767); got != 32767 {
		fmt.Printf("add_int16 32767%s0 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(-32768); got != -32767 {
		fmt.Printf("add_int16 1%s-32768 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(-32768); got != -32767 {
		fmt.Printf("add_int16 -32768%s1 = %d, wanted -32767\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(-32767); got != -32766 {
		fmt.Printf("add_int16 1%s-32767 = %d, wanted -32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(-32767); got != -32766 {
		fmt.Printf("add_int16 -32767%s1 = %d, wanted -32766\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(-1); got != 0 {
		fmt.Printf("add_int16 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(-1); got != 0 {
		fmt.Printf("add_int16 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(0); got != 1 {
		fmt.Printf("add_int16 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(0); got != 1 {
		fmt.Printf("add_int16 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(1); got != 2 {
		fmt.Printf("add_int16 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(1); got != 2 {
		fmt.Printf("add_int16 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(32766); got != 32767 {
		fmt.Printf("add_int16 1%s32766 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(32766); got != 32767 {
		fmt.Printf("add_int16 32766%s1 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_1_int16_ssa(32767); got != -32768 {
		fmt.Printf("add_int16 1%s32767 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_1_ssa(32767); got != -32768 {
		fmt.Printf("add_int16 32767%s1 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(-32768); got != -2 {
		fmt.Printf("add_int16 32766%s-32768 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(-32768); got != -2 {
		fmt.Printf("add_int16 -32768%s32766 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(-32767); got != -1 {
		fmt.Printf("add_int16 32766%s-32767 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(-32767); got != -1 {
		fmt.Printf("add_int16 -32767%s32766 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(-1); got != 32765 {
		fmt.Printf("add_int16 32766%s-1 = %d, wanted 32765\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(-1); got != 32765 {
		fmt.Printf("add_int16 -1%s32766 = %d, wanted 32765\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(0); got != 32766 {
		fmt.Printf("add_int16 32766%s0 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(0); got != 32766 {
		fmt.Printf("add_int16 0%s32766 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(1); got != 32767 {
		fmt.Printf("add_int16 32766%s1 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(1); got != 32767 {
		fmt.Printf("add_int16 1%s32766 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(32766); got != -4 {
		fmt.Printf("add_int16 32766%s32766 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(32766); got != -4 {
		fmt.Printf("add_int16 32766%s32766 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_32766_int16_ssa(32767); got != -3 {
		fmt.Printf("add_int16 32766%s32767 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int16_32766_ssa(32767); got != -3 {
		fmt.Printf("add_int16 32767%s32766 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(-32768); got != -1 {
		fmt.Printf("add_int16 32767%s-32768 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(-32768); got != -1 {
		fmt.Printf("add_int16 -32768%s32767 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(-32767); got != 0 {
		fmt.Printf("add_int16 32767%s-32767 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(-32767); got != 0 {
		fmt.Printf("add_int16 -32767%s32767 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(-1); got != 32766 {
		fmt.Printf("add_int16 32767%s-1 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(-1); got != 32766 {
		fmt.Printf("add_int16 -1%s32767 = %d, wanted 32766\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(0); got != 32767 {
		fmt.Printf("add_int16 32767%s0 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(0); got != 32767 {
		fmt.Printf("add_int16 0%s32767 = %d, wanted 32767\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(1); got != -32768 {
		fmt.Printf("add_int16 32767%s1 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(1); got != -32768 {
		fmt.Printf("add_int16 1%s32767 = %d, wanted -32768\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(32766); got != -3 {
		fmt.Printf("add_int16 32767%s32766 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(32766); got != -3 {
		fmt.Printf("add_int16 32766%s32767 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_32767_int16_ssa(32767); got != -2 {
		fmt.Printf("add_int16 32767%s32767 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int16_32767_ssa(32767); got != -2 {
		fmt.Printf("add_int16 32767%s32767 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(-32768); got != 0 {
		fmt.Printf("sub_int16 -32768%s-32768 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(-32768); got != 0 {
		fmt.Printf("sub_int16 -32768%s-32768 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(-32767); got != -1 {
		fmt.Printf("sub_int16 -32768%s-32767 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(-32767); got != 1 {
		fmt.Printf("sub_int16 -32767%s-32768 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(-1); got != -32767 {
		fmt.Printf("sub_int16 -32768%s-1 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(-1); got != 32767 {
		fmt.Printf("sub_int16 -1%s-32768 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(0); got != -32768 {
		fmt.Printf("sub_int16 -32768%s0 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(0); got != -32768 {
		fmt.Printf("sub_int16 0%s-32768 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(1); got != 32767 {
		fmt.Printf("sub_int16 -32768%s1 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(1); got != -32767 {
		fmt.Printf("sub_int16 1%s-32768 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(32766); got != 2 {
		fmt.Printf("sub_int16 -32768%s32766 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(32766); got != -2 {
		fmt.Printf("sub_int16 32766%s-32768 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32768_int16_ssa(32767); got != 1 {
		fmt.Printf("sub_int16 -32768%s32767 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32768_ssa(32767); got != -1 {
		fmt.Printf("sub_int16 32767%s-32768 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(-32768); got != 1 {
		fmt.Printf("sub_int16 -32767%s-32768 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(-32768); got != -1 {
		fmt.Printf("sub_int16 -32768%s-32767 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(-32767); got != 0 {
		fmt.Printf("sub_int16 -32767%s-32767 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(-32767); got != 0 {
		fmt.Printf("sub_int16 -32767%s-32767 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(-1); got != -32766 {
		fmt.Printf("sub_int16 -32767%s-1 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(-1); got != 32766 {
		fmt.Printf("sub_int16 -1%s-32767 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(0); got != -32767 {
		fmt.Printf("sub_int16 -32767%s0 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(0); got != 32767 {
		fmt.Printf("sub_int16 0%s-32767 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(1); got != -32768 {
		fmt.Printf("sub_int16 -32767%s1 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(1); got != -32768 {
		fmt.Printf("sub_int16 1%s-32767 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(32766); got != 3 {
		fmt.Printf("sub_int16 -32767%s32766 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(32766); got != -3 {
		fmt.Printf("sub_int16 32766%s-32767 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_Neg32767_int16_ssa(32767); got != 2 {
		fmt.Printf("sub_int16 -32767%s32767 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg32767_ssa(32767); got != -2 {
		fmt.Printf("sub_int16 32767%s-32767 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(-32768); got != 32767 {
		fmt.Printf("sub_int16 -1%s-32768 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(-32768); got != -32767 {
		fmt.Printf("sub_int16 -32768%s-1 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(-32767); got != 32766 {
		fmt.Printf("sub_int16 -1%s-32767 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(-32767); got != -32766 {
		fmt.Printf("sub_int16 -32767%s-1 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(-1); got != 0 {
		fmt.Printf("sub_int16 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(-1); got != 0 {
		fmt.Printf("sub_int16 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(0); got != -1 {
		fmt.Printf("sub_int16 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(0); got != 1 {
		fmt.Printf("sub_int16 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(1); got != -2 {
		fmt.Printf("sub_int16 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(1); got != 2 {
		fmt.Printf("sub_int16 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(32766); got != -32767 {
		fmt.Printf("sub_int16 -1%s32766 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(32766); got != 32767 {
		fmt.Printf("sub_int16 32766%s-1 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int16_ssa(32767); got != -32768 {
		fmt.Printf("sub_int16 -1%s32767 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_Neg1_ssa(32767); got != -32768 {
		fmt.Printf("sub_int16 32767%s-1 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(-32768); got != -32768 {
		fmt.Printf("sub_int16 0%s-32768 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(-32768); got != -32768 {
		fmt.Printf("sub_int16 -32768%s0 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(-32767); got != 32767 {
		fmt.Printf("sub_int16 0%s-32767 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(-32767); got != -32767 {
		fmt.Printf("sub_int16 -32767%s0 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(-1); got != 1 {
		fmt.Printf("sub_int16 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(-1); got != -1 {
		fmt.Printf("sub_int16 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(0); got != 0 {
		fmt.Printf("sub_int16 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(0); got != 0 {
		fmt.Printf("sub_int16 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(1); got != -1 {
		fmt.Printf("sub_int16 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(1); got != 1 {
		fmt.Printf("sub_int16 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(32766); got != -32766 {
		fmt.Printf("sub_int16 0%s32766 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(32766); got != 32766 {
		fmt.Printf("sub_int16 32766%s0 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_0_int16_ssa(32767); got != -32767 {
		fmt.Printf("sub_int16 0%s32767 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_0_ssa(32767); got != 32767 {
		fmt.Printf("sub_int16 32767%s0 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(-32768); got != -32767 {
		fmt.Printf("sub_int16 1%s-32768 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(-32768); got != 32767 {
		fmt.Printf("sub_int16 -32768%s1 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(-32767); got != -32768 {
		fmt.Printf("sub_int16 1%s-32767 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(-32767); got != -32768 {
		fmt.Printf("sub_int16 -32767%s1 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(-1); got != 2 {
		fmt.Printf("sub_int16 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(-1); got != -2 {
		fmt.Printf("sub_int16 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(0); got != 1 {
		fmt.Printf("sub_int16 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(0); got != -1 {
		fmt.Printf("sub_int16 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(1); got != 0 {
		fmt.Printf("sub_int16 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(1); got != 0 {
		fmt.Printf("sub_int16 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(32766); got != -32765 {
		fmt.Printf("sub_int16 1%s32766 = %d, wanted -32765\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(32766); got != 32765 {
		fmt.Printf("sub_int16 32766%s1 = %d, wanted 32765\n", `-`, got)
		failed = true
	}

	if got := sub_1_int16_ssa(32767); got != -32766 {
		fmt.Printf("sub_int16 1%s32767 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_1_ssa(32767); got != 32766 {
		fmt.Printf("sub_int16 32767%s1 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(-32768); got != -2 {
		fmt.Printf("sub_int16 32766%s-32768 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(-32768); got != 2 {
		fmt.Printf("sub_int16 -32768%s32766 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(-32767); got != -3 {
		fmt.Printf("sub_int16 32766%s-32767 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(-32767); got != 3 {
		fmt.Printf("sub_int16 -32767%s32766 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(-1); got != 32767 {
		fmt.Printf("sub_int16 32766%s-1 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(-1); got != -32767 {
		fmt.Printf("sub_int16 -1%s32766 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(0); got != 32766 {
		fmt.Printf("sub_int16 32766%s0 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(0); got != -32766 {
		fmt.Printf("sub_int16 0%s32766 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(1); got != 32765 {
		fmt.Printf("sub_int16 32766%s1 = %d, wanted 32765\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(1); got != -32765 {
		fmt.Printf("sub_int16 1%s32766 = %d, wanted -32765\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(32766); got != 0 {
		fmt.Printf("sub_int16 32766%s32766 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(32766); got != 0 {
		fmt.Printf("sub_int16 32766%s32766 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_32766_int16_ssa(32767); got != -1 {
		fmt.Printf("sub_int16 32766%s32767 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32766_ssa(32767); got != 1 {
		fmt.Printf("sub_int16 32767%s32766 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(-32768); got != -1 {
		fmt.Printf("sub_int16 32767%s-32768 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(-32768); got != 1 {
		fmt.Printf("sub_int16 -32768%s32767 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(-32767); got != -2 {
		fmt.Printf("sub_int16 32767%s-32767 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(-32767); got != 2 {
		fmt.Printf("sub_int16 -32767%s32767 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(-1); got != -32768 {
		fmt.Printf("sub_int16 32767%s-1 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(-1); got != -32768 {
		fmt.Printf("sub_int16 -1%s32767 = %d, wanted -32768\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(0); got != 32767 {
		fmt.Printf("sub_int16 32767%s0 = %d, wanted 32767\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(0); got != -32767 {
		fmt.Printf("sub_int16 0%s32767 = %d, wanted -32767\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(1); got != 32766 {
		fmt.Printf("sub_int16 32767%s1 = %d, wanted 32766\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(1); got != -32766 {
		fmt.Printf("sub_int16 1%s32767 = %d, wanted -32766\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(32766); got != 1 {
		fmt.Printf("sub_int16 32767%s32766 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(32766); got != -1 {
		fmt.Printf("sub_int16 32766%s32767 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_32767_int16_ssa(32767); got != 0 {
		fmt.Printf("sub_int16 32767%s32767 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int16_32767_ssa(32767); got != 0 {
		fmt.Printf("sub_int16 32767%s32767 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(-32768); got != 1 {
		fmt.Printf("div_int16 -32768%s-32768 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(-32768); got != 1 {
		fmt.Printf("div_int16 -32768%s-32768 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(-32767); got != 1 {
		fmt.Printf("div_int16 -32768%s-32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(-32767); got != 0 {
		fmt.Printf("div_int16 -32767%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(-1); got != -32768 {
		fmt.Printf("div_int16 -32768%s-1 = %d, wanted -32768\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(-1); got != 0 {
		fmt.Printf("div_int16 -1%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(1); got != -32768 {
		fmt.Printf("div_int16 -32768%s1 = %d, wanted -32768\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(1); got != 0 {
		fmt.Printf("div_int16 1%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(32766); got != -1 {
		fmt.Printf("div_int16 -32768%s32766 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(32766); got != 0 {
		fmt.Printf("div_int16 32766%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32768_int16_ssa(32767); got != -1 {
		fmt.Printf("div_int16 -32768%s32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32768_ssa(32767); got != 0 {
		fmt.Printf("div_int16 32767%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 -32767%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(-32768); got != 1 {
		fmt.Printf("div_int16 -32768%s-32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(-32767); got != 1 {
		fmt.Printf("div_int16 -32767%s-32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(-32767); got != 1 {
		fmt.Printf("div_int16 -32767%s-32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(-1); got != 32767 {
		fmt.Printf("div_int16 -32767%s-1 = %d, wanted 32767\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(-1); got != 0 {
		fmt.Printf("div_int16 -1%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(1); got != -32767 {
		fmt.Printf("div_int16 -32767%s1 = %d, wanted -32767\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(1); got != 0 {
		fmt.Printf("div_int16 1%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(32766); got != -1 {
		fmt.Printf("div_int16 -32767%s32766 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(32766); got != 0 {
		fmt.Printf("div_int16 32766%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg32767_int16_ssa(32767); got != -1 {
		fmt.Printf("div_int16 -32767%s32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg32767_ssa(32767); got != -1 {
		fmt.Printf("div_int16 32767%s-32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 -1%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(-32768); got != -32768 {
		fmt.Printf("div_int16 -32768%s-1 = %d, wanted -32768\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(-32767); got != 0 {
		fmt.Printf("div_int16 -1%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(-32767); got != 32767 {
		fmt.Printf("div_int16 -32767%s-1 = %d, wanted 32767\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(-1); got != 1 {
		fmt.Printf("div_int16 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(-1); got != 1 {
		fmt.Printf("div_int16 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(1); got != -1 {
		fmt.Printf("div_int16 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(1); got != -1 {
		fmt.Printf("div_int16 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(32766); got != 0 {
		fmt.Printf("div_int16 -1%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(32766); got != -32766 {
		fmt.Printf("div_int16 32766%s-1 = %d, wanted -32766\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int16_ssa(32767); got != 0 {
		fmt.Printf("div_int16 -1%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_Neg1_ssa(32767); got != -32767 {
		fmt.Printf("div_int16 32767%s-1 = %d, wanted -32767\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 0%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(-32767); got != 0 {
		fmt.Printf("div_int16 0%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(-1); got != 0 {
		fmt.Printf("div_int16 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(1); got != 0 {
		fmt.Printf("div_int16 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(32766); got != 0 {
		fmt.Printf("div_int16 0%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int16_ssa(32767); got != 0 {
		fmt.Printf("div_int16 0%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 1%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(-32768); got != -32768 {
		fmt.Printf("div_int16 -32768%s1 = %d, wanted -32768\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(-32767); got != 0 {
		fmt.Printf("div_int16 1%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(-32767); got != -32767 {
		fmt.Printf("div_int16 -32767%s1 = %d, wanted -32767\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(-1); got != -1 {
		fmt.Printf("div_int16 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(-1); got != -1 {
		fmt.Printf("div_int16 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(1); got != 1 {
		fmt.Printf("div_int16 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(1); got != 1 {
		fmt.Printf("div_int16 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(32766); got != 0 {
		fmt.Printf("div_int16 1%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(32766); got != 32766 {
		fmt.Printf("div_int16 32766%s1 = %d, wanted 32766\n", `/`, got)
		failed = true
	}

	if got := div_1_int16_ssa(32767); got != 0 {
		fmt.Printf("div_int16 1%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_1_ssa(32767); got != 32767 {
		fmt.Printf("div_int16 32767%s1 = %d, wanted 32767\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 32766%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(-32768); got != -1 {
		fmt.Printf("div_int16 -32768%s32766 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(-32767); got != 0 {
		fmt.Printf("div_int16 32766%s-32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(-32767); got != -1 {
		fmt.Printf("div_int16 -32767%s32766 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(-1); got != -32766 {
		fmt.Printf("div_int16 32766%s-1 = %d, wanted -32766\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(-1); got != 0 {
		fmt.Printf("div_int16 -1%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(1); got != 32766 {
		fmt.Printf("div_int16 32766%s1 = %d, wanted 32766\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(1); got != 0 {
		fmt.Printf("div_int16 1%s32766 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(32766); got != 1 {
		fmt.Printf("div_int16 32766%s32766 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(32766); got != 1 {
		fmt.Printf("div_int16 32766%s32766 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_32766_int16_ssa(32767); got != 0 {
		fmt.Printf("div_int16 32766%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32766_ssa(32767); got != 1 {
		fmt.Printf("div_int16 32767%s32766 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(-32768); got != 0 {
		fmt.Printf("div_int16 32767%s-32768 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(-32768); got != -1 {
		fmt.Printf("div_int16 -32768%s32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(-32767); got != -1 {
		fmt.Printf("div_int16 32767%s-32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(-32767); got != -1 {
		fmt.Printf("div_int16 -32767%s32767 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(-1); got != -32767 {
		fmt.Printf("div_int16 32767%s-1 = %d, wanted -32767\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(-1); got != 0 {
		fmt.Printf("div_int16 -1%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(0); got != 0 {
		fmt.Printf("div_int16 0%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(1); got != 32767 {
		fmt.Printf("div_int16 32767%s1 = %d, wanted 32767\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(1); got != 0 {
		fmt.Printf("div_int16 1%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(32766); got != 1 {
		fmt.Printf("div_int16 32767%s32766 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(32766); got != 0 {
		fmt.Printf("div_int16 32766%s32767 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_32767_int16_ssa(32767); got != 1 {
		fmt.Printf("div_int16 32767%s32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int16_32767_ssa(32767); got != 1 {
		fmt.Printf("div_int16 32767%s32767 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 -32768%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 -32768%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(-32767); got != -32768 {
		fmt.Printf("mul_int16 -32768%s-32767 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(-32767); got != -32768 {
		fmt.Printf("mul_int16 -32767%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(-1); got != -32768 {
		fmt.Printf("mul_int16 -32768%s-1 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(-1); got != -32768 {
		fmt.Printf("mul_int16 -1%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 -32768%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(1); got != -32768 {
		fmt.Printf("mul_int16 -32768%s1 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(1); got != -32768 {
		fmt.Printf("mul_int16 1%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(32766); got != 0 {
		fmt.Printf("mul_int16 -32768%s32766 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(32766); got != 0 {
		fmt.Printf("mul_int16 32766%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32768_int16_ssa(32767); got != -32768 {
		fmt.Printf("mul_int16 -32768%s32767 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32768_ssa(32767); got != -32768 {
		fmt.Printf("mul_int16 32767%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -32767%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -32768%s-32767 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(-32767); got != 1 {
		fmt.Printf("mul_int16 -32767%s-32767 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(-32767); got != 1 {
		fmt.Printf("mul_int16 -32767%s-32767 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(-1); got != 32767 {
		fmt.Printf("mul_int16 -32767%s-1 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(-1); got != 32767 {
		fmt.Printf("mul_int16 -1%s-32767 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 -32767%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s-32767 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(1); got != -32767 {
		fmt.Printf("mul_int16 -32767%s1 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(1); got != -32767 {
		fmt.Printf("mul_int16 1%s-32767 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(32766); got != 32766 {
		fmt.Printf("mul_int16 -32767%s32766 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(32766); got != 32766 {
		fmt.Printf("mul_int16 32766%s-32767 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_Neg32767_int16_ssa(32767); got != -1 {
		fmt.Printf("mul_int16 -32767%s32767 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg32767_ssa(32767); got != -1 {
		fmt.Printf("mul_int16 32767%s-32767 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -1%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -32768%s-1 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(-32767); got != 32767 {
		fmt.Printf("mul_int16 -1%s-32767 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(-32767); got != 32767 {
		fmt.Printf("mul_int16 -32767%s-1 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(-1); got != 1 {
		fmt.Printf("mul_int16 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(-1); got != 1 {
		fmt.Printf("mul_int16 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(1); got != -1 {
		fmt.Printf("mul_int16 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(1); got != -1 {
		fmt.Printf("mul_int16 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(32766); got != -32766 {
		fmt.Printf("mul_int16 -1%s32766 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(32766); got != -32766 {
		fmt.Printf("mul_int16 32766%s-1 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int16_ssa(32767); got != -32767 {
		fmt.Printf("mul_int16 -1%s32767 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_Neg1_ssa(32767); got != -32767 {
		fmt.Printf("mul_int16 32767%s-1 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 0%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 -32768%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(-32767); got != 0 {
		fmt.Printf("mul_int16 0%s-32767 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(-32767); got != 0 {
		fmt.Printf("mul_int16 -32767%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(-1); got != 0 {
		fmt.Printf("mul_int16 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(-1); got != 0 {
		fmt.Printf("mul_int16 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(1); got != 0 {
		fmt.Printf("mul_int16 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(1); got != 0 {
		fmt.Printf("mul_int16 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(32766); got != 0 {
		fmt.Printf("mul_int16 0%s32766 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(32766); got != 0 {
		fmt.Printf("mul_int16 32766%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int16_ssa(32767); got != 0 {
		fmt.Printf("mul_int16 0%s32767 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_0_ssa(32767); got != 0 {
		fmt.Printf("mul_int16 32767%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 1%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -32768%s1 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(-32767); got != -32767 {
		fmt.Printf("mul_int16 1%s-32767 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(-32767); got != -32767 {
		fmt.Printf("mul_int16 -32767%s1 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(-1); got != -1 {
		fmt.Printf("mul_int16 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(-1); got != -1 {
		fmt.Printf("mul_int16 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(1); got != 1 {
		fmt.Printf("mul_int16 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(1); got != 1 {
		fmt.Printf("mul_int16 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(32766); got != 32766 {
		fmt.Printf("mul_int16 1%s32766 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(32766); got != 32766 {
		fmt.Printf("mul_int16 32766%s1 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_1_int16_ssa(32767); got != 32767 {
		fmt.Printf("mul_int16 1%s32767 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_1_ssa(32767); got != 32767 {
		fmt.Printf("mul_int16 32767%s1 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 32766%s-32768 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(-32768); got != 0 {
		fmt.Printf("mul_int16 -32768%s32766 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(-32767); got != 32766 {
		fmt.Printf("mul_int16 32766%s-32767 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(-32767); got != 32766 {
		fmt.Printf("mul_int16 -32767%s32766 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(-1); got != -32766 {
		fmt.Printf("mul_int16 32766%s-1 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(-1); got != -32766 {
		fmt.Printf("mul_int16 -1%s32766 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 32766%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s32766 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(1); got != 32766 {
		fmt.Printf("mul_int16 32766%s1 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(1); got != 32766 {
		fmt.Printf("mul_int16 1%s32766 = %d, wanted 32766\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(32766); got != 4 {
		fmt.Printf("mul_int16 32766%s32766 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(32766); got != 4 {
		fmt.Printf("mul_int16 32766%s32766 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_32766_int16_ssa(32767); got != -32766 {
		fmt.Printf("mul_int16 32766%s32767 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32766_ssa(32767); got != -32766 {
		fmt.Printf("mul_int16 32767%s32766 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 32767%s-32768 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(-32768); got != -32768 {
		fmt.Printf("mul_int16 -32768%s32767 = %d, wanted -32768\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(-32767); got != -1 {
		fmt.Printf("mul_int16 32767%s-32767 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(-32767); got != -1 {
		fmt.Printf("mul_int16 -32767%s32767 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(-1); got != -32767 {
		fmt.Printf("mul_int16 32767%s-1 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(-1); got != -32767 {
		fmt.Printf("mul_int16 -1%s32767 = %d, wanted -32767\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(0); got != 0 {
		fmt.Printf("mul_int16 32767%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(0); got != 0 {
		fmt.Printf("mul_int16 0%s32767 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(1); got != 32767 {
		fmt.Printf("mul_int16 32767%s1 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(1); got != 32767 {
		fmt.Printf("mul_int16 1%s32767 = %d, wanted 32767\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(32766); got != -32766 {
		fmt.Printf("mul_int16 32767%s32766 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(32766); got != -32766 {
		fmt.Printf("mul_int16 32766%s32767 = %d, wanted -32766\n", `*`, got)
		failed = true
	}

	if got := mul_32767_int16_ssa(32767); got != 1 {
		fmt.Printf("mul_int16 32767%s32767 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int16_32767_ssa(32767); got != 1 {
		fmt.Printf("mul_int16 32767%s32767 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(-32768); got != 0 {
		fmt.Printf("mod_int16 -32768%s-32768 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(-32768); got != 0 {
		fmt.Printf("mod_int16 -32768%s-32768 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(-32767); got != -1 {
		fmt.Printf("mod_int16 -32768%s-32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(-32767); got != -32767 {
		fmt.Printf("mod_int16 -32767%s-32768 = %d, wanted -32767\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 -32768%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(-1); got != -1 {
		fmt.Printf("mod_int16 -1%s-32768 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s-32768 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 -32768%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(1); got != 1 {
		fmt.Printf("mod_int16 1%s-32768 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(32766); got != -2 {
		fmt.Printf("mod_int16 -32768%s32766 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(32766); got != 32766 {
		fmt.Printf("mod_int16 32766%s-32768 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32768_int16_ssa(32767); got != -1 {
		fmt.Printf("mod_int16 -32768%s32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32768_ssa(32767); got != 32767 {
		fmt.Printf("mod_int16 32767%s-32768 = %d, wanted 32767\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(-32768); got != -32767 {
		fmt.Printf("mod_int16 -32767%s-32768 = %d, wanted -32767\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(-32768); got != -1 {
		fmt.Printf("mod_int16 -32768%s-32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 -32767%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(-1); got != -1 {
		fmt.Printf("mod_int16 -1%s-32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 -32767%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(1); got != 1 {
		fmt.Printf("mod_int16 1%s-32767 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(32766); got != -1 {
		fmt.Printf("mod_int16 -32767%s32766 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(32766); got != 32766 {
		fmt.Printf("mod_int16 32766%s-32767 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_Neg32767_int16_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg32767_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 32767%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(-32768); got != -1 {
		fmt.Printf("mod_int16 -1%s-32768 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(-32768); got != 0 {
		fmt.Printf("mod_int16 -32768%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(-32767); got != -1 {
		fmt.Printf("mod_int16 -1%s-32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(1); got != 0 {
		fmt.Printf("mod_int16 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(32766); got != -1 {
		fmt.Printf("mod_int16 -1%s32766 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(32766); got != 0 {
		fmt.Printf("mod_int16 32766%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int16_ssa(32767); got != -1 {
		fmt.Printf("mod_int16 -1%s32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_Neg1_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 32767%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(-32768); got != 0 {
		fmt.Printf("mod_int16 0%s-32768 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 0%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(32766); got != 0 {
		fmt.Printf("mod_int16 0%s32766 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int16_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 0%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(-32768); got != 1 {
		fmt.Printf("mod_int16 1%s-32768 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(-32768); got != 0 {
		fmt.Printf("mod_int16 -32768%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(-32767); got != 1 {
		fmt.Printf("mod_int16 1%s-32767 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(1); got != 0 {
		fmt.Printf("mod_int16 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(32766); got != 1 {
		fmt.Printf("mod_int16 1%s32766 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(32766); got != 0 {
		fmt.Printf("mod_int16 32766%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int16_ssa(32767); got != 1 {
		fmt.Printf("mod_int16 1%s32767 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_1_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 32767%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(-32768); got != 32766 {
		fmt.Printf("mod_int16 32766%s-32768 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(-32768); got != -2 {
		fmt.Printf("mod_int16 -32768%s32766 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(-32767); got != 32766 {
		fmt.Printf("mod_int16 32766%s-32767 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(-32767); got != -1 {
		fmt.Printf("mod_int16 -32767%s32766 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 32766%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(-1); got != -1 {
		fmt.Printf("mod_int16 -1%s32766 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s32766 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 32766%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(1); got != 1 {
		fmt.Printf("mod_int16 1%s32766 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(32766); got != 0 {
		fmt.Printf("mod_int16 32766%s32766 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(32766); got != 0 {
		fmt.Printf("mod_int16 32766%s32766 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_32766_int16_ssa(32767); got != 32766 {
		fmt.Printf("mod_int16 32766%s32767 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32766_ssa(32767); got != 1 {
		fmt.Printf("mod_int16 32767%s32766 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(-32768); got != 32767 {
		fmt.Printf("mod_int16 32767%s-32768 = %d, wanted 32767\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(-32768); got != -1 {
		fmt.Printf("mod_int16 -32768%s32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 32767%s-32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(-32767); got != 0 {
		fmt.Printf("mod_int16 -32767%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(-1); got != 0 {
		fmt.Printf("mod_int16 32767%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(-1); got != -1 {
		fmt.Printf("mod_int16 -1%s32767 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(0); got != 0 {
		fmt.Printf("mod_int16 0%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(1); got != 0 {
		fmt.Printf("mod_int16 32767%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(1); got != 1 {
		fmt.Printf("mod_int16 1%s32767 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(32766); got != 1 {
		fmt.Printf("mod_int16 32767%s32766 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(32766); got != 32766 {
		fmt.Printf("mod_int16 32766%s32767 = %d, wanted 32766\n", `%`, got)
		failed = true
	}

	if got := mod_32767_int16_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 32767%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int16_32767_ssa(32767); got != 0 {
		fmt.Printf("mod_int16 32767%s32767 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_0_uint8_ssa(0); got != 0 {
		fmt.Printf("add_uint8 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint8_0_ssa(0); got != 0 {
		fmt.Printf("add_uint8 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_uint8_ssa(1); got != 1 {
		fmt.Printf("add_uint8 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint8_0_ssa(1); got != 1 {
		fmt.Printf("add_uint8 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_uint8_ssa(255); got != 255 {
		fmt.Printf("add_uint8 0%s255 = %d, wanted 255\n", `+`, got)
		failed = true
	}

	if got := add_uint8_0_ssa(255); got != 255 {
		fmt.Printf("add_uint8 255%s0 = %d, wanted 255\n", `+`, got)
		failed = true
	}

	if got := add_1_uint8_ssa(0); got != 1 {
		fmt.Printf("add_uint8 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_uint8_1_ssa(0); got != 1 {
		fmt.Printf("add_uint8 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_uint8_ssa(1); got != 2 {
		fmt.Printf("add_uint8 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_uint8_1_ssa(1); got != 2 {
		fmt.Printf("add_uint8 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_uint8_ssa(255); got != 0 {
		fmt.Printf("add_uint8 1%s255 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint8_1_ssa(255); got != 0 {
		fmt.Printf("add_uint8 255%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_255_uint8_ssa(0); got != 255 {
		fmt.Printf("add_uint8 255%s0 = %d, wanted 255\n", `+`, got)
		failed = true
	}

	if got := add_uint8_255_ssa(0); got != 255 {
		fmt.Printf("add_uint8 0%s255 = %d, wanted 255\n", `+`, got)
		failed = true
	}

	if got := add_255_uint8_ssa(1); got != 0 {
		fmt.Printf("add_uint8 255%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_uint8_255_ssa(1); got != 0 {
		fmt.Printf("add_uint8 1%s255 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_255_uint8_ssa(255); got != 254 {
		fmt.Printf("add_uint8 255%s255 = %d, wanted 254\n", `+`, got)
		failed = true
	}

	if got := add_uint8_255_ssa(255); got != 254 {
		fmt.Printf("add_uint8 255%s255 = %d, wanted 254\n", `+`, got)
		failed = true
	}

	if got := sub_0_uint8_ssa(0); got != 0 {
		fmt.Printf("sub_uint8 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_0_ssa(0); got != 0 {
		fmt.Printf("sub_uint8 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint8_ssa(1); got != 255 {
		fmt.Printf("sub_uint8 0%s1 = %d, wanted 255\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_0_ssa(1); got != 1 {
		fmt.Printf("sub_uint8 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_uint8_ssa(255); got != 1 {
		fmt.Printf("sub_uint8 0%s255 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_0_ssa(255); got != 255 {
		fmt.Printf("sub_uint8 255%s0 = %d, wanted 255\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint8_ssa(0); got != 1 {
		fmt.Printf("sub_uint8 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_1_ssa(0); got != 255 {
		fmt.Printf("sub_uint8 0%s1 = %d, wanted 255\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint8_ssa(1); got != 0 {
		fmt.Printf("sub_uint8 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_1_ssa(1); got != 0 {
		fmt.Printf("sub_uint8 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_uint8_ssa(255); got != 2 {
		fmt.Printf("sub_uint8 1%s255 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_1_ssa(255); got != 254 {
		fmt.Printf("sub_uint8 255%s1 = %d, wanted 254\n", `-`, got)
		failed = true
	}

	if got := sub_255_uint8_ssa(0); got != 255 {
		fmt.Printf("sub_uint8 255%s0 = %d, wanted 255\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_255_ssa(0); got != 1 {
		fmt.Printf("sub_uint8 0%s255 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_255_uint8_ssa(1); got != 254 {
		fmt.Printf("sub_uint8 255%s1 = %d, wanted 254\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_255_ssa(1); got != 2 {
		fmt.Printf("sub_uint8 1%s255 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_255_uint8_ssa(255); got != 0 {
		fmt.Printf("sub_uint8 255%s255 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_uint8_255_ssa(255); got != 0 {
		fmt.Printf("sub_uint8 255%s255 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_0_uint8_ssa(1); got != 0 {
		fmt.Printf("div_uint8 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_uint8_ssa(255); got != 0 {
		fmt.Printf("div_uint8 0%s255 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint8_1_ssa(0); got != 0 {
		fmt.Printf("div_uint8 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_uint8_ssa(1); got != 1 {
		fmt.Printf("div_uint8 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint8_1_ssa(1); got != 1 {
		fmt.Printf("div_uint8 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_uint8_ssa(255); got != 0 {
		fmt.Printf("div_uint8 1%s255 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_uint8_1_ssa(255); got != 255 {
		fmt.Printf("div_uint8 255%s1 = %d, wanted 255\n", `/`, got)
		failed = true
	}

	if got := div_uint8_255_ssa(0); got != 0 {
		fmt.Printf("div_uint8 0%s255 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_255_uint8_ssa(1); got != 255 {
		fmt.Printf("div_uint8 255%s1 = %d, wanted 255\n", `/`, got)
		failed = true
	}

	if got := div_uint8_255_ssa(1); got != 0 {
		fmt.Printf("div_uint8 1%s255 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_255_uint8_ssa(255); got != 1 {
		fmt.Printf("div_uint8 255%s255 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_uint8_255_ssa(255); got != 1 {
		fmt.Printf("div_uint8 255%s255 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_0_uint8_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_0_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint8_ssa(1); got != 0 {
		fmt.Printf("mul_uint8 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_0_ssa(1); got != 0 {
		fmt.Printf("mul_uint8 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_uint8_ssa(255); got != 0 {
		fmt.Printf("mul_uint8 0%s255 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_0_ssa(255); got != 0 {
		fmt.Printf("mul_uint8 255%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint8_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_1_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint8_ssa(1); got != 1 {
		fmt.Printf("mul_uint8 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_1_ssa(1); got != 1 {
		fmt.Printf("mul_uint8 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_uint8_ssa(255); got != 255 {
		fmt.Printf("mul_uint8 1%s255 = %d, wanted 255\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_1_ssa(255); got != 255 {
		fmt.Printf("mul_uint8 255%s1 = %d, wanted 255\n", `*`, got)
		failed = true
	}

	if got := mul_255_uint8_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 255%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_255_ssa(0); got != 0 {
		fmt.Printf("mul_uint8 0%s255 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_255_uint8_ssa(1); got != 255 {
		fmt.Printf("mul_uint8 255%s1 = %d, wanted 255\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_255_ssa(1); got != 255 {
		fmt.Printf("mul_uint8 1%s255 = %d, wanted 255\n", `*`, got)
		failed = true
	}

	if got := mul_255_uint8_ssa(255); got != 1 {
		fmt.Printf("mul_uint8 255%s255 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_uint8_255_ssa(255); got != 1 {
		fmt.Printf("mul_uint8 255%s255 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := lsh_0_uint8_ssa(0); got != 0 {
		fmt.Printf("lsh_uint8 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_0_ssa(0); got != 0 {
		fmt.Printf("lsh_uint8 0%s0 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint8_ssa(1); got != 0 {
		fmt.Printf("lsh_uint8 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_0_ssa(1); got != 1 {
		fmt.Printf("lsh_uint8 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_0_uint8_ssa(255); got != 0 {
		fmt.Printf("lsh_uint8 0%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_0_ssa(255); got != 255 {
		fmt.Printf("lsh_uint8 255%s0 = %d, wanted 255\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint8_ssa(0); got != 1 {
		fmt.Printf("lsh_uint8 1%s0 = %d, wanted 1\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_1_ssa(0); got != 0 {
		fmt.Printf("lsh_uint8 0%s1 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint8_ssa(1); got != 2 {
		fmt.Printf("lsh_uint8 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_1_ssa(1); got != 2 {
		fmt.Printf("lsh_uint8 1%s1 = %d, wanted 2\n", `<<`, got)
		failed = true
	}

	if got := lsh_1_uint8_ssa(255); got != 0 {
		fmt.Printf("lsh_uint8 1%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_1_ssa(255); got != 254 {
		fmt.Printf("lsh_uint8 255%s1 = %d, wanted 254\n", `<<`, got)
		failed = true
	}

	if got := lsh_255_uint8_ssa(0); got != 255 {
		fmt.Printf("lsh_uint8 255%s0 = %d, wanted 255\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_255_ssa(0); got != 0 {
		fmt.Printf("lsh_uint8 0%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_255_uint8_ssa(1); got != 254 {
		fmt.Printf("lsh_uint8 255%s1 = %d, wanted 254\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_255_ssa(1); got != 0 {
		fmt.Printf("lsh_uint8 1%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_255_uint8_ssa(255); got != 0 {
		fmt.Printf("lsh_uint8 255%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := lsh_uint8_255_ssa(255); got != 0 {
		fmt.Printf("lsh_uint8 255%s255 = %d, wanted 0\n", `<<`, got)
		failed = true
	}

	if got := rsh_0_uint8_ssa(0); got != 0 {
		fmt.Printf("rsh_uint8 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_0_ssa(0); got != 0 {
		fmt.Printf("rsh_uint8 0%s0 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint8_ssa(1); got != 0 {
		fmt.Printf("rsh_uint8 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_0_ssa(1); got != 1 {
		fmt.Printf("rsh_uint8 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_0_uint8_ssa(255); got != 0 {
		fmt.Printf("rsh_uint8 0%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_0_ssa(255); got != 255 {
		fmt.Printf("rsh_uint8 255%s0 = %d, wanted 255\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint8_ssa(0); got != 1 {
		fmt.Printf("rsh_uint8 1%s0 = %d, wanted 1\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_1_ssa(0); got != 0 {
		fmt.Printf("rsh_uint8 0%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint8_ssa(1); got != 0 {
		fmt.Printf("rsh_uint8 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_1_ssa(1); got != 0 {
		fmt.Printf("rsh_uint8 1%s1 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_1_uint8_ssa(255); got != 0 {
		fmt.Printf("rsh_uint8 1%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_1_ssa(255); got != 127 {
		fmt.Printf("rsh_uint8 255%s1 = %d, wanted 127\n", `>>`, got)
		failed = true
	}

	if got := rsh_255_uint8_ssa(0); got != 255 {
		fmt.Printf("rsh_uint8 255%s0 = %d, wanted 255\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_255_ssa(0); got != 0 {
		fmt.Printf("rsh_uint8 0%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_255_uint8_ssa(1); got != 127 {
		fmt.Printf("rsh_uint8 255%s1 = %d, wanted 127\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_255_ssa(1); got != 0 {
		fmt.Printf("rsh_uint8 1%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_255_uint8_ssa(255); got != 0 {
		fmt.Printf("rsh_uint8 255%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := rsh_uint8_255_ssa(255); got != 0 {
		fmt.Printf("rsh_uint8 255%s255 = %d, wanted 0\n", `>>`, got)
		failed = true
	}

	if got := mod_0_uint8_ssa(1); got != 0 {
		fmt.Printf("mod_uint8 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_uint8_ssa(255); got != 0 {
		fmt.Printf("mod_uint8 0%s255 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_1_ssa(0); got != 0 {
		fmt.Printf("mod_uint8 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint8_ssa(1); got != 0 {
		fmt.Printf("mod_uint8 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_1_ssa(1); got != 0 {
		fmt.Printf("mod_uint8 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_uint8_ssa(255); got != 1 {
		fmt.Printf("mod_uint8 1%s255 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_1_ssa(255); got != 0 {
		fmt.Printf("mod_uint8 255%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_255_ssa(0); got != 0 {
		fmt.Printf("mod_uint8 0%s255 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_255_uint8_ssa(1); got != 0 {
		fmt.Printf("mod_uint8 255%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_255_ssa(1); got != 1 {
		fmt.Printf("mod_uint8 1%s255 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_255_uint8_ssa(255); got != 0 {
		fmt.Printf("mod_uint8 255%s255 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_uint8_255_ssa(255); got != 0 {
		fmt.Printf("mod_uint8 255%s255 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(-128); got != 0 {
		fmt.Printf("add_int8 -128%s-128 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(-128); got != 0 {
		fmt.Printf("add_int8 -128%s-128 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(-127); got != 1 {
		fmt.Printf("add_int8 -128%s-127 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(-127); got != 1 {
		fmt.Printf("add_int8 -127%s-128 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(-1); got != 127 {
		fmt.Printf("add_int8 -128%s-1 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(-1); got != 127 {
		fmt.Printf("add_int8 -1%s-128 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(0); got != -128 {
		fmt.Printf("add_int8 -128%s0 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(0); got != -128 {
		fmt.Printf("add_int8 0%s-128 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(1); got != -127 {
		fmt.Printf("add_int8 -128%s1 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(1); got != -127 {
		fmt.Printf("add_int8 1%s-128 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(126); got != -2 {
		fmt.Printf("add_int8 -128%s126 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(126); got != -2 {
		fmt.Printf("add_int8 126%s-128 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg128_int8_ssa(127); got != -1 {
		fmt.Printf("add_int8 -128%s127 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg128_ssa(127); got != -1 {
		fmt.Printf("add_int8 127%s-128 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(-128); got != 1 {
		fmt.Printf("add_int8 -127%s-128 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(-128); got != 1 {
		fmt.Printf("add_int8 -128%s-127 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(-127); got != 2 {
		fmt.Printf("add_int8 -127%s-127 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(-127); got != 2 {
		fmt.Printf("add_int8 -127%s-127 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(-1); got != -128 {
		fmt.Printf("add_int8 -127%s-1 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(-1); got != -128 {
		fmt.Printf("add_int8 -1%s-127 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(0); got != -127 {
		fmt.Printf("add_int8 -127%s0 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(0); got != -127 {
		fmt.Printf("add_int8 0%s-127 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(1); got != -126 {
		fmt.Printf("add_int8 -127%s1 = %d, wanted -126\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(1); got != -126 {
		fmt.Printf("add_int8 1%s-127 = %d, wanted -126\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(126); got != -1 {
		fmt.Printf("add_int8 -127%s126 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(126); got != -1 {
		fmt.Printf("add_int8 126%s-127 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg127_int8_ssa(127); got != 0 {
		fmt.Printf("add_int8 -127%s127 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg127_ssa(127); got != 0 {
		fmt.Printf("add_int8 127%s-127 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(-128); got != 127 {
		fmt.Printf("add_int8 -1%s-128 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(-128); got != 127 {
		fmt.Printf("add_int8 -128%s-1 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(-127); got != -128 {
		fmt.Printf("add_int8 -1%s-127 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(-127); got != -128 {
		fmt.Printf("add_int8 -127%s-1 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(-1); got != -2 {
		fmt.Printf("add_int8 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(-1); got != -2 {
		fmt.Printf("add_int8 -1%s-1 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(0); got != -1 {
		fmt.Printf("add_int8 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(0); got != -1 {
		fmt.Printf("add_int8 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(1); got != 0 {
		fmt.Printf("add_int8 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(1); got != 0 {
		fmt.Printf("add_int8 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(126); got != 125 {
		fmt.Printf("add_int8 -1%s126 = %d, wanted 125\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(126); got != 125 {
		fmt.Printf("add_int8 126%s-1 = %d, wanted 125\n", `+`, got)
		failed = true
	}

	if got := add_Neg1_int8_ssa(127); got != 126 {
		fmt.Printf("add_int8 -1%s127 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_int8_Neg1_ssa(127); got != 126 {
		fmt.Printf("add_int8 127%s-1 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(-128); got != -128 {
		fmt.Printf("add_int8 0%s-128 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(-128); got != -128 {
		fmt.Printf("add_int8 -128%s0 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(-127); got != -127 {
		fmt.Printf("add_int8 0%s-127 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(-127); got != -127 {
		fmt.Printf("add_int8 -127%s0 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(-1); got != -1 {
		fmt.Printf("add_int8 0%s-1 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(-1); got != -1 {
		fmt.Printf("add_int8 -1%s0 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(0); got != 0 {
		fmt.Printf("add_int8 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(0); got != 0 {
		fmt.Printf("add_int8 0%s0 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(1); got != 1 {
		fmt.Printf("add_int8 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(1); got != 1 {
		fmt.Printf("add_int8 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(126); got != 126 {
		fmt.Printf("add_int8 0%s126 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(126); got != 126 {
		fmt.Printf("add_int8 126%s0 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_0_int8_ssa(127); got != 127 {
		fmt.Printf("add_int8 0%s127 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_0_ssa(127); got != 127 {
		fmt.Printf("add_int8 127%s0 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(-128); got != -127 {
		fmt.Printf("add_int8 1%s-128 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(-128); got != -127 {
		fmt.Printf("add_int8 -128%s1 = %d, wanted -127\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(-127); got != -126 {
		fmt.Printf("add_int8 1%s-127 = %d, wanted -126\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(-127); got != -126 {
		fmt.Printf("add_int8 -127%s1 = %d, wanted -126\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(-1); got != 0 {
		fmt.Printf("add_int8 1%s-1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(-1); got != 0 {
		fmt.Printf("add_int8 -1%s1 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(0); got != 1 {
		fmt.Printf("add_int8 1%s0 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(0); got != 1 {
		fmt.Printf("add_int8 0%s1 = %d, wanted 1\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(1); got != 2 {
		fmt.Printf("add_int8 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(1); got != 2 {
		fmt.Printf("add_int8 1%s1 = %d, wanted 2\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(126); got != 127 {
		fmt.Printf("add_int8 1%s126 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(126); got != 127 {
		fmt.Printf("add_int8 126%s1 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_1_int8_ssa(127); got != -128 {
		fmt.Printf("add_int8 1%s127 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_1_ssa(127); got != -128 {
		fmt.Printf("add_int8 127%s1 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(-128); got != -2 {
		fmt.Printf("add_int8 126%s-128 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(-128); got != -2 {
		fmt.Printf("add_int8 -128%s126 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(-127); got != -1 {
		fmt.Printf("add_int8 126%s-127 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(-127); got != -1 {
		fmt.Printf("add_int8 -127%s126 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(-1); got != 125 {
		fmt.Printf("add_int8 126%s-1 = %d, wanted 125\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(-1); got != 125 {
		fmt.Printf("add_int8 -1%s126 = %d, wanted 125\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(0); got != 126 {
		fmt.Printf("add_int8 126%s0 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(0); got != 126 {
		fmt.Printf("add_int8 0%s126 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(1); got != 127 {
		fmt.Printf("add_int8 126%s1 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(1); got != 127 {
		fmt.Printf("add_int8 1%s126 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(126); got != -4 {
		fmt.Printf("add_int8 126%s126 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(126); got != -4 {
		fmt.Printf("add_int8 126%s126 = %d, wanted -4\n", `+`, got)
		failed = true
	}

	if got := add_126_int8_ssa(127); got != -3 {
		fmt.Printf("add_int8 126%s127 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int8_126_ssa(127); got != -3 {
		fmt.Printf("add_int8 127%s126 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(-128); got != -1 {
		fmt.Printf("add_int8 127%s-128 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(-128); got != -1 {
		fmt.Printf("add_int8 -128%s127 = %d, wanted -1\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(-127); got != 0 {
		fmt.Printf("add_int8 127%s-127 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(-127); got != 0 {
		fmt.Printf("add_int8 -127%s127 = %d, wanted 0\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(-1); got != 126 {
		fmt.Printf("add_int8 127%s-1 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(-1); got != 126 {
		fmt.Printf("add_int8 -1%s127 = %d, wanted 126\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(0); got != 127 {
		fmt.Printf("add_int8 127%s0 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(0); got != 127 {
		fmt.Printf("add_int8 0%s127 = %d, wanted 127\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(1); got != -128 {
		fmt.Printf("add_int8 127%s1 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(1); got != -128 {
		fmt.Printf("add_int8 1%s127 = %d, wanted -128\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(126); got != -3 {
		fmt.Printf("add_int8 127%s126 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(126); got != -3 {
		fmt.Printf("add_int8 126%s127 = %d, wanted -3\n", `+`, got)
		failed = true
	}

	if got := add_127_int8_ssa(127); got != -2 {
		fmt.Printf("add_int8 127%s127 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := add_int8_127_ssa(127); got != -2 {
		fmt.Printf("add_int8 127%s127 = %d, wanted -2\n", `+`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(-128); got != 0 {
		fmt.Printf("sub_int8 -128%s-128 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(-128); got != 0 {
		fmt.Printf("sub_int8 -128%s-128 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(-127); got != -1 {
		fmt.Printf("sub_int8 -128%s-127 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(-127); got != 1 {
		fmt.Printf("sub_int8 -127%s-128 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(-1); got != -127 {
		fmt.Printf("sub_int8 -128%s-1 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(-1); got != 127 {
		fmt.Printf("sub_int8 -1%s-128 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(0); got != -128 {
		fmt.Printf("sub_int8 -128%s0 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(0); got != -128 {
		fmt.Printf("sub_int8 0%s-128 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(1); got != 127 {
		fmt.Printf("sub_int8 -128%s1 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(1); got != -127 {
		fmt.Printf("sub_int8 1%s-128 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(126); got != 2 {
		fmt.Printf("sub_int8 -128%s126 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(126); got != -2 {
		fmt.Printf("sub_int8 126%s-128 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg128_int8_ssa(127); got != 1 {
		fmt.Printf("sub_int8 -128%s127 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg128_ssa(127); got != -1 {
		fmt.Printf("sub_int8 127%s-128 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(-128); got != 1 {
		fmt.Printf("sub_int8 -127%s-128 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(-128); got != -1 {
		fmt.Printf("sub_int8 -128%s-127 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(-127); got != 0 {
		fmt.Printf("sub_int8 -127%s-127 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(-127); got != 0 {
		fmt.Printf("sub_int8 -127%s-127 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(-1); got != -126 {
		fmt.Printf("sub_int8 -127%s-1 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(-1); got != 126 {
		fmt.Printf("sub_int8 -1%s-127 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(0); got != -127 {
		fmt.Printf("sub_int8 -127%s0 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(0); got != 127 {
		fmt.Printf("sub_int8 0%s-127 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(1); got != -128 {
		fmt.Printf("sub_int8 -127%s1 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(1); got != -128 {
		fmt.Printf("sub_int8 1%s-127 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(126); got != 3 {
		fmt.Printf("sub_int8 -127%s126 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(126); got != -3 {
		fmt.Printf("sub_int8 126%s-127 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_Neg127_int8_ssa(127); got != 2 {
		fmt.Printf("sub_int8 -127%s127 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg127_ssa(127); got != -2 {
		fmt.Printf("sub_int8 127%s-127 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(-128); got != 127 {
		fmt.Printf("sub_int8 -1%s-128 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(-128); got != -127 {
		fmt.Printf("sub_int8 -128%s-1 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(-127); got != 126 {
		fmt.Printf("sub_int8 -1%s-127 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(-127); got != -126 {
		fmt.Printf("sub_int8 -127%s-1 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(-1); got != 0 {
		fmt.Printf("sub_int8 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(-1); got != 0 {
		fmt.Printf("sub_int8 -1%s-1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(0); got != -1 {
		fmt.Printf("sub_int8 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(0); got != 1 {
		fmt.Printf("sub_int8 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(1); got != -2 {
		fmt.Printf("sub_int8 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(1); got != 2 {
		fmt.Printf("sub_int8 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(126); got != -127 {
		fmt.Printf("sub_int8 -1%s126 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(126); got != 127 {
		fmt.Printf("sub_int8 126%s-1 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_Neg1_int8_ssa(127); got != -128 {
		fmt.Printf("sub_int8 -1%s127 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_Neg1_ssa(127); got != -128 {
		fmt.Printf("sub_int8 127%s-1 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(-128); got != -128 {
		fmt.Printf("sub_int8 0%s-128 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(-128); got != -128 {
		fmt.Printf("sub_int8 -128%s0 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(-127); got != 127 {
		fmt.Printf("sub_int8 0%s-127 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(-127); got != -127 {
		fmt.Printf("sub_int8 -127%s0 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(-1); got != 1 {
		fmt.Printf("sub_int8 0%s-1 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(-1); got != -1 {
		fmt.Printf("sub_int8 -1%s0 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(0); got != 0 {
		fmt.Printf("sub_int8 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(0); got != 0 {
		fmt.Printf("sub_int8 0%s0 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(1); got != -1 {
		fmt.Printf("sub_int8 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(1); got != 1 {
		fmt.Printf("sub_int8 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(126); got != -126 {
		fmt.Printf("sub_int8 0%s126 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(126); got != 126 {
		fmt.Printf("sub_int8 126%s0 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_0_int8_ssa(127); got != -127 {
		fmt.Printf("sub_int8 0%s127 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_0_ssa(127); got != 127 {
		fmt.Printf("sub_int8 127%s0 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(-128); got != -127 {
		fmt.Printf("sub_int8 1%s-128 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(-128); got != 127 {
		fmt.Printf("sub_int8 -128%s1 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(-127); got != -128 {
		fmt.Printf("sub_int8 1%s-127 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(-127); got != -128 {
		fmt.Printf("sub_int8 -127%s1 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(-1); got != 2 {
		fmt.Printf("sub_int8 1%s-1 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(-1); got != -2 {
		fmt.Printf("sub_int8 -1%s1 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(0); got != 1 {
		fmt.Printf("sub_int8 1%s0 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(0); got != -1 {
		fmt.Printf("sub_int8 0%s1 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(1); got != 0 {
		fmt.Printf("sub_int8 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(1); got != 0 {
		fmt.Printf("sub_int8 1%s1 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(126); got != -125 {
		fmt.Printf("sub_int8 1%s126 = %d, wanted -125\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(126); got != 125 {
		fmt.Printf("sub_int8 126%s1 = %d, wanted 125\n", `-`, got)
		failed = true
	}

	if got := sub_1_int8_ssa(127); got != -126 {
		fmt.Printf("sub_int8 1%s127 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_1_ssa(127); got != 126 {
		fmt.Printf("sub_int8 127%s1 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(-128); got != -2 {
		fmt.Printf("sub_int8 126%s-128 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(-128); got != 2 {
		fmt.Printf("sub_int8 -128%s126 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(-127); got != -3 {
		fmt.Printf("sub_int8 126%s-127 = %d, wanted -3\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(-127); got != 3 {
		fmt.Printf("sub_int8 -127%s126 = %d, wanted 3\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(-1); got != 127 {
		fmt.Printf("sub_int8 126%s-1 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(-1); got != -127 {
		fmt.Printf("sub_int8 -1%s126 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(0); got != 126 {
		fmt.Printf("sub_int8 126%s0 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(0); got != -126 {
		fmt.Printf("sub_int8 0%s126 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(1); got != 125 {
		fmt.Printf("sub_int8 126%s1 = %d, wanted 125\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(1); got != -125 {
		fmt.Printf("sub_int8 1%s126 = %d, wanted -125\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(126); got != 0 {
		fmt.Printf("sub_int8 126%s126 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(126); got != 0 {
		fmt.Printf("sub_int8 126%s126 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_126_int8_ssa(127); got != -1 {
		fmt.Printf("sub_int8 126%s127 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_126_ssa(127); got != 1 {
		fmt.Printf("sub_int8 127%s126 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(-128); got != -1 {
		fmt.Printf("sub_int8 127%s-128 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(-128); got != 1 {
		fmt.Printf("sub_int8 -128%s127 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(-127); got != -2 {
		fmt.Printf("sub_int8 127%s-127 = %d, wanted -2\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(-127); got != 2 {
		fmt.Printf("sub_int8 -127%s127 = %d, wanted 2\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(-1); got != -128 {
		fmt.Printf("sub_int8 127%s-1 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(-1); got != -128 {
		fmt.Printf("sub_int8 -1%s127 = %d, wanted -128\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(0); got != 127 {
		fmt.Printf("sub_int8 127%s0 = %d, wanted 127\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(0); got != -127 {
		fmt.Printf("sub_int8 0%s127 = %d, wanted -127\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(1); got != 126 {
		fmt.Printf("sub_int8 127%s1 = %d, wanted 126\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(1); got != -126 {
		fmt.Printf("sub_int8 1%s127 = %d, wanted -126\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(126); got != 1 {
		fmt.Printf("sub_int8 127%s126 = %d, wanted 1\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(126); got != -1 {
		fmt.Printf("sub_int8 126%s127 = %d, wanted -1\n", `-`, got)
		failed = true
	}

	if got := sub_127_int8_ssa(127); got != 0 {
		fmt.Printf("sub_int8 127%s127 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := sub_int8_127_ssa(127); got != 0 {
		fmt.Printf("sub_int8 127%s127 = %d, wanted 0\n", `-`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(-128); got != 1 {
		fmt.Printf("div_int8 -128%s-128 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(-128); got != 1 {
		fmt.Printf("div_int8 -128%s-128 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(-127); got != 1 {
		fmt.Printf("div_int8 -128%s-127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(-127); got != 0 {
		fmt.Printf("div_int8 -127%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(-1); got != -128 {
		fmt.Printf("div_int8 -128%s-1 = %d, wanted -128\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(-1); got != 0 {
		fmt.Printf("div_int8 -1%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(1); got != -128 {
		fmt.Printf("div_int8 -128%s1 = %d, wanted -128\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(1); got != 0 {
		fmt.Printf("div_int8 1%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(126); got != -1 {
		fmt.Printf("div_int8 -128%s126 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(126); got != 0 {
		fmt.Printf("div_int8 126%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg128_int8_ssa(127); got != -1 {
		fmt.Printf("div_int8 -128%s127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg128_ssa(127); got != 0 {
		fmt.Printf("div_int8 127%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 -127%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(-128); got != 1 {
		fmt.Printf("div_int8 -128%s-127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(-127); got != 1 {
		fmt.Printf("div_int8 -127%s-127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(-127); got != 1 {
		fmt.Printf("div_int8 -127%s-127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(-1); got != 127 {
		fmt.Printf("div_int8 -127%s-1 = %d, wanted 127\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(-1); got != 0 {
		fmt.Printf("div_int8 -1%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(1); got != -127 {
		fmt.Printf("div_int8 -127%s1 = %d, wanted -127\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(1); got != 0 {
		fmt.Printf("div_int8 1%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(126); got != -1 {
		fmt.Printf("div_int8 -127%s126 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(126); got != 0 {
		fmt.Printf("div_int8 126%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg127_int8_ssa(127); got != -1 {
		fmt.Printf("div_int8 -127%s127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg127_ssa(127); got != -1 {
		fmt.Printf("div_int8 127%s-127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 -1%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(-128); got != -128 {
		fmt.Printf("div_int8 -128%s-1 = %d, wanted -128\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(-127); got != 0 {
		fmt.Printf("div_int8 -1%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(-127); got != 127 {
		fmt.Printf("div_int8 -127%s-1 = %d, wanted 127\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(-1); got != 1 {
		fmt.Printf("div_int8 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(-1); got != 1 {
		fmt.Printf("div_int8 -1%s-1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(1); got != -1 {
		fmt.Printf("div_int8 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(1); got != -1 {
		fmt.Printf("div_int8 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(126); got != 0 {
		fmt.Printf("div_int8 -1%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(126); got != -126 {
		fmt.Printf("div_int8 126%s-1 = %d, wanted -126\n", `/`, got)
		failed = true
	}

	if got := div_Neg1_int8_ssa(127); got != 0 {
		fmt.Printf("div_int8 -1%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_Neg1_ssa(127); got != -127 {
		fmt.Printf("div_int8 127%s-1 = %d, wanted -127\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 0%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(-127); got != 0 {
		fmt.Printf("div_int8 0%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(-1); got != 0 {
		fmt.Printf("div_int8 0%s-1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(1); got != 0 {
		fmt.Printf("div_int8 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(126); got != 0 {
		fmt.Printf("div_int8 0%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_0_int8_ssa(127); got != 0 {
		fmt.Printf("div_int8 0%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 1%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(-128); got != -128 {
		fmt.Printf("div_int8 -128%s1 = %d, wanted -128\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(-127); got != 0 {
		fmt.Printf("div_int8 1%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(-127); got != -127 {
		fmt.Printf("div_int8 -127%s1 = %d, wanted -127\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(-1); got != -1 {
		fmt.Printf("div_int8 1%s-1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(-1); got != -1 {
		fmt.Printf("div_int8 -1%s1 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s1 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(1); got != 1 {
		fmt.Printf("div_int8 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(1); got != 1 {
		fmt.Printf("div_int8 1%s1 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(126); got != 0 {
		fmt.Printf("div_int8 1%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(126); got != 126 {
		fmt.Printf("div_int8 126%s1 = %d, wanted 126\n", `/`, got)
		failed = true
	}

	if got := div_1_int8_ssa(127); got != 0 {
		fmt.Printf("div_int8 1%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_1_ssa(127); got != 127 {
		fmt.Printf("div_int8 127%s1 = %d, wanted 127\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 126%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(-128); got != -1 {
		fmt.Printf("div_int8 -128%s126 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(-127); got != 0 {
		fmt.Printf("div_int8 126%s-127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(-127); got != -1 {
		fmt.Printf("div_int8 -127%s126 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(-1); got != -126 {
		fmt.Printf("div_int8 126%s-1 = %d, wanted -126\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(-1); got != 0 {
		fmt.Printf("div_int8 -1%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(1); got != 126 {
		fmt.Printf("div_int8 126%s1 = %d, wanted 126\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(1); got != 0 {
		fmt.Printf("div_int8 1%s126 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(126); got != 1 {
		fmt.Printf("div_int8 126%s126 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(126); got != 1 {
		fmt.Printf("div_int8 126%s126 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_126_int8_ssa(127); got != 0 {
		fmt.Printf("div_int8 126%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_126_ssa(127); got != 1 {
		fmt.Printf("div_int8 127%s126 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(-128); got != 0 {
		fmt.Printf("div_int8 127%s-128 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(-128); got != -1 {
		fmt.Printf("div_int8 -128%s127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(-127); got != -1 {
		fmt.Printf("div_int8 127%s-127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(-127); got != -1 {
		fmt.Printf("div_int8 -127%s127 = %d, wanted -1\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(-1); got != -127 {
		fmt.Printf("div_int8 127%s-1 = %d, wanted -127\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(-1); got != 0 {
		fmt.Printf("div_int8 -1%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(0); got != 0 {
		fmt.Printf("div_int8 0%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(1); got != 127 {
		fmt.Printf("div_int8 127%s1 = %d, wanted 127\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(1); got != 0 {
		fmt.Printf("div_int8 1%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(126); got != 1 {
		fmt.Printf("div_int8 127%s126 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(126); got != 0 {
		fmt.Printf("div_int8 126%s127 = %d, wanted 0\n", `/`, got)
		failed = true
	}

	if got := div_127_int8_ssa(127); got != 1 {
		fmt.Printf("div_int8 127%s127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := div_int8_127_ssa(127); got != 1 {
		fmt.Printf("div_int8 127%s127 = %d, wanted 1\n", `/`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 -128%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 -128%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(-127); got != -128 {
		fmt.Printf("mul_int8 -128%s-127 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(-127); got != -128 {
		fmt.Printf("mul_int8 -127%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(-1); got != -128 {
		fmt.Printf("mul_int8 -128%s-1 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(-1); got != -128 {
		fmt.Printf("mul_int8 -1%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 -128%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(1); got != -128 {
		fmt.Printf("mul_int8 -128%s1 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(1); got != -128 {
		fmt.Printf("mul_int8 1%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(126); got != 0 {
		fmt.Printf("mul_int8 -128%s126 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(126); got != 0 {
		fmt.Printf("mul_int8 126%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg128_int8_ssa(127); got != -128 {
		fmt.Printf("mul_int8 -128%s127 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg128_ssa(127); got != -128 {
		fmt.Printf("mul_int8 127%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -127%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -128%s-127 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(-127); got != 1 {
		fmt.Printf("mul_int8 -127%s-127 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(-127); got != 1 {
		fmt.Printf("mul_int8 -127%s-127 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(-1); got != 127 {
		fmt.Printf("mul_int8 -127%s-1 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(-1); got != 127 {
		fmt.Printf("mul_int8 -1%s-127 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 -127%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s-127 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(1); got != -127 {
		fmt.Printf("mul_int8 -127%s1 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(1); got != -127 {
		fmt.Printf("mul_int8 1%s-127 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(126); got != 126 {
		fmt.Printf("mul_int8 -127%s126 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(126); got != 126 {
		fmt.Printf("mul_int8 126%s-127 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_Neg127_int8_ssa(127); got != -1 {
		fmt.Printf("mul_int8 -127%s127 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg127_ssa(127); got != -1 {
		fmt.Printf("mul_int8 127%s-127 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -1%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -128%s-1 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(-127); got != 127 {
		fmt.Printf("mul_int8 -1%s-127 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(-127); got != 127 {
		fmt.Printf("mul_int8 -127%s-1 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(-1); got != 1 {
		fmt.Printf("mul_int8 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(-1); got != 1 {
		fmt.Printf("mul_int8 -1%s-1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(1); got != -1 {
		fmt.Printf("mul_int8 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(1); got != -1 {
		fmt.Printf("mul_int8 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(126); got != -126 {
		fmt.Printf("mul_int8 -1%s126 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(126); got != -126 {
		fmt.Printf("mul_int8 126%s-1 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_Neg1_int8_ssa(127); got != -127 {
		fmt.Printf("mul_int8 -1%s127 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_Neg1_ssa(127); got != -127 {
		fmt.Printf("mul_int8 127%s-1 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 0%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 -128%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(-127); got != 0 {
		fmt.Printf("mul_int8 0%s-127 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(-127); got != 0 {
		fmt.Printf("mul_int8 -127%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(-1); got != 0 {
		fmt.Printf("mul_int8 0%s-1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(-1); got != 0 {
		fmt.Printf("mul_int8 -1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(1); got != 0 {
		fmt.Printf("mul_int8 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(1); got != 0 {
		fmt.Printf("mul_int8 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(126); got != 0 {
		fmt.Printf("mul_int8 0%s126 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(126); got != 0 {
		fmt.Printf("mul_int8 126%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_0_int8_ssa(127); got != 0 {
		fmt.Printf("mul_int8 0%s127 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_0_ssa(127); got != 0 {
		fmt.Printf("mul_int8 127%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 1%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -128%s1 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(-127); got != -127 {
		fmt.Printf("mul_int8 1%s-127 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(-127); got != -127 {
		fmt.Printf("mul_int8 -127%s1 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(-1); got != -1 {
		fmt.Printf("mul_int8 1%s-1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(-1); got != -1 {
		fmt.Printf("mul_int8 -1%s1 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 1%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s1 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(1); got != 1 {
		fmt.Printf("mul_int8 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(1); got != 1 {
		fmt.Printf("mul_int8 1%s1 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(126); got != 126 {
		fmt.Printf("mul_int8 1%s126 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(126); got != 126 {
		fmt.Printf("mul_int8 126%s1 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_1_int8_ssa(127); got != 127 {
		fmt.Printf("mul_int8 1%s127 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_1_ssa(127); got != 127 {
		fmt.Printf("mul_int8 127%s1 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 126%s-128 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(-128); got != 0 {
		fmt.Printf("mul_int8 -128%s126 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(-127); got != 126 {
		fmt.Printf("mul_int8 126%s-127 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(-127); got != 126 {
		fmt.Printf("mul_int8 -127%s126 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(-1); got != -126 {
		fmt.Printf("mul_int8 126%s-1 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(-1); got != -126 {
		fmt.Printf("mul_int8 -1%s126 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 126%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s126 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(1); got != 126 {
		fmt.Printf("mul_int8 126%s1 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(1); got != 126 {
		fmt.Printf("mul_int8 1%s126 = %d, wanted 126\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(126); got != 4 {
		fmt.Printf("mul_int8 126%s126 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(126); got != 4 {
		fmt.Printf("mul_int8 126%s126 = %d, wanted 4\n", `*`, got)
		failed = true
	}

	if got := mul_126_int8_ssa(127); got != -126 {
		fmt.Printf("mul_int8 126%s127 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_126_ssa(127); got != -126 {
		fmt.Printf("mul_int8 127%s126 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 127%s-128 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(-128); got != -128 {
		fmt.Printf("mul_int8 -128%s127 = %d, wanted -128\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(-127); got != -1 {
		fmt.Printf("mul_int8 127%s-127 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(-127); got != -1 {
		fmt.Printf("mul_int8 -127%s127 = %d, wanted -1\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(-1); got != -127 {
		fmt.Printf("mul_int8 127%s-1 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(-1); got != -127 {
		fmt.Printf("mul_int8 -1%s127 = %d, wanted -127\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(0); got != 0 {
		fmt.Printf("mul_int8 127%s0 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(0); got != 0 {
		fmt.Printf("mul_int8 0%s127 = %d, wanted 0\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(1); got != 127 {
		fmt.Printf("mul_int8 127%s1 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(1); got != 127 {
		fmt.Printf("mul_int8 1%s127 = %d, wanted 127\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(126); got != -126 {
		fmt.Printf("mul_int8 127%s126 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(126); got != -126 {
		fmt.Printf("mul_int8 126%s127 = %d, wanted -126\n", `*`, got)
		failed = true
	}

	if got := mul_127_int8_ssa(127); got != 1 {
		fmt.Printf("mul_int8 127%s127 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mul_int8_127_ssa(127); got != 1 {
		fmt.Printf("mul_int8 127%s127 = %d, wanted 1\n", `*`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(-128); got != 0 {
		fmt.Printf("mod_int8 -128%s-128 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(-128); got != 0 {
		fmt.Printf("mod_int8 -128%s-128 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(-127); got != -1 {
		fmt.Printf("mod_int8 -128%s-127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(-127); got != -127 {
		fmt.Printf("mod_int8 -127%s-128 = %d, wanted -127\n", `%`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 -128%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(-1); got != -1 {
		fmt.Printf("mod_int8 -1%s-128 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s-128 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 -128%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(1); got != 1 {
		fmt.Printf("mod_int8 1%s-128 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(126); got != -2 {
		fmt.Printf("mod_int8 -128%s126 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(126); got != 126 {
		fmt.Printf("mod_int8 126%s-128 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_Neg128_int8_ssa(127); got != -1 {
		fmt.Printf("mod_int8 -128%s127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg128_ssa(127); got != 127 {
		fmt.Printf("mod_int8 127%s-128 = %d, wanted 127\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(-128); got != -127 {
		fmt.Printf("mod_int8 -127%s-128 = %d, wanted -127\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(-128); got != -1 {
		fmt.Printf("mod_int8 -128%s-127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 -127%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 -127%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 -127%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(-1); got != -1 {
		fmt.Printf("mod_int8 -1%s-127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 -127%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(1); got != 1 {
		fmt.Printf("mod_int8 1%s-127 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(126); got != -1 {
		fmt.Printf("mod_int8 -127%s126 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(126); got != 126 {
		fmt.Printf("mod_int8 126%s-127 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_Neg127_int8_ssa(127); got != 0 {
		fmt.Printf("mod_int8 -127%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg127_ssa(127); got != 0 {
		fmt.Printf("mod_int8 127%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(-128); got != -1 {
		fmt.Printf("mod_int8 -1%s-128 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(-128); got != 0 {
		fmt.Printf("mod_int8 -128%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(-127); got != -1 {
		fmt.Printf("mod_int8 -1%s-127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 -127%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 -1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(1); got != 0 {
		fmt.Printf("mod_int8 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(126); got != -1 {
		fmt.Printf("mod_int8 -1%s126 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(126); got != 0 {
		fmt.Printf("mod_int8 126%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_Neg1_int8_ssa(127); got != -1 {
		fmt.Printf("mod_int8 -1%s127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_Neg1_ssa(127); got != 0 {
		fmt.Printf("mod_int8 127%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(-128); got != 0 {
		fmt.Printf("mod_int8 0%s-128 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 0%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 0%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(126); got != 0 {
		fmt.Printf("mod_int8 0%s126 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_0_int8_ssa(127); got != 0 {
		fmt.Printf("mod_int8 0%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(-128); got != 1 {
		fmt.Printf("mod_int8 1%s-128 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(-128); got != 0 {
		fmt.Printf("mod_int8 -128%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(-127); got != 1 {
		fmt.Printf("mod_int8 1%s-127 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 -127%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 1%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 -1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(1); got != 0 {
		fmt.Printf("mod_int8 1%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(126); got != 1 {
		fmt.Printf("mod_int8 1%s126 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(126); got != 0 {
		fmt.Printf("mod_int8 126%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_1_int8_ssa(127); got != 1 {
		fmt.Printf("mod_int8 1%s127 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_1_ssa(127); got != 0 {
		fmt.Printf("mod_int8 127%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(-128); got != 126 {
		fmt.Printf("mod_int8 126%s-128 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(-128); got != -2 {
		fmt.Printf("mod_int8 -128%s126 = %d, wanted -2\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(-127); got != 126 {
		fmt.Printf("mod_int8 126%s-127 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(-127); got != -1 {
		fmt.Printf("mod_int8 -127%s126 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 126%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(-1); got != -1 {
		fmt.Printf("mod_int8 -1%s126 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s126 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 126%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(1); got != 1 {
		fmt.Printf("mod_int8 1%s126 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(126); got != 0 {
		fmt.Printf("mod_int8 126%s126 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(126); got != 0 {
		fmt.Printf("mod_int8 126%s126 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_126_int8_ssa(127); got != 126 {
		fmt.Printf("mod_int8 126%s127 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_int8_126_ssa(127); got != 1 {
		fmt.Printf("mod_int8 127%s126 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(-128); got != 127 {
		fmt.Printf("mod_int8 127%s-128 = %d, wanted 127\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(-128); got != -1 {
		fmt.Printf("mod_int8 -128%s127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 127%s-127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(-127); got != 0 {
		fmt.Printf("mod_int8 -127%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(-1); got != 0 {
		fmt.Printf("mod_int8 127%s-1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(-1); got != -1 {
		fmt.Printf("mod_int8 -1%s127 = %d, wanted -1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(0); got != 0 {
		fmt.Printf("mod_int8 0%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(1); got != 0 {
		fmt.Printf("mod_int8 127%s1 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(1); got != 1 {
		fmt.Printf("mod_int8 1%s127 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(126); got != 1 {
		fmt.Printf("mod_int8 127%s126 = %d, wanted 1\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(126); got != 126 {
		fmt.Printf("mod_int8 126%s127 = %d, wanted 126\n", `%`, got)
		failed = true
	}

	if got := mod_127_int8_ssa(127); got != 0 {
		fmt.Printf("mod_int8 127%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}

	if got := mod_int8_127_ssa(127); got != 0 {
		fmt.Printf("mod_int8 127%s127 = %d, wanted 0\n", `%`, got)
		failed = true
	}
	if failed {
		panic("tests failed")
	}
}
