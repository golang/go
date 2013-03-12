// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test stack split logic by calling functions of every frame size
// from near 0 up to and beyond the default segment size (4k).
// Each of those functions reports its SP + stack limit, and then
// the test (the caller) checks that those make sense.  By not
// doing the actual checking and reporting from the suspect functions,
// we minimize the possibility of crashes during the test itself.
//
// Exhaustive test for http://golang.org/issue/3310.
// The linker used to get a few sizes near the segment size wrong:
//
// --- FAIL: TestStackSplit (0.01 seconds)
// 	stack_test.go:22: after runtime_test.stack3812: sp=0x7f7818d5d078 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3816: sp=0x7f7818d5d078 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3820: sp=0x7f7818d5d070 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3824: sp=0x7f7818d5d070 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3828: sp=0x7f7818d5d068 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3832: sp=0x7f7818d5d068 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3836: sp=0x7f7818d5d060 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3840: sp=0x7f7818d5d060 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3844: sp=0x7f7818d5d058 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3848: sp=0x7f7818d5d058 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3852: sp=0x7f7818d5d050 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3856: sp=0x7f7818d5d050 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3860: sp=0x7f7818d5d048 < limit=0x7f7818d5d080
// 	stack_test.go:22: after runtime_test.stack3864: sp=0x7f7818d5d048 < limit=0x7f7818d5d080
// FAIL

package runtime_test

import (
	. "runtime"
	"testing"
	"time"
	"unsafe"
)

// See stack.h.
const (
	StackGuard = 256
	StackLimit = 128
)

func TestStackSplit(t *testing.T) {
	for _, f := range splitTests {
		sp, guard := f()
		bottom := guard - StackGuard
		if sp < bottom+StackLimit {
			fun := FuncForPC(*(*uintptr)(unsafe.Pointer(&f)))
			t.Errorf("after %s: sp=%#x < limit=%#x (guard=%#x, bottom=%#x)",
				fun.Name(), sp, bottom+StackLimit, guard, bottom)
		}
	}
}

var splitTests = []func() (uintptr, uintptr){
	// Edit .+1,/^}/-1|seq 4 4 5000 | sed 's/.*/	stack&,/' | fmt
	stack4, stack8, stack12, stack16, stack20, stack24, stack28,
	stack32, stack36, stack40, stack44, stack48, stack52, stack56,
	stack60, stack64, stack68, stack72, stack76, stack80, stack84,
	stack88, stack92, stack96, stack100, stack104, stack108, stack112,
	stack116, stack120, stack124, stack128, stack132, stack136,
	stack140, stack144, stack148, stack152, stack156, stack160,
	stack164, stack168, stack172, stack176, stack180, stack184,
	stack188, stack192, stack196, stack200, stack204, stack208,
	stack212, stack216, stack220, stack224, stack228, stack232,
	stack236, stack240, stack244, stack248, stack252, stack256,
	stack260, stack264, stack268, stack272, stack276, stack280,
	stack284, stack288, stack292, stack296, stack300, stack304,
	stack308, stack312, stack316, stack320, stack324, stack328,
	stack332, stack336, stack340, stack344, stack348, stack352,
	stack356, stack360, stack364, stack368, stack372, stack376,
	stack380, stack384, stack388, stack392, stack396, stack400,
	stack404, stack408, stack412, stack416, stack420, stack424,
	stack428, stack432, stack436, stack440, stack444, stack448,
	stack452, stack456, stack460, stack464, stack468, stack472,
	stack476, stack480, stack484, stack488, stack492, stack496,
	stack500, stack504, stack508, stack512, stack516, stack520,
	stack524, stack528, stack532, stack536, stack540, stack544,
	stack548, stack552, stack556, stack560, stack564, stack568,
	stack572, stack576, stack580, stack584, stack588, stack592,
	stack596, stack600, stack604, stack608, stack612, stack616,
	stack620, stack624, stack628, stack632, stack636, stack640,
	stack644, stack648, stack652, stack656, stack660, stack664,
	stack668, stack672, stack676, stack680, stack684, stack688,
	stack692, stack696, stack700, stack704, stack708, stack712,
	stack716, stack720, stack724, stack728, stack732, stack736,
	stack740, stack744, stack748, stack752, stack756, stack760,
	stack764, stack768, stack772, stack776, stack780, stack784,
	stack788, stack792, stack796, stack800, stack804, stack808,
	stack812, stack816, stack820, stack824, stack828, stack832,
	stack836, stack840, stack844, stack848, stack852, stack856,
	stack860, stack864, stack868, stack872, stack876, stack880,
	stack884, stack888, stack892, stack896, stack900, stack904,
	stack908, stack912, stack916, stack920, stack924, stack928,
	stack932, stack936, stack940, stack944, stack948, stack952,
	stack956, stack960, stack964, stack968, stack972, stack976,
	stack980, stack984, stack988, stack992, stack996, stack1000,
	stack1004, stack1008, stack1012, stack1016, stack1020, stack1024,
	stack1028, stack1032, stack1036, stack1040, stack1044, stack1048,
	stack1052, stack1056, stack1060, stack1064, stack1068, stack1072,
	stack1076, stack1080, stack1084, stack1088, stack1092, stack1096,
	stack1100, stack1104, stack1108, stack1112, stack1116, stack1120,
	stack1124, stack1128, stack1132, stack1136, stack1140, stack1144,
	stack1148, stack1152, stack1156, stack1160, stack1164, stack1168,
	stack1172, stack1176, stack1180, stack1184, stack1188, stack1192,
	stack1196, stack1200, stack1204, stack1208, stack1212, stack1216,
	stack1220, stack1224, stack1228, stack1232, stack1236, stack1240,
	stack1244, stack1248, stack1252, stack1256, stack1260, stack1264,
	stack1268, stack1272, stack1276, stack1280, stack1284, stack1288,
	stack1292, stack1296, stack1300, stack1304, stack1308, stack1312,
	stack1316, stack1320, stack1324, stack1328, stack1332, stack1336,
	stack1340, stack1344, stack1348, stack1352, stack1356, stack1360,
	stack1364, stack1368, stack1372, stack1376, stack1380, stack1384,
	stack1388, stack1392, stack1396, stack1400, stack1404, stack1408,
	stack1412, stack1416, stack1420, stack1424, stack1428, stack1432,
	stack1436, stack1440, stack1444, stack1448, stack1452, stack1456,
	stack1460, stack1464, stack1468, stack1472, stack1476, stack1480,
	stack1484, stack1488, stack1492, stack1496, stack1500, stack1504,
	stack1508, stack1512, stack1516, stack1520, stack1524, stack1528,
	stack1532, stack1536, stack1540, stack1544, stack1548, stack1552,
	stack1556, stack1560, stack1564, stack1568, stack1572, stack1576,
	stack1580, stack1584, stack1588, stack1592, stack1596, stack1600,
	stack1604, stack1608, stack1612, stack1616, stack1620, stack1624,
	stack1628, stack1632, stack1636, stack1640, stack1644, stack1648,
	stack1652, stack1656, stack1660, stack1664, stack1668, stack1672,
	stack1676, stack1680, stack1684, stack1688, stack1692, stack1696,
	stack1700, stack1704, stack1708, stack1712, stack1716, stack1720,
	stack1724, stack1728, stack1732, stack1736, stack1740, stack1744,
	stack1748, stack1752, stack1756, stack1760, stack1764, stack1768,
	stack1772, stack1776, stack1780, stack1784, stack1788, stack1792,
	stack1796, stack1800, stack1804, stack1808, stack1812, stack1816,
	stack1820, stack1824, stack1828, stack1832, stack1836, stack1840,
	stack1844, stack1848, stack1852, stack1856, stack1860, stack1864,
	stack1868, stack1872, stack1876, stack1880, stack1884, stack1888,
	stack1892, stack1896, stack1900, stack1904, stack1908, stack1912,
	stack1916, stack1920, stack1924, stack1928, stack1932, stack1936,
	stack1940, stack1944, stack1948, stack1952, stack1956, stack1960,
	stack1964, stack1968, stack1972, stack1976, stack1980, stack1984,
	stack1988, stack1992, stack1996, stack2000, stack2004, stack2008,
	stack2012, stack2016, stack2020, stack2024, stack2028, stack2032,
	stack2036, stack2040, stack2044, stack2048, stack2052, stack2056,
	stack2060, stack2064, stack2068, stack2072, stack2076, stack2080,
	stack2084, stack2088, stack2092, stack2096, stack2100, stack2104,
	stack2108, stack2112, stack2116, stack2120, stack2124, stack2128,
	stack2132, stack2136, stack2140, stack2144, stack2148, stack2152,
	stack2156, stack2160, stack2164, stack2168, stack2172, stack2176,
	stack2180, stack2184, stack2188, stack2192, stack2196, stack2200,
	stack2204, stack2208, stack2212, stack2216, stack2220, stack2224,
	stack2228, stack2232, stack2236, stack2240, stack2244, stack2248,
	stack2252, stack2256, stack2260, stack2264, stack2268, stack2272,
	stack2276, stack2280, stack2284, stack2288, stack2292, stack2296,
	stack2300, stack2304, stack2308, stack2312, stack2316, stack2320,
	stack2324, stack2328, stack2332, stack2336, stack2340, stack2344,
	stack2348, stack2352, stack2356, stack2360, stack2364, stack2368,
	stack2372, stack2376, stack2380, stack2384, stack2388, stack2392,
	stack2396, stack2400, stack2404, stack2408, stack2412, stack2416,
	stack2420, stack2424, stack2428, stack2432, stack2436, stack2440,
	stack2444, stack2448, stack2452, stack2456, stack2460, stack2464,
	stack2468, stack2472, stack2476, stack2480, stack2484, stack2488,
	stack2492, stack2496, stack2500, stack2504, stack2508, stack2512,
	stack2516, stack2520, stack2524, stack2528, stack2532, stack2536,
	stack2540, stack2544, stack2548, stack2552, stack2556, stack2560,
	stack2564, stack2568, stack2572, stack2576, stack2580, stack2584,
	stack2588, stack2592, stack2596, stack2600, stack2604, stack2608,
	stack2612, stack2616, stack2620, stack2624, stack2628, stack2632,
	stack2636, stack2640, stack2644, stack2648, stack2652, stack2656,
	stack2660, stack2664, stack2668, stack2672, stack2676, stack2680,
	stack2684, stack2688, stack2692, stack2696, stack2700, stack2704,
	stack2708, stack2712, stack2716, stack2720, stack2724, stack2728,
	stack2732, stack2736, stack2740, stack2744, stack2748, stack2752,
	stack2756, stack2760, stack2764, stack2768, stack2772, stack2776,
	stack2780, stack2784, stack2788, stack2792, stack2796, stack2800,
	stack2804, stack2808, stack2812, stack2816, stack2820, stack2824,
	stack2828, stack2832, stack2836, stack2840, stack2844, stack2848,
	stack2852, stack2856, stack2860, stack2864, stack2868, stack2872,
	stack2876, stack2880, stack2884, stack2888, stack2892, stack2896,
	stack2900, stack2904, stack2908, stack2912, stack2916, stack2920,
	stack2924, stack2928, stack2932, stack2936, stack2940, stack2944,
	stack2948, stack2952, stack2956, stack2960, stack2964, stack2968,
	stack2972, stack2976, stack2980, stack2984, stack2988, stack2992,
	stack2996, stack3000, stack3004, stack3008, stack3012, stack3016,
	stack3020, stack3024, stack3028, stack3032, stack3036, stack3040,
	stack3044, stack3048, stack3052, stack3056, stack3060, stack3064,
	stack3068, stack3072, stack3076, stack3080, stack3084, stack3088,
	stack3092, stack3096, stack3100, stack3104, stack3108, stack3112,
	stack3116, stack3120, stack3124, stack3128, stack3132, stack3136,
	stack3140, stack3144, stack3148, stack3152, stack3156, stack3160,
	stack3164, stack3168, stack3172, stack3176, stack3180, stack3184,
	stack3188, stack3192, stack3196, stack3200, stack3204, stack3208,
	stack3212, stack3216, stack3220, stack3224, stack3228, stack3232,
	stack3236, stack3240, stack3244, stack3248, stack3252, stack3256,
	stack3260, stack3264, stack3268, stack3272, stack3276, stack3280,
	stack3284, stack3288, stack3292, stack3296, stack3300, stack3304,
	stack3308, stack3312, stack3316, stack3320, stack3324, stack3328,
	stack3332, stack3336, stack3340, stack3344, stack3348, stack3352,
	stack3356, stack3360, stack3364, stack3368, stack3372, stack3376,
	stack3380, stack3384, stack3388, stack3392, stack3396, stack3400,
	stack3404, stack3408, stack3412, stack3416, stack3420, stack3424,
	stack3428, stack3432, stack3436, stack3440, stack3444, stack3448,
	stack3452, stack3456, stack3460, stack3464, stack3468, stack3472,
	stack3476, stack3480, stack3484, stack3488, stack3492, stack3496,
	stack3500, stack3504, stack3508, stack3512, stack3516, stack3520,
	stack3524, stack3528, stack3532, stack3536, stack3540, stack3544,
	stack3548, stack3552, stack3556, stack3560, stack3564, stack3568,
	stack3572, stack3576, stack3580, stack3584, stack3588, stack3592,
	stack3596, stack3600, stack3604, stack3608, stack3612, stack3616,
	stack3620, stack3624, stack3628, stack3632, stack3636, stack3640,
	stack3644, stack3648, stack3652, stack3656, stack3660, stack3664,
	stack3668, stack3672, stack3676, stack3680, stack3684, stack3688,
	stack3692, stack3696, stack3700, stack3704, stack3708, stack3712,
	stack3716, stack3720, stack3724, stack3728, stack3732, stack3736,
	stack3740, stack3744, stack3748, stack3752, stack3756, stack3760,
	stack3764, stack3768, stack3772, stack3776, stack3780, stack3784,
	stack3788, stack3792, stack3796, stack3800, stack3804, stack3808,
	stack3812, stack3816, stack3820, stack3824, stack3828, stack3832,
	stack3836, stack3840, stack3844, stack3848, stack3852, stack3856,
	stack3860, stack3864, stack3868, stack3872, stack3876, stack3880,
	stack3884, stack3888, stack3892, stack3896, stack3900, stack3904,
	stack3908, stack3912, stack3916, stack3920, stack3924, stack3928,
	stack3932, stack3936, stack3940, stack3944, stack3948, stack3952,
	stack3956, stack3960, stack3964, stack3968, stack3972, stack3976,
	stack3980, stack3984, stack3988, stack3992, stack3996, stack4000,
	stack4004, stack4008, stack4012, stack4016, stack4020, stack4024,
	stack4028, stack4032, stack4036, stack4040, stack4044, stack4048,
	stack4052, stack4056, stack4060, stack4064, stack4068, stack4072,
	stack4076, stack4080, stack4084, stack4088, stack4092, stack4096,
	stack4100, stack4104, stack4108, stack4112, stack4116, stack4120,
	stack4124, stack4128, stack4132, stack4136, stack4140, stack4144,
	stack4148, stack4152, stack4156, stack4160, stack4164, stack4168,
	stack4172, stack4176, stack4180, stack4184, stack4188, stack4192,
	stack4196, stack4200, stack4204, stack4208, stack4212, stack4216,
	stack4220, stack4224, stack4228, stack4232, stack4236, stack4240,
	stack4244, stack4248, stack4252, stack4256, stack4260, stack4264,
	stack4268, stack4272, stack4276, stack4280, stack4284, stack4288,
	stack4292, stack4296, stack4300, stack4304, stack4308, stack4312,
	stack4316, stack4320, stack4324, stack4328, stack4332, stack4336,
	stack4340, stack4344, stack4348, stack4352, stack4356, stack4360,
	stack4364, stack4368, stack4372, stack4376, stack4380, stack4384,
	stack4388, stack4392, stack4396, stack4400, stack4404, stack4408,
	stack4412, stack4416, stack4420, stack4424, stack4428, stack4432,
	stack4436, stack4440, stack4444, stack4448, stack4452, stack4456,
	stack4460, stack4464, stack4468, stack4472, stack4476, stack4480,
	stack4484, stack4488, stack4492, stack4496, stack4500, stack4504,
	stack4508, stack4512, stack4516, stack4520, stack4524, stack4528,
	stack4532, stack4536, stack4540, stack4544, stack4548, stack4552,
	stack4556, stack4560, stack4564, stack4568, stack4572, stack4576,
	stack4580, stack4584, stack4588, stack4592, stack4596, stack4600,
	stack4604, stack4608, stack4612, stack4616, stack4620, stack4624,
	stack4628, stack4632, stack4636, stack4640, stack4644, stack4648,
	stack4652, stack4656, stack4660, stack4664, stack4668, stack4672,
	stack4676, stack4680, stack4684, stack4688, stack4692, stack4696,
	stack4700, stack4704, stack4708, stack4712, stack4716, stack4720,
	stack4724, stack4728, stack4732, stack4736, stack4740, stack4744,
	stack4748, stack4752, stack4756, stack4760, stack4764, stack4768,
	stack4772, stack4776, stack4780, stack4784, stack4788, stack4792,
	stack4796, stack4800, stack4804, stack4808, stack4812, stack4816,
	stack4820, stack4824, stack4828, stack4832, stack4836, stack4840,
	stack4844, stack4848, stack4852, stack4856, stack4860, stack4864,
	stack4868, stack4872, stack4876, stack4880, stack4884, stack4888,
	stack4892, stack4896, stack4900, stack4904, stack4908, stack4912,
	stack4916, stack4920, stack4924, stack4928, stack4932, stack4936,
	stack4940, stack4944, stack4948, stack4952, stack4956, stack4960,
	stack4964, stack4968, stack4972, stack4976, stack4980, stack4984,
	stack4988, stack4992, stack4996, stack5000,
}

var Used byte

func use(buf []byte) {
	for _, c := range buf {
		Used += c
	}
}

// Edit .+1,$ | seq 4 4 5000 | sed 's/.*/func stack&()(uintptr, uintptr) { var buf [&]byte; use(buf[:]); return Stackguard() }/'
func stack4() (uintptr, uintptr)    { var buf [4]byte; use(buf[:]); return Stackguard() }
func stack8() (uintptr, uintptr)    { var buf [8]byte; use(buf[:]); return Stackguard() }
func stack12() (uintptr, uintptr)   { var buf [12]byte; use(buf[:]); return Stackguard() }
func stack16() (uintptr, uintptr)   { var buf [16]byte; use(buf[:]); return Stackguard() }
func stack20() (uintptr, uintptr)   { var buf [20]byte; use(buf[:]); return Stackguard() }
func stack24() (uintptr, uintptr)   { var buf [24]byte; use(buf[:]); return Stackguard() }
func stack28() (uintptr, uintptr)   { var buf [28]byte; use(buf[:]); return Stackguard() }
func stack32() (uintptr, uintptr)   { var buf [32]byte; use(buf[:]); return Stackguard() }
func stack36() (uintptr, uintptr)   { var buf [36]byte; use(buf[:]); return Stackguard() }
func stack40() (uintptr, uintptr)   { var buf [40]byte; use(buf[:]); return Stackguard() }
func stack44() (uintptr, uintptr)   { var buf [44]byte; use(buf[:]); return Stackguard() }
func stack48() (uintptr, uintptr)   { var buf [48]byte; use(buf[:]); return Stackguard() }
func stack52() (uintptr, uintptr)   { var buf [52]byte; use(buf[:]); return Stackguard() }
func stack56() (uintptr, uintptr)   { var buf [56]byte; use(buf[:]); return Stackguard() }
func stack60() (uintptr, uintptr)   { var buf [60]byte; use(buf[:]); return Stackguard() }
func stack64() (uintptr, uintptr)   { var buf [64]byte; use(buf[:]); return Stackguard() }
func stack68() (uintptr, uintptr)   { var buf [68]byte; use(buf[:]); return Stackguard() }
func stack72() (uintptr, uintptr)   { var buf [72]byte; use(buf[:]); return Stackguard() }
func stack76() (uintptr, uintptr)   { var buf [76]byte; use(buf[:]); return Stackguard() }
func stack80() (uintptr, uintptr)   { var buf [80]byte; use(buf[:]); return Stackguard() }
func stack84() (uintptr, uintptr)   { var buf [84]byte; use(buf[:]); return Stackguard() }
func stack88() (uintptr, uintptr)   { var buf [88]byte; use(buf[:]); return Stackguard() }
func stack92() (uintptr, uintptr)   { var buf [92]byte; use(buf[:]); return Stackguard() }
func stack96() (uintptr, uintptr)   { var buf [96]byte; use(buf[:]); return Stackguard() }
func stack100() (uintptr, uintptr)  { var buf [100]byte; use(buf[:]); return Stackguard() }
func stack104() (uintptr, uintptr)  { var buf [104]byte; use(buf[:]); return Stackguard() }
func stack108() (uintptr, uintptr)  { var buf [108]byte; use(buf[:]); return Stackguard() }
func stack112() (uintptr, uintptr)  { var buf [112]byte; use(buf[:]); return Stackguard() }
func stack116() (uintptr, uintptr)  { var buf [116]byte; use(buf[:]); return Stackguard() }
func stack120() (uintptr, uintptr)  { var buf [120]byte; use(buf[:]); return Stackguard() }
func stack124() (uintptr, uintptr)  { var buf [124]byte; use(buf[:]); return Stackguard() }
func stack128() (uintptr, uintptr)  { var buf [128]byte; use(buf[:]); return Stackguard() }
func stack132() (uintptr, uintptr)  { var buf [132]byte; use(buf[:]); return Stackguard() }
func stack136() (uintptr, uintptr)  { var buf [136]byte; use(buf[:]); return Stackguard() }
func stack140() (uintptr, uintptr)  { var buf [140]byte; use(buf[:]); return Stackguard() }
func stack144() (uintptr, uintptr)  { var buf [144]byte; use(buf[:]); return Stackguard() }
func stack148() (uintptr, uintptr)  { var buf [148]byte; use(buf[:]); return Stackguard() }
func stack152() (uintptr, uintptr)  { var buf [152]byte; use(buf[:]); return Stackguard() }
func stack156() (uintptr, uintptr)  { var buf [156]byte; use(buf[:]); return Stackguard() }
func stack160() (uintptr, uintptr)  { var buf [160]byte; use(buf[:]); return Stackguard() }
func stack164() (uintptr, uintptr)  { var buf [164]byte; use(buf[:]); return Stackguard() }
func stack168() (uintptr, uintptr)  { var buf [168]byte; use(buf[:]); return Stackguard() }
func stack172() (uintptr, uintptr)  { var buf [172]byte; use(buf[:]); return Stackguard() }
func stack176() (uintptr, uintptr)  { var buf [176]byte; use(buf[:]); return Stackguard() }
func stack180() (uintptr, uintptr)  { var buf [180]byte; use(buf[:]); return Stackguard() }
func stack184() (uintptr, uintptr)  { var buf [184]byte; use(buf[:]); return Stackguard() }
func stack188() (uintptr, uintptr)  { var buf [188]byte; use(buf[:]); return Stackguard() }
func stack192() (uintptr, uintptr)  { var buf [192]byte; use(buf[:]); return Stackguard() }
func stack196() (uintptr, uintptr)  { var buf [196]byte; use(buf[:]); return Stackguard() }
func stack200() (uintptr, uintptr)  { var buf [200]byte; use(buf[:]); return Stackguard() }
func stack204() (uintptr, uintptr)  { var buf [204]byte; use(buf[:]); return Stackguard() }
func stack208() (uintptr, uintptr)  { var buf [208]byte; use(buf[:]); return Stackguard() }
func stack212() (uintptr, uintptr)  { var buf [212]byte; use(buf[:]); return Stackguard() }
func stack216() (uintptr, uintptr)  { var buf [216]byte; use(buf[:]); return Stackguard() }
func stack220() (uintptr, uintptr)  { var buf [220]byte; use(buf[:]); return Stackguard() }
func stack224() (uintptr, uintptr)  { var buf [224]byte; use(buf[:]); return Stackguard() }
func stack228() (uintptr, uintptr)  { var buf [228]byte; use(buf[:]); return Stackguard() }
func stack232() (uintptr, uintptr)  { var buf [232]byte; use(buf[:]); return Stackguard() }
func stack236() (uintptr, uintptr)  { var buf [236]byte; use(buf[:]); return Stackguard() }
func stack240() (uintptr, uintptr)  { var buf [240]byte; use(buf[:]); return Stackguard() }
func stack244() (uintptr, uintptr)  { var buf [244]byte; use(buf[:]); return Stackguard() }
func stack248() (uintptr, uintptr)  { var buf [248]byte; use(buf[:]); return Stackguard() }
func stack252() (uintptr, uintptr)  { var buf [252]byte; use(buf[:]); return Stackguard() }
func stack256() (uintptr, uintptr)  { var buf [256]byte; use(buf[:]); return Stackguard() }
func stack260() (uintptr, uintptr)  { var buf [260]byte; use(buf[:]); return Stackguard() }
func stack264() (uintptr, uintptr)  { var buf [264]byte; use(buf[:]); return Stackguard() }
func stack268() (uintptr, uintptr)  { var buf [268]byte; use(buf[:]); return Stackguard() }
func stack272() (uintptr, uintptr)  { var buf [272]byte; use(buf[:]); return Stackguard() }
func stack276() (uintptr, uintptr)  { var buf [276]byte; use(buf[:]); return Stackguard() }
func stack280() (uintptr, uintptr)  { var buf [280]byte; use(buf[:]); return Stackguard() }
func stack284() (uintptr, uintptr)  { var buf [284]byte; use(buf[:]); return Stackguard() }
func stack288() (uintptr, uintptr)  { var buf [288]byte; use(buf[:]); return Stackguard() }
func stack292() (uintptr, uintptr)  { var buf [292]byte; use(buf[:]); return Stackguard() }
func stack296() (uintptr, uintptr)  { var buf [296]byte; use(buf[:]); return Stackguard() }
func stack300() (uintptr, uintptr)  { var buf [300]byte; use(buf[:]); return Stackguard() }
func stack304() (uintptr, uintptr)  { var buf [304]byte; use(buf[:]); return Stackguard() }
func stack308() (uintptr, uintptr)  { var buf [308]byte; use(buf[:]); return Stackguard() }
func stack312() (uintptr, uintptr)  { var buf [312]byte; use(buf[:]); return Stackguard() }
func stack316() (uintptr, uintptr)  { var buf [316]byte; use(buf[:]); return Stackguard() }
func stack320() (uintptr, uintptr)  { var buf [320]byte; use(buf[:]); return Stackguard() }
func stack324() (uintptr, uintptr)  { var buf [324]byte; use(buf[:]); return Stackguard() }
func stack328() (uintptr, uintptr)  { var buf [328]byte; use(buf[:]); return Stackguard() }
func stack332() (uintptr, uintptr)  { var buf [332]byte; use(buf[:]); return Stackguard() }
func stack336() (uintptr, uintptr)  { var buf [336]byte; use(buf[:]); return Stackguard() }
func stack340() (uintptr, uintptr)  { var buf [340]byte; use(buf[:]); return Stackguard() }
func stack344() (uintptr, uintptr)  { var buf [344]byte; use(buf[:]); return Stackguard() }
func stack348() (uintptr, uintptr)  { var buf [348]byte; use(buf[:]); return Stackguard() }
func stack352() (uintptr, uintptr)  { var buf [352]byte; use(buf[:]); return Stackguard() }
func stack356() (uintptr, uintptr)  { var buf [356]byte; use(buf[:]); return Stackguard() }
func stack360() (uintptr, uintptr)  { var buf [360]byte; use(buf[:]); return Stackguard() }
func stack364() (uintptr, uintptr)  { var buf [364]byte; use(buf[:]); return Stackguard() }
func stack368() (uintptr, uintptr)  { var buf [368]byte; use(buf[:]); return Stackguard() }
func stack372() (uintptr, uintptr)  { var buf [372]byte; use(buf[:]); return Stackguard() }
func stack376() (uintptr, uintptr)  { var buf [376]byte; use(buf[:]); return Stackguard() }
func stack380() (uintptr, uintptr)  { var buf [380]byte; use(buf[:]); return Stackguard() }
func stack384() (uintptr, uintptr)  { var buf [384]byte; use(buf[:]); return Stackguard() }
func stack388() (uintptr, uintptr)  { var buf [388]byte; use(buf[:]); return Stackguard() }
func stack392() (uintptr, uintptr)  { var buf [392]byte; use(buf[:]); return Stackguard() }
func stack396() (uintptr, uintptr)  { var buf [396]byte; use(buf[:]); return Stackguard() }
func stack400() (uintptr, uintptr)  { var buf [400]byte; use(buf[:]); return Stackguard() }
func stack404() (uintptr, uintptr)  { var buf [404]byte; use(buf[:]); return Stackguard() }
func stack408() (uintptr, uintptr)  { var buf [408]byte; use(buf[:]); return Stackguard() }
func stack412() (uintptr, uintptr)  { var buf [412]byte; use(buf[:]); return Stackguard() }
func stack416() (uintptr, uintptr)  { var buf [416]byte; use(buf[:]); return Stackguard() }
func stack420() (uintptr, uintptr)  { var buf [420]byte; use(buf[:]); return Stackguard() }
func stack424() (uintptr, uintptr)  { var buf [424]byte; use(buf[:]); return Stackguard() }
func stack428() (uintptr, uintptr)  { var buf [428]byte; use(buf[:]); return Stackguard() }
func stack432() (uintptr, uintptr)  { var buf [432]byte; use(buf[:]); return Stackguard() }
func stack436() (uintptr, uintptr)  { var buf [436]byte; use(buf[:]); return Stackguard() }
func stack440() (uintptr, uintptr)  { var buf [440]byte; use(buf[:]); return Stackguard() }
func stack444() (uintptr, uintptr)  { var buf [444]byte; use(buf[:]); return Stackguard() }
func stack448() (uintptr, uintptr)  { var buf [448]byte; use(buf[:]); return Stackguard() }
func stack452() (uintptr, uintptr)  { var buf [452]byte; use(buf[:]); return Stackguard() }
func stack456() (uintptr, uintptr)  { var buf [456]byte; use(buf[:]); return Stackguard() }
func stack460() (uintptr, uintptr)  { var buf [460]byte; use(buf[:]); return Stackguard() }
func stack464() (uintptr, uintptr)  { var buf [464]byte; use(buf[:]); return Stackguard() }
func stack468() (uintptr, uintptr)  { var buf [468]byte; use(buf[:]); return Stackguard() }
func stack472() (uintptr, uintptr)  { var buf [472]byte; use(buf[:]); return Stackguard() }
func stack476() (uintptr, uintptr)  { var buf [476]byte; use(buf[:]); return Stackguard() }
func stack480() (uintptr, uintptr)  { var buf [480]byte; use(buf[:]); return Stackguard() }
func stack484() (uintptr, uintptr)  { var buf [484]byte; use(buf[:]); return Stackguard() }
func stack488() (uintptr, uintptr)  { var buf [488]byte; use(buf[:]); return Stackguard() }
func stack492() (uintptr, uintptr)  { var buf [492]byte; use(buf[:]); return Stackguard() }
func stack496() (uintptr, uintptr)  { var buf [496]byte; use(buf[:]); return Stackguard() }
func stack500() (uintptr, uintptr)  { var buf [500]byte; use(buf[:]); return Stackguard() }
func stack504() (uintptr, uintptr)  { var buf [504]byte; use(buf[:]); return Stackguard() }
func stack508() (uintptr, uintptr)  { var buf [508]byte; use(buf[:]); return Stackguard() }
func stack512() (uintptr, uintptr)  { var buf [512]byte; use(buf[:]); return Stackguard() }
func stack516() (uintptr, uintptr)  { var buf [516]byte; use(buf[:]); return Stackguard() }
func stack520() (uintptr, uintptr)  { var buf [520]byte; use(buf[:]); return Stackguard() }
func stack524() (uintptr, uintptr)  { var buf [524]byte; use(buf[:]); return Stackguard() }
func stack528() (uintptr, uintptr)  { var buf [528]byte; use(buf[:]); return Stackguard() }
func stack532() (uintptr, uintptr)  { var buf [532]byte; use(buf[:]); return Stackguard() }
func stack536() (uintptr, uintptr)  { var buf [536]byte; use(buf[:]); return Stackguard() }
func stack540() (uintptr, uintptr)  { var buf [540]byte; use(buf[:]); return Stackguard() }
func stack544() (uintptr, uintptr)  { var buf [544]byte; use(buf[:]); return Stackguard() }
func stack548() (uintptr, uintptr)  { var buf [548]byte; use(buf[:]); return Stackguard() }
func stack552() (uintptr, uintptr)  { var buf [552]byte; use(buf[:]); return Stackguard() }
func stack556() (uintptr, uintptr)  { var buf [556]byte; use(buf[:]); return Stackguard() }
func stack560() (uintptr, uintptr)  { var buf [560]byte; use(buf[:]); return Stackguard() }
func stack564() (uintptr, uintptr)  { var buf [564]byte; use(buf[:]); return Stackguard() }
func stack568() (uintptr, uintptr)  { var buf [568]byte; use(buf[:]); return Stackguard() }
func stack572() (uintptr, uintptr)  { var buf [572]byte; use(buf[:]); return Stackguard() }
func stack576() (uintptr, uintptr)  { var buf [576]byte; use(buf[:]); return Stackguard() }
func stack580() (uintptr, uintptr)  { var buf [580]byte; use(buf[:]); return Stackguard() }
func stack584() (uintptr, uintptr)  { var buf [584]byte; use(buf[:]); return Stackguard() }
func stack588() (uintptr, uintptr)  { var buf [588]byte; use(buf[:]); return Stackguard() }
func stack592() (uintptr, uintptr)  { var buf [592]byte; use(buf[:]); return Stackguard() }
func stack596() (uintptr, uintptr)  { var buf [596]byte; use(buf[:]); return Stackguard() }
func stack600() (uintptr, uintptr)  { var buf [600]byte; use(buf[:]); return Stackguard() }
func stack604() (uintptr, uintptr)  { var buf [604]byte; use(buf[:]); return Stackguard() }
func stack608() (uintptr, uintptr)  { var buf [608]byte; use(buf[:]); return Stackguard() }
func stack612() (uintptr, uintptr)  { var buf [612]byte; use(buf[:]); return Stackguard() }
func stack616() (uintptr, uintptr)  { var buf [616]byte; use(buf[:]); return Stackguard() }
func stack620() (uintptr, uintptr)  { var buf [620]byte; use(buf[:]); return Stackguard() }
func stack624() (uintptr, uintptr)  { var buf [624]byte; use(buf[:]); return Stackguard() }
func stack628() (uintptr, uintptr)  { var buf [628]byte; use(buf[:]); return Stackguard() }
func stack632() (uintptr, uintptr)  { var buf [632]byte; use(buf[:]); return Stackguard() }
func stack636() (uintptr, uintptr)  { var buf [636]byte; use(buf[:]); return Stackguard() }
func stack640() (uintptr, uintptr)  { var buf [640]byte; use(buf[:]); return Stackguard() }
func stack644() (uintptr, uintptr)  { var buf [644]byte; use(buf[:]); return Stackguard() }
func stack648() (uintptr, uintptr)  { var buf [648]byte; use(buf[:]); return Stackguard() }
func stack652() (uintptr, uintptr)  { var buf [652]byte; use(buf[:]); return Stackguard() }
func stack656() (uintptr, uintptr)  { var buf [656]byte; use(buf[:]); return Stackguard() }
func stack660() (uintptr, uintptr)  { var buf [660]byte; use(buf[:]); return Stackguard() }
func stack664() (uintptr, uintptr)  { var buf [664]byte; use(buf[:]); return Stackguard() }
func stack668() (uintptr, uintptr)  { var buf [668]byte; use(buf[:]); return Stackguard() }
func stack672() (uintptr, uintptr)  { var buf [672]byte; use(buf[:]); return Stackguard() }
func stack676() (uintptr, uintptr)  { var buf [676]byte; use(buf[:]); return Stackguard() }
func stack680() (uintptr, uintptr)  { var buf [680]byte; use(buf[:]); return Stackguard() }
func stack684() (uintptr, uintptr)  { var buf [684]byte; use(buf[:]); return Stackguard() }
func stack688() (uintptr, uintptr)  { var buf [688]byte; use(buf[:]); return Stackguard() }
func stack692() (uintptr, uintptr)  { var buf [692]byte; use(buf[:]); return Stackguard() }
func stack696() (uintptr, uintptr)  { var buf [696]byte; use(buf[:]); return Stackguard() }
func stack700() (uintptr, uintptr)  { var buf [700]byte; use(buf[:]); return Stackguard() }
func stack704() (uintptr, uintptr)  { var buf [704]byte; use(buf[:]); return Stackguard() }
func stack708() (uintptr, uintptr)  { var buf [708]byte; use(buf[:]); return Stackguard() }
func stack712() (uintptr, uintptr)  { var buf [712]byte; use(buf[:]); return Stackguard() }
func stack716() (uintptr, uintptr)  { var buf [716]byte; use(buf[:]); return Stackguard() }
func stack720() (uintptr, uintptr)  { var buf [720]byte; use(buf[:]); return Stackguard() }
func stack724() (uintptr, uintptr)  { var buf [724]byte; use(buf[:]); return Stackguard() }
func stack728() (uintptr, uintptr)  { var buf [728]byte; use(buf[:]); return Stackguard() }
func stack732() (uintptr, uintptr)  { var buf [732]byte; use(buf[:]); return Stackguard() }
func stack736() (uintptr, uintptr)  { var buf [736]byte; use(buf[:]); return Stackguard() }
func stack740() (uintptr, uintptr)  { var buf [740]byte; use(buf[:]); return Stackguard() }
func stack744() (uintptr, uintptr)  { var buf [744]byte; use(buf[:]); return Stackguard() }
func stack748() (uintptr, uintptr)  { var buf [748]byte; use(buf[:]); return Stackguard() }
func stack752() (uintptr, uintptr)  { var buf [752]byte; use(buf[:]); return Stackguard() }
func stack756() (uintptr, uintptr)  { var buf [756]byte; use(buf[:]); return Stackguard() }
func stack760() (uintptr, uintptr)  { var buf [760]byte; use(buf[:]); return Stackguard() }
func stack764() (uintptr, uintptr)  { var buf [764]byte; use(buf[:]); return Stackguard() }
func stack768() (uintptr, uintptr)  { var buf [768]byte; use(buf[:]); return Stackguard() }
func stack772() (uintptr, uintptr)  { var buf [772]byte; use(buf[:]); return Stackguard() }
func stack776() (uintptr, uintptr)  { var buf [776]byte; use(buf[:]); return Stackguard() }
func stack780() (uintptr, uintptr)  { var buf [780]byte; use(buf[:]); return Stackguard() }
func stack784() (uintptr, uintptr)  { var buf [784]byte; use(buf[:]); return Stackguard() }
func stack788() (uintptr, uintptr)  { var buf [788]byte; use(buf[:]); return Stackguard() }
func stack792() (uintptr, uintptr)  { var buf [792]byte; use(buf[:]); return Stackguard() }
func stack796() (uintptr, uintptr)  { var buf [796]byte; use(buf[:]); return Stackguard() }
func stack800() (uintptr, uintptr)  { var buf [800]byte; use(buf[:]); return Stackguard() }
func stack804() (uintptr, uintptr)  { var buf [804]byte; use(buf[:]); return Stackguard() }
func stack808() (uintptr, uintptr)  { var buf [808]byte; use(buf[:]); return Stackguard() }
func stack812() (uintptr, uintptr)  { var buf [812]byte; use(buf[:]); return Stackguard() }
func stack816() (uintptr, uintptr)  { var buf [816]byte; use(buf[:]); return Stackguard() }
func stack820() (uintptr, uintptr)  { var buf [820]byte; use(buf[:]); return Stackguard() }
func stack824() (uintptr, uintptr)  { var buf [824]byte; use(buf[:]); return Stackguard() }
func stack828() (uintptr, uintptr)  { var buf [828]byte; use(buf[:]); return Stackguard() }
func stack832() (uintptr, uintptr)  { var buf [832]byte; use(buf[:]); return Stackguard() }
func stack836() (uintptr, uintptr)  { var buf [836]byte; use(buf[:]); return Stackguard() }
func stack840() (uintptr, uintptr)  { var buf [840]byte; use(buf[:]); return Stackguard() }
func stack844() (uintptr, uintptr)  { var buf [844]byte; use(buf[:]); return Stackguard() }
func stack848() (uintptr, uintptr)  { var buf [848]byte; use(buf[:]); return Stackguard() }
func stack852() (uintptr, uintptr)  { var buf [852]byte; use(buf[:]); return Stackguard() }
func stack856() (uintptr, uintptr)  { var buf [856]byte; use(buf[:]); return Stackguard() }
func stack860() (uintptr, uintptr)  { var buf [860]byte; use(buf[:]); return Stackguard() }
func stack864() (uintptr, uintptr)  { var buf [864]byte; use(buf[:]); return Stackguard() }
func stack868() (uintptr, uintptr)  { var buf [868]byte; use(buf[:]); return Stackguard() }
func stack872() (uintptr, uintptr)  { var buf [872]byte; use(buf[:]); return Stackguard() }
func stack876() (uintptr, uintptr)  { var buf [876]byte; use(buf[:]); return Stackguard() }
func stack880() (uintptr, uintptr)  { var buf [880]byte; use(buf[:]); return Stackguard() }
func stack884() (uintptr, uintptr)  { var buf [884]byte; use(buf[:]); return Stackguard() }
func stack888() (uintptr, uintptr)  { var buf [888]byte; use(buf[:]); return Stackguard() }
func stack892() (uintptr, uintptr)  { var buf [892]byte; use(buf[:]); return Stackguard() }
func stack896() (uintptr, uintptr)  { var buf [896]byte; use(buf[:]); return Stackguard() }
func stack900() (uintptr, uintptr)  { var buf [900]byte; use(buf[:]); return Stackguard() }
func stack904() (uintptr, uintptr)  { var buf [904]byte; use(buf[:]); return Stackguard() }
func stack908() (uintptr, uintptr)  { var buf [908]byte; use(buf[:]); return Stackguard() }
func stack912() (uintptr, uintptr)  { var buf [912]byte; use(buf[:]); return Stackguard() }
func stack916() (uintptr, uintptr)  { var buf [916]byte; use(buf[:]); return Stackguard() }
func stack920() (uintptr, uintptr)  { var buf [920]byte; use(buf[:]); return Stackguard() }
func stack924() (uintptr, uintptr)  { var buf [924]byte; use(buf[:]); return Stackguard() }
func stack928() (uintptr, uintptr)  { var buf [928]byte; use(buf[:]); return Stackguard() }
func stack932() (uintptr, uintptr)  { var buf [932]byte; use(buf[:]); return Stackguard() }
func stack936() (uintptr, uintptr)  { var buf [936]byte; use(buf[:]); return Stackguard() }
func stack940() (uintptr, uintptr)  { var buf [940]byte; use(buf[:]); return Stackguard() }
func stack944() (uintptr, uintptr)  { var buf [944]byte; use(buf[:]); return Stackguard() }
func stack948() (uintptr, uintptr)  { var buf [948]byte; use(buf[:]); return Stackguard() }
func stack952() (uintptr, uintptr)  { var buf [952]byte; use(buf[:]); return Stackguard() }
func stack956() (uintptr, uintptr)  { var buf [956]byte; use(buf[:]); return Stackguard() }
func stack960() (uintptr, uintptr)  { var buf [960]byte; use(buf[:]); return Stackguard() }
func stack964() (uintptr, uintptr)  { var buf [964]byte; use(buf[:]); return Stackguard() }
func stack968() (uintptr, uintptr)  { var buf [968]byte; use(buf[:]); return Stackguard() }
func stack972() (uintptr, uintptr)  { var buf [972]byte; use(buf[:]); return Stackguard() }
func stack976() (uintptr, uintptr)  { var buf [976]byte; use(buf[:]); return Stackguard() }
func stack980() (uintptr, uintptr)  { var buf [980]byte; use(buf[:]); return Stackguard() }
func stack984() (uintptr, uintptr)  { var buf [984]byte; use(buf[:]); return Stackguard() }
func stack988() (uintptr, uintptr)  { var buf [988]byte; use(buf[:]); return Stackguard() }
func stack992() (uintptr, uintptr)  { var buf [992]byte; use(buf[:]); return Stackguard() }
func stack996() (uintptr, uintptr)  { var buf [996]byte; use(buf[:]); return Stackguard() }
func stack1000() (uintptr, uintptr) { var buf [1000]byte; use(buf[:]); return Stackguard() }
func stack1004() (uintptr, uintptr) { var buf [1004]byte; use(buf[:]); return Stackguard() }
func stack1008() (uintptr, uintptr) { var buf [1008]byte; use(buf[:]); return Stackguard() }
func stack1012() (uintptr, uintptr) { var buf [1012]byte; use(buf[:]); return Stackguard() }
func stack1016() (uintptr, uintptr) { var buf [1016]byte; use(buf[:]); return Stackguard() }
func stack1020() (uintptr, uintptr) { var buf [1020]byte; use(buf[:]); return Stackguard() }
func stack1024() (uintptr, uintptr) { var buf [1024]byte; use(buf[:]); return Stackguard() }
func stack1028() (uintptr, uintptr) { var buf [1028]byte; use(buf[:]); return Stackguard() }
func stack1032() (uintptr, uintptr) { var buf [1032]byte; use(buf[:]); return Stackguard() }
func stack1036() (uintptr, uintptr) { var buf [1036]byte; use(buf[:]); return Stackguard() }
func stack1040() (uintptr, uintptr) { var buf [1040]byte; use(buf[:]); return Stackguard() }
func stack1044() (uintptr, uintptr) { var buf [1044]byte; use(buf[:]); return Stackguard() }
func stack1048() (uintptr, uintptr) { var buf [1048]byte; use(buf[:]); return Stackguard() }
func stack1052() (uintptr, uintptr) { var buf [1052]byte; use(buf[:]); return Stackguard() }
func stack1056() (uintptr, uintptr) { var buf [1056]byte; use(buf[:]); return Stackguard() }
func stack1060() (uintptr, uintptr) { var buf [1060]byte; use(buf[:]); return Stackguard() }
func stack1064() (uintptr, uintptr) { var buf [1064]byte; use(buf[:]); return Stackguard() }
func stack1068() (uintptr, uintptr) { var buf [1068]byte; use(buf[:]); return Stackguard() }
func stack1072() (uintptr, uintptr) { var buf [1072]byte; use(buf[:]); return Stackguard() }
func stack1076() (uintptr, uintptr) { var buf [1076]byte; use(buf[:]); return Stackguard() }
func stack1080() (uintptr, uintptr) { var buf [1080]byte; use(buf[:]); return Stackguard() }
func stack1084() (uintptr, uintptr) { var buf [1084]byte; use(buf[:]); return Stackguard() }
func stack1088() (uintptr, uintptr) { var buf [1088]byte; use(buf[:]); return Stackguard() }
func stack1092() (uintptr, uintptr) { var buf [1092]byte; use(buf[:]); return Stackguard() }
func stack1096() (uintptr, uintptr) { var buf [1096]byte; use(buf[:]); return Stackguard() }
func stack1100() (uintptr, uintptr) { var buf [1100]byte; use(buf[:]); return Stackguard() }
func stack1104() (uintptr, uintptr) { var buf [1104]byte; use(buf[:]); return Stackguard() }
func stack1108() (uintptr, uintptr) { var buf [1108]byte; use(buf[:]); return Stackguard() }
func stack1112() (uintptr, uintptr) { var buf [1112]byte; use(buf[:]); return Stackguard() }
func stack1116() (uintptr, uintptr) { var buf [1116]byte; use(buf[:]); return Stackguard() }
func stack1120() (uintptr, uintptr) { var buf [1120]byte; use(buf[:]); return Stackguard() }
func stack1124() (uintptr, uintptr) { var buf [1124]byte; use(buf[:]); return Stackguard() }
func stack1128() (uintptr, uintptr) { var buf [1128]byte; use(buf[:]); return Stackguard() }
func stack1132() (uintptr, uintptr) { var buf [1132]byte; use(buf[:]); return Stackguard() }
func stack1136() (uintptr, uintptr) { var buf [1136]byte; use(buf[:]); return Stackguard() }
func stack1140() (uintptr, uintptr) { var buf [1140]byte; use(buf[:]); return Stackguard() }
func stack1144() (uintptr, uintptr) { var buf [1144]byte; use(buf[:]); return Stackguard() }
func stack1148() (uintptr, uintptr) { var buf [1148]byte; use(buf[:]); return Stackguard() }
func stack1152() (uintptr, uintptr) { var buf [1152]byte; use(buf[:]); return Stackguard() }
func stack1156() (uintptr, uintptr) { var buf [1156]byte; use(buf[:]); return Stackguard() }
func stack1160() (uintptr, uintptr) { var buf [1160]byte; use(buf[:]); return Stackguard() }
func stack1164() (uintptr, uintptr) { var buf [1164]byte; use(buf[:]); return Stackguard() }
func stack1168() (uintptr, uintptr) { var buf [1168]byte; use(buf[:]); return Stackguard() }
func stack1172() (uintptr, uintptr) { var buf [1172]byte; use(buf[:]); return Stackguard() }
func stack1176() (uintptr, uintptr) { var buf [1176]byte; use(buf[:]); return Stackguard() }
func stack1180() (uintptr, uintptr) { var buf [1180]byte; use(buf[:]); return Stackguard() }
func stack1184() (uintptr, uintptr) { var buf [1184]byte; use(buf[:]); return Stackguard() }
func stack1188() (uintptr, uintptr) { var buf [1188]byte; use(buf[:]); return Stackguard() }
func stack1192() (uintptr, uintptr) { var buf [1192]byte; use(buf[:]); return Stackguard() }
func stack1196() (uintptr, uintptr) { var buf [1196]byte; use(buf[:]); return Stackguard() }
func stack1200() (uintptr, uintptr) { var buf [1200]byte; use(buf[:]); return Stackguard() }
func stack1204() (uintptr, uintptr) { var buf [1204]byte; use(buf[:]); return Stackguard() }
func stack1208() (uintptr, uintptr) { var buf [1208]byte; use(buf[:]); return Stackguard() }
func stack1212() (uintptr, uintptr) { var buf [1212]byte; use(buf[:]); return Stackguard() }
func stack1216() (uintptr, uintptr) { var buf [1216]byte; use(buf[:]); return Stackguard() }
func stack1220() (uintptr, uintptr) { var buf [1220]byte; use(buf[:]); return Stackguard() }
func stack1224() (uintptr, uintptr) { var buf [1224]byte; use(buf[:]); return Stackguard() }
func stack1228() (uintptr, uintptr) { var buf [1228]byte; use(buf[:]); return Stackguard() }
func stack1232() (uintptr, uintptr) { var buf [1232]byte; use(buf[:]); return Stackguard() }
func stack1236() (uintptr, uintptr) { var buf [1236]byte; use(buf[:]); return Stackguard() }
func stack1240() (uintptr, uintptr) { var buf [1240]byte; use(buf[:]); return Stackguard() }
func stack1244() (uintptr, uintptr) { var buf [1244]byte; use(buf[:]); return Stackguard() }
func stack1248() (uintptr, uintptr) { var buf [1248]byte; use(buf[:]); return Stackguard() }
func stack1252() (uintptr, uintptr) { var buf [1252]byte; use(buf[:]); return Stackguard() }
func stack1256() (uintptr, uintptr) { var buf [1256]byte; use(buf[:]); return Stackguard() }
func stack1260() (uintptr, uintptr) { var buf [1260]byte; use(buf[:]); return Stackguard() }
func stack1264() (uintptr, uintptr) { var buf [1264]byte; use(buf[:]); return Stackguard() }
func stack1268() (uintptr, uintptr) { var buf [1268]byte; use(buf[:]); return Stackguard() }
func stack1272() (uintptr, uintptr) { var buf [1272]byte; use(buf[:]); return Stackguard() }
func stack1276() (uintptr, uintptr) { var buf [1276]byte; use(buf[:]); return Stackguard() }
func stack1280() (uintptr, uintptr) { var buf [1280]byte; use(buf[:]); return Stackguard() }
func stack1284() (uintptr, uintptr) { var buf [1284]byte; use(buf[:]); return Stackguard() }
func stack1288() (uintptr, uintptr) { var buf [1288]byte; use(buf[:]); return Stackguard() }
func stack1292() (uintptr, uintptr) { var buf [1292]byte; use(buf[:]); return Stackguard() }
func stack1296() (uintptr, uintptr) { var buf [1296]byte; use(buf[:]); return Stackguard() }
func stack1300() (uintptr, uintptr) { var buf [1300]byte; use(buf[:]); return Stackguard() }
func stack1304() (uintptr, uintptr) { var buf [1304]byte; use(buf[:]); return Stackguard() }
func stack1308() (uintptr, uintptr) { var buf [1308]byte; use(buf[:]); return Stackguard() }
func stack1312() (uintptr, uintptr) { var buf [1312]byte; use(buf[:]); return Stackguard() }
func stack1316() (uintptr, uintptr) { var buf [1316]byte; use(buf[:]); return Stackguard() }
func stack1320() (uintptr, uintptr) { var buf [1320]byte; use(buf[:]); return Stackguard() }
func stack1324() (uintptr, uintptr) { var buf [1324]byte; use(buf[:]); return Stackguard() }
func stack1328() (uintptr, uintptr) { var buf [1328]byte; use(buf[:]); return Stackguard() }
func stack1332() (uintptr, uintptr) { var buf [1332]byte; use(buf[:]); return Stackguard() }
func stack1336() (uintptr, uintptr) { var buf [1336]byte; use(buf[:]); return Stackguard() }
func stack1340() (uintptr, uintptr) { var buf [1340]byte; use(buf[:]); return Stackguard() }
func stack1344() (uintptr, uintptr) { var buf [1344]byte; use(buf[:]); return Stackguard() }
func stack1348() (uintptr, uintptr) { var buf [1348]byte; use(buf[:]); return Stackguard() }
func stack1352() (uintptr, uintptr) { var buf [1352]byte; use(buf[:]); return Stackguard() }
func stack1356() (uintptr, uintptr) { var buf [1356]byte; use(buf[:]); return Stackguard() }
func stack1360() (uintptr, uintptr) { var buf [1360]byte; use(buf[:]); return Stackguard() }
func stack1364() (uintptr, uintptr) { var buf [1364]byte; use(buf[:]); return Stackguard() }
func stack1368() (uintptr, uintptr) { var buf [1368]byte; use(buf[:]); return Stackguard() }
func stack1372() (uintptr, uintptr) { var buf [1372]byte; use(buf[:]); return Stackguard() }
func stack1376() (uintptr, uintptr) { var buf [1376]byte; use(buf[:]); return Stackguard() }
func stack1380() (uintptr, uintptr) { var buf [1380]byte; use(buf[:]); return Stackguard() }
func stack1384() (uintptr, uintptr) { var buf [1384]byte; use(buf[:]); return Stackguard() }
func stack1388() (uintptr, uintptr) { var buf [1388]byte; use(buf[:]); return Stackguard() }
func stack1392() (uintptr, uintptr) { var buf [1392]byte; use(buf[:]); return Stackguard() }
func stack1396() (uintptr, uintptr) { var buf [1396]byte; use(buf[:]); return Stackguard() }
func stack1400() (uintptr, uintptr) { var buf [1400]byte; use(buf[:]); return Stackguard() }
func stack1404() (uintptr, uintptr) { var buf [1404]byte; use(buf[:]); return Stackguard() }
func stack1408() (uintptr, uintptr) { var buf [1408]byte; use(buf[:]); return Stackguard() }
func stack1412() (uintptr, uintptr) { var buf [1412]byte; use(buf[:]); return Stackguard() }
func stack1416() (uintptr, uintptr) { var buf [1416]byte; use(buf[:]); return Stackguard() }
func stack1420() (uintptr, uintptr) { var buf [1420]byte; use(buf[:]); return Stackguard() }
func stack1424() (uintptr, uintptr) { var buf [1424]byte; use(buf[:]); return Stackguard() }
func stack1428() (uintptr, uintptr) { var buf [1428]byte; use(buf[:]); return Stackguard() }
func stack1432() (uintptr, uintptr) { var buf [1432]byte; use(buf[:]); return Stackguard() }
func stack1436() (uintptr, uintptr) { var buf [1436]byte; use(buf[:]); return Stackguard() }
func stack1440() (uintptr, uintptr) { var buf [1440]byte; use(buf[:]); return Stackguard() }
func stack1444() (uintptr, uintptr) { var buf [1444]byte; use(buf[:]); return Stackguard() }
func stack1448() (uintptr, uintptr) { var buf [1448]byte; use(buf[:]); return Stackguard() }
func stack1452() (uintptr, uintptr) { var buf [1452]byte; use(buf[:]); return Stackguard() }
func stack1456() (uintptr, uintptr) { var buf [1456]byte; use(buf[:]); return Stackguard() }
func stack1460() (uintptr, uintptr) { var buf [1460]byte; use(buf[:]); return Stackguard() }
func stack1464() (uintptr, uintptr) { var buf [1464]byte; use(buf[:]); return Stackguard() }
func stack1468() (uintptr, uintptr) { var buf [1468]byte; use(buf[:]); return Stackguard() }
func stack1472() (uintptr, uintptr) { var buf [1472]byte; use(buf[:]); return Stackguard() }
func stack1476() (uintptr, uintptr) { var buf [1476]byte; use(buf[:]); return Stackguard() }
func stack1480() (uintptr, uintptr) { var buf [1480]byte; use(buf[:]); return Stackguard() }
func stack1484() (uintptr, uintptr) { var buf [1484]byte; use(buf[:]); return Stackguard() }
func stack1488() (uintptr, uintptr) { var buf [1488]byte; use(buf[:]); return Stackguard() }
func stack1492() (uintptr, uintptr) { var buf [1492]byte; use(buf[:]); return Stackguard() }
func stack1496() (uintptr, uintptr) { var buf [1496]byte; use(buf[:]); return Stackguard() }
func stack1500() (uintptr, uintptr) { var buf [1500]byte; use(buf[:]); return Stackguard() }
func stack1504() (uintptr, uintptr) { var buf [1504]byte; use(buf[:]); return Stackguard() }
func stack1508() (uintptr, uintptr) { var buf [1508]byte; use(buf[:]); return Stackguard() }
func stack1512() (uintptr, uintptr) { var buf [1512]byte; use(buf[:]); return Stackguard() }
func stack1516() (uintptr, uintptr) { var buf [1516]byte; use(buf[:]); return Stackguard() }
func stack1520() (uintptr, uintptr) { var buf [1520]byte; use(buf[:]); return Stackguard() }
func stack1524() (uintptr, uintptr) { var buf [1524]byte; use(buf[:]); return Stackguard() }
func stack1528() (uintptr, uintptr) { var buf [1528]byte; use(buf[:]); return Stackguard() }
func stack1532() (uintptr, uintptr) { var buf [1532]byte; use(buf[:]); return Stackguard() }
func stack1536() (uintptr, uintptr) { var buf [1536]byte; use(buf[:]); return Stackguard() }
func stack1540() (uintptr, uintptr) { var buf [1540]byte; use(buf[:]); return Stackguard() }
func stack1544() (uintptr, uintptr) { var buf [1544]byte; use(buf[:]); return Stackguard() }
func stack1548() (uintptr, uintptr) { var buf [1548]byte; use(buf[:]); return Stackguard() }
func stack1552() (uintptr, uintptr) { var buf [1552]byte; use(buf[:]); return Stackguard() }
func stack1556() (uintptr, uintptr) { var buf [1556]byte; use(buf[:]); return Stackguard() }
func stack1560() (uintptr, uintptr) { var buf [1560]byte; use(buf[:]); return Stackguard() }
func stack1564() (uintptr, uintptr) { var buf [1564]byte; use(buf[:]); return Stackguard() }
func stack1568() (uintptr, uintptr) { var buf [1568]byte; use(buf[:]); return Stackguard() }
func stack1572() (uintptr, uintptr) { var buf [1572]byte; use(buf[:]); return Stackguard() }
func stack1576() (uintptr, uintptr) { var buf [1576]byte; use(buf[:]); return Stackguard() }
func stack1580() (uintptr, uintptr) { var buf [1580]byte; use(buf[:]); return Stackguard() }
func stack1584() (uintptr, uintptr) { var buf [1584]byte; use(buf[:]); return Stackguard() }
func stack1588() (uintptr, uintptr) { var buf [1588]byte; use(buf[:]); return Stackguard() }
func stack1592() (uintptr, uintptr) { var buf [1592]byte; use(buf[:]); return Stackguard() }
func stack1596() (uintptr, uintptr) { var buf [1596]byte; use(buf[:]); return Stackguard() }
func stack1600() (uintptr, uintptr) { var buf [1600]byte; use(buf[:]); return Stackguard() }
func stack1604() (uintptr, uintptr) { var buf [1604]byte; use(buf[:]); return Stackguard() }
func stack1608() (uintptr, uintptr) { var buf [1608]byte; use(buf[:]); return Stackguard() }
func stack1612() (uintptr, uintptr) { var buf [1612]byte; use(buf[:]); return Stackguard() }
func stack1616() (uintptr, uintptr) { var buf [1616]byte; use(buf[:]); return Stackguard() }
func stack1620() (uintptr, uintptr) { var buf [1620]byte; use(buf[:]); return Stackguard() }
func stack1624() (uintptr, uintptr) { var buf [1624]byte; use(buf[:]); return Stackguard() }
func stack1628() (uintptr, uintptr) { var buf [1628]byte; use(buf[:]); return Stackguard() }
func stack1632() (uintptr, uintptr) { var buf [1632]byte; use(buf[:]); return Stackguard() }
func stack1636() (uintptr, uintptr) { var buf [1636]byte; use(buf[:]); return Stackguard() }
func stack1640() (uintptr, uintptr) { var buf [1640]byte; use(buf[:]); return Stackguard() }
func stack1644() (uintptr, uintptr) { var buf [1644]byte; use(buf[:]); return Stackguard() }
func stack1648() (uintptr, uintptr) { var buf [1648]byte; use(buf[:]); return Stackguard() }
func stack1652() (uintptr, uintptr) { var buf [1652]byte; use(buf[:]); return Stackguard() }
func stack1656() (uintptr, uintptr) { var buf [1656]byte; use(buf[:]); return Stackguard() }
func stack1660() (uintptr, uintptr) { var buf [1660]byte; use(buf[:]); return Stackguard() }
func stack1664() (uintptr, uintptr) { var buf [1664]byte; use(buf[:]); return Stackguard() }
func stack1668() (uintptr, uintptr) { var buf [1668]byte; use(buf[:]); return Stackguard() }
func stack1672() (uintptr, uintptr) { var buf [1672]byte; use(buf[:]); return Stackguard() }
func stack1676() (uintptr, uintptr) { var buf [1676]byte; use(buf[:]); return Stackguard() }
func stack1680() (uintptr, uintptr) { var buf [1680]byte; use(buf[:]); return Stackguard() }
func stack1684() (uintptr, uintptr) { var buf [1684]byte; use(buf[:]); return Stackguard() }
func stack1688() (uintptr, uintptr) { var buf [1688]byte; use(buf[:]); return Stackguard() }
func stack1692() (uintptr, uintptr) { var buf [1692]byte; use(buf[:]); return Stackguard() }
func stack1696() (uintptr, uintptr) { var buf [1696]byte; use(buf[:]); return Stackguard() }
func stack1700() (uintptr, uintptr) { var buf [1700]byte; use(buf[:]); return Stackguard() }
func stack1704() (uintptr, uintptr) { var buf [1704]byte; use(buf[:]); return Stackguard() }
func stack1708() (uintptr, uintptr) { var buf [1708]byte; use(buf[:]); return Stackguard() }
func stack1712() (uintptr, uintptr) { var buf [1712]byte; use(buf[:]); return Stackguard() }
func stack1716() (uintptr, uintptr) { var buf [1716]byte; use(buf[:]); return Stackguard() }
func stack1720() (uintptr, uintptr) { var buf [1720]byte; use(buf[:]); return Stackguard() }
func stack1724() (uintptr, uintptr) { var buf [1724]byte; use(buf[:]); return Stackguard() }
func stack1728() (uintptr, uintptr) { var buf [1728]byte; use(buf[:]); return Stackguard() }
func stack1732() (uintptr, uintptr) { var buf [1732]byte; use(buf[:]); return Stackguard() }
func stack1736() (uintptr, uintptr) { var buf [1736]byte; use(buf[:]); return Stackguard() }
func stack1740() (uintptr, uintptr) { var buf [1740]byte; use(buf[:]); return Stackguard() }
func stack1744() (uintptr, uintptr) { var buf [1744]byte; use(buf[:]); return Stackguard() }
func stack1748() (uintptr, uintptr) { var buf [1748]byte; use(buf[:]); return Stackguard() }
func stack1752() (uintptr, uintptr) { var buf [1752]byte; use(buf[:]); return Stackguard() }
func stack1756() (uintptr, uintptr) { var buf [1756]byte; use(buf[:]); return Stackguard() }
func stack1760() (uintptr, uintptr) { var buf [1760]byte; use(buf[:]); return Stackguard() }
func stack1764() (uintptr, uintptr) { var buf [1764]byte; use(buf[:]); return Stackguard() }
func stack1768() (uintptr, uintptr) { var buf [1768]byte; use(buf[:]); return Stackguard() }
func stack1772() (uintptr, uintptr) { var buf [1772]byte; use(buf[:]); return Stackguard() }
func stack1776() (uintptr, uintptr) { var buf [1776]byte; use(buf[:]); return Stackguard() }
func stack1780() (uintptr, uintptr) { var buf [1780]byte; use(buf[:]); return Stackguard() }
func stack1784() (uintptr, uintptr) { var buf [1784]byte; use(buf[:]); return Stackguard() }
func stack1788() (uintptr, uintptr) { var buf [1788]byte; use(buf[:]); return Stackguard() }
func stack1792() (uintptr, uintptr) { var buf [1792]byte; use(buf[:]); return Stackguard() }
func stack1796() (uintptr, uintptr) { var buf [1796]byte; use(buf[:]); return Stackguard() }
func stack1800() (uintptr, uintptr) { var buf [1800]byte; use(buf[:]); return Stackguard() }
func stack1804() (uintptr, uintptr) { var buf [1804]byte; use(buf[:]); return Stackguard() }
func stack1808() (uintptr, uintptr) { var buf [1808]byte; use(buf[:]); return Stackguard() }
func stack1812() (uintptr, uintptr) { var buf [1812]byte; use(buf[:]); return Stackguard() }
func stack1816() (uintptr, uintptr) { var buf [1816]byte; use(buf[:]); return Stackguard() }
func stack1820() (uintptr, uintptr) { var buf [1820]byte; use(buf[:]); return Stackguard() }
func stack1824() (uintptr, uintptr) { var buf [1824]byte; use(buf[:]); return Stackguard() }
func stack1828() (uintptr, uintptr) { var buf [1828]byte; use(buf[:]); return Stackguard() }
func stack1832() (uintptr, uintptr) { var buf [1832]byte; use(buf[:]); return Stackguard() }
func stack1836() (uintptr, uintptr) { var buf [1836]byte; use(buf[:]); return Stackguard() }
func stack1840() (uintptr, uintptr) { var buf [1840]byte; use(buf[:]); return Stackguard() }
func stack1844() (uintptr, uintptr) { var buf [1844]byte; use(buf[:]); return Stackguard() }
func stack1848() (uintptr, uintptr) { var buf [1848]byte; use(buf[:]); return Stackguard() }
func stack1852() (uintptr, uintptr) { var buf [1852]byte; use(buf[:]); return Stackguard() }
func stack1856() (uintptr, uintptr) { var buf [1856]byte; use(buf[:]); return Stackguard() }
func stack1860() (uintptr, uintptr) { var buf [1860]byte; use(buf[:]); return Stackguard() }
func stack1864() (uintptr, uintptr) { var buf [1864]byte; use(buf[:]); return Stackguard() }
func stack1868() (uintptr, uintptr) { var buf [1868]byte; use(buf[:]); return Stackguard() }
func stack1872() (uintptr, uintptr) { var buf [1872]byte; use(buf[:]); return Stackguard() }
func stack1876() (uintptr, uintptr) { var buf [1876]byte; use(buf[:]); return Stackguard() }
func stack1880() (uintptr, uintptr) { var buf [1880]byte; use(buf[:]); return Stackguard() }
func stack1884() (uintptr, uintptr) { var buf [1884]byte; use(buf[:]); return Stackguard() }
func stack1888() (uintptr, uintptr) { var buf [1888]byte; use(buf[:]); return Stackguard() }
func stack1892() (uintptr, uintptr) { var buf [1892]byte; use(buf[:]); return Stackguard() }
func stack1896() (uintptr, uintptr) { var buf [1896]byte; use(buf[:]); return Stackguard() }
func stack1900() (uintptr, uintptr) { var buf [1900]byte; use(buf[:]); return Stackguard() }
func stack1904() (uintptr, uintptr) { var buf [1904]byte; use(buf[:]); return Stackguard() }
func stack1908() (uintptr, uintptr) { var buf [1908]byte; use(buf[:]); return Stackguard() }
func stack1912() (uintptr, uintptr) { var buf [1912]byte; use(buf[:]); return Stackguard() }
func stack1916() (uintptr, uintptr) { var buf [1916]byte; use(buf[:]); return Stackguard() }
func stack1920() (uintptr, uintptr) { var buf [1920]byte; use(buf[:]); return Stackguard() }
func stack1924() (uintptr, uintptr) { var buf [1924]byte; use(buf[:]); return Stackguard() }
func stack1928() (uintptr, uintptr) { var buf [1928]byte; use(buf[:]); return Stackguard() }
func stack1932() (uintptr, uintptr) { var buf [1932]byte; use(buf[:]); return Stackguard() }
func stack1936() (uintptr, uintptr) { var buf [1936]byte; use(buf[:]); return Stackguard() }
func stack1940() (uintptr, uintptr) { var buf [1940]byte; use(buf[:]); return Stackguard() }
func stack1944() (uintptr, uintptr) { var buf [1944]byte; use(buf[:]); return Stackguard() }
func stack1948() (uintptr, uintptr) { var buf [1948]byte; use(buf[:]); return Stackguard() }
func stack1952() (uintptr, uintptr) { var buf [1952]byte; use(buf[:]); return Stackguard() }
func stack1956() (uintptr, uintptr) { var buf [1956]byte; use(buf[:]); return Stackguard() }
func stack1960() (uintptr, uintptr) { var buf [1960]byte; use(buf[:]); return Stackguard() }
func stack1964() (uintptr, uintptr) { var buf [1964]byte; use(buf[:]); return Stackguard() }
func stack1968() (uintptr, uintptr) { var buf [1968]byte; use(buf[:]); return Stackguard() }
func stack1972() (uintptr, uintptr) { var buf [1972]byte; use(buf[:]); return Stackguard() }
func stack1976() (uintptr, uintptr) { var buf [1976]byte; use(buf[:]); return Stackguard() }
func stack1980() (uintptr, uintptr) { var buf [1980]byte; use(buf[:]); return Stackguard() }
func stack1984() (uintptr, uintptr) { var buf [1984]byte; use(buf[:]); return Stackguard() }
func stack1988() (uintptr, uintptr) { var buf [1988]byte; use(buf[:]); return Stackguard() }
func stack1992() (uintptr, uintptr) { var buf [1992]byte; use(buf[:]); return Stackguard() }
func stack1996() (uintptr, uintptr) { var buf [1996]byte; use(buf[:]); return Stackguard() }
func stack2000() (uintptr, uintptr) { var buf [2000]byte; use(buf[:]); return Stackguard() }
func stack2004() (uintptr, uintptr) { var buf [2004]byte; use(buf[:]); return Stackguard() }
func stack2008() (uintptr, uintptr) { var buf [2008]byte; use(buf[:]); return Stackguard() }
func stack2012() (uintptr, uintptr) { var buf [2012]byte; use(buf[:]); return Stackguard() }
func stack2016() (uintptr, uintptr) { var buf [2016]byte; use(buf[:]); return Stackguard() }
func stack2020() (uintptr, uintptr) { var buf [2020]byte; use(buf[:]); return Stackguard() }
func stack2024() (uintptr, uintptr) { var buf [2024]byte; use(buf[:]); return Stackguard() }
func stack2028() (uintptr, uintptr) { var buf [2028]byte; use(buf[:]); return Stackguard() }
func stack2032() (uintptr, uintptr) { var buf [2032]byte; use(buf[:]); return Stackguard() }
func stack2036() (uintptr, uintptr) { var buf [2036]byte; use(buf[:]); return Stackguard() }
func stack2040() (uintptr, uintptr) { var buf [2040]byte; use(buf[:]); return Stackguard() }
func stack2044() (uintptr, uintptr) { var buf [2044]byte; use(buf[:]); return Stackguard() }
func stack2048() (uintptr, uintptr) { var buf [2048]byte; use(buf[:]); return Stackguard() }
func stack2052() (uintptr, uintptr) { var buf [2052]byte; use(buf[:]); return Stackguard() }
func stack2056() (uintptr, uintptr) { var buf [2056]byte; use(buf[:]); return Stackguard() }
func stack2060() (uintptr, uintptr) { var buf [2060]byte; use(buf[:]); return Stackguard() }
func stack2064() (uintptr, uintptr) { var buf [2064]byte; use(buf[:]); return Stackguard() }
func stack2068() (uintptr, uintptr) { var buf [2068]byte; use(buf[:]); return Stackguard() }
func stack2072() (uintptr, uintptr) { var buf [2072]byte; use(buf[:]); return Stackguard() }
func stack2076() (uintptr, uintptr) { var buf [2076]byte; use(buf[:]); return Stackguard() }
func stack2080() (uintptr, uintptr) { var buf [2080]byte; use(buf[:]); return Stackguard() }
func stack2084() (uintptr, uintptr) { var buf [2084]byte; use(buf[:]); return Stackguard() }
func stack2088() (uintptr, uintptr) { var buf [2088]byte; use(buf[:]); return Stackguard() }
func stack2092() (uintptr, uintptr) { var buf [2092]byte; use(buf[:]); return Stackguard() }
func stack2096() (uintptr, uintptr) { var buf [2096]byte; use(buf[:]); return Stackguard() }
func stack2100() (uintptr, uintptr) { var buf [2100]byte; use(buf[:]); return Stackguard() }
func stack2104() (uintptr, uintptr) { var buf [2104]byte; use(buf[:]); return Stackguard() }
func stack2108() (uintptr, uintptr) { var buf [2108]byte; use(buf[:]); return Stackguard() }
func stack2112() (uintptr, uintptr) { var buf [2112]byte; use(buf[:]); return Stackguard() }
func stack2116() (uintptr, uintptr) { var buf [2116]byte; use(buf[:]); return Stackguard() }
func stack2120() (uintptr, uintptr) { var buf [2120]byte; use(buf[:]); return Stackguard() }
func stack2124() (uintptr, uintptr) { var buf [2124]byte; use(buf[:]); return Stackguard() }
func stack2128() (uintptr, uintptr) { var buf [2128]byte; use(buf[:]); return Stackguard() }
func stack2132() (uintptr, uintptr) { var buf [2132]byte; use(buf[:]); return Stackguard() }
func stack2136() (uintptr, uintptr) { var buf [2136]byte; use(buf[:]); return Stackguard() }
func stack2140() (uintptr, uintptr) { var buf [2140]byte; use(buf[:]); return Stackguard() }
func stack2144() (uintptr, uintptr) { var buf [2144]byte; use(buf[:]); return Stackguard() }
func stack2148() (uintptr, uintptr) { var buf [2148]byte; use(buf[:]); return Stackguard() }
func stack2152() (uintptr, uintptr) { var buf [2152]byte; use(buf[:]); return Stackguard() }
func stack2156() (uintptr, uintptr) { var buf [2156]byte; use(buf[:]); return Stackguard() }
func stack2160() (uintptr, uintptr) { var buf [2160]byte; use(buf[:]); return Stackguard() }
func stack2164() (uintptr, uintptr) { var buf [2164]byte; use(buf[:]); return Stackguard() }
func stack2168() (uintptr, uintptr) { var buf [2168]byte; use(buf[:]); return Stackguard() }
func stack2172() (uintptr, uintptr) { var buf [2172]byte; use(buf[:]); return Stackguard() }
func stack2176() (uintptr, uintptr) { var buf [2176]byte; use(buf[:]); return Stackguard() }
func stack2180() (uintptr, uintptr) { var buf [2180]byte; use(buf[:]); return Stackguard() }
func stack2184() (uintptr, uintptr) { var buf [2184]byte; use(buf[:]); return Stackguard() }
func stack2188() (uintptr, uintptr) { var buf [2188]byte; use(buf[:]); return Stackguard() }
func stack2192() (uintptr, uintptr) { var buf [2192]byte; use(buf[:]); return Stackguard() }
func stack2196() (uintptr, uintptr) { var buf [2196]byte; use(buf[:]); return Stackguard() }
func stack2200() (uintptr, uintptr) { var buf [2200]byte; use(buf[:]); return Stackguard() }
func stack2204() (uintptr, uintptr) { var buf [2204]byte; use(buf[:]); return Stackguard() }
func stack2208() (uintptr, uintptr) { var buf [2208]byte; use(buf[:]); return Stackguard() }
func stack2212() (uintptr, uintptr) { var buf [2212]byte; use(buf[:]); return Stackguard() }
func stack2216() (uintptr, uintptr) { var buf [2216]byte; use(buf[:]); return Stackguard() }
func stack2220() (uintptr, uintptr) { var buf [2220]byte; use(buf[:]); return Stackguard() }
func stack2224() (uintptr, uintptr) { var buf [2224]byte; use(buf[:]); return Stackguard() }
func stack2228() (uintptr, uintptr) { var buf [2228]byte; use(buf[:]); return Stackguard() }
func stack2232() (uintptr, uintptr) { var buf [2232]byte; use(buf[:]); return Stackguard() }
func stack2236() (uintptr, uintptr) { var buf [2236]byte; use(buf[:]); return Stackguard() }
func stack2240() (uintptr, uintptr) { var buf [2240]byte; use(buf[:]); return Stackguard() }
func stack2244() (uintptr, uintptr) { var buf [2244]byte; use(buf[:]); return Stackguard() }
func stack2248() (uintptr, uintptr) { var buf [2248]byte; use(buf[:]); return Stackguard() }
func stack2252() (uintptr, uintptr) { var buf [2252]byte; use(buf[:]); return Stackguard() }
func stack2256() (uintptr, uintptr) { var buf [2256]byte; use(buf[:]); return Stackguard() }
func stack2260() (uintptr, uintptr) { var buf [2260]byte; use(buf[:]); return Stackguard() }
func stack2264() (uintptr, uintptr) { var buf [2264]byte; use(buf[:]); return Stackguard() }
func stack2268() (uintptr, uintptr) { var buf [2268]byte; use(buf[:]); return Stackguard() }
func stack2272() (uintptr, uintptr) { var buf [2272]byte; use(buf[:]); return Stackguard() }
func stack2276() (uintptr, uintptr) { var buf [2276]byte; use(buf[:]); return Stackguard() }
func stack2280() (uintptr, uintptr) { var buf [2280]byte; use(buf[:]); return Stackguard() }
func stack2284() (uintptr, uintptr) { var buf [2284]byte; use(buf[:]); return Stackguard() }
func stack2288() (uintptr, uintptr) { var buf [2288]byte; use(buf[:]); return Stackguard() }
func stack2292() (uintptr, uintptr) { var buf [2292]byte; use(buf[:]); return Stackguard() }
func stack2296() (uintptr, uintptr) { var buf [2296]byte; use(buf[:]); return Stackguard() }
func stack2300() (uintptr, uintptr) { var buf [2300]byte; use(buf[:]); return Stackguard() }
func stack2304() (uintptr, uintptr) { var buf [2304]byte; use(buf[:]); return Stackguard() }
func stack2308() (uintptr, uintptr) { var buf [2308]byte; use(buf[:]); return Stackguard() }
func stack2312() (uintptr, uintptr) { var buf [2312]byte; use(buf[:]); return Stackguard() }
func stack2316() (uintptr, uintptr) { var buf [2316]byte; use(buf[:]); return Stackguard() }
func stack2320() (uintptr, uintptr) { var buf [2320]byte; use(buf[:]); return Stackguard() }
func stack2324() (uintptr, uintptr) { var buf [2324]byte; use(buf[:]); return Stackguard() }
func stack2328() (uintptr, uintptr) { var buf [2328]byte; use(buf[:]); return Stackguard() }
func stack2332() (uintptr, uintptr) { var buf [2332]byte; use(buf[:]); return Stackguard() }
func stack2336() (uintptr, uintptr) { var buf [2336]byte; use(buf[:]); return Stackguard() }
func stack2340() (uintptr, uintptr) { var buf [2340]byte; use(buf[:]); return Stackguard() }
func stack2344() (uintptr, uintptr) { var buf [2344]byte; use(buf[:]); return Stackguard() }
func stack2348() (uintptr, uintptr) { var buf [2348]byte; use(buf[:]); return Stackguard() }
func stack2352() (uintptr, uintptr) { var buf [2352]byte; use(buf[:]); return Stackguard() }
func stack2356() (uintptr, uintptr) { var buf [2356]byte; use(buf[:]); return Stackguard() }
func stack2360() (uintptr, uintptr) { var buf [2360]byte; use(buf[:]); return Stackguard() }
func stack2364() (uintptr, uintptr) { var buf [2364]byte; use(buf[:]); return Stackguard() }
func stack2368() (uintptr, uintptr) { var buf [2368]byte; use(buf[:]); return Stackguard() }
func stack2372() (uintptr, uintptr) { var buf [2372]byte; use(buf[:]); return Stackguard() }
func stack2376() (uintptr, uintptr) { var buf [2376]byte; use(buf[:]); return Stackguard() }
func stack2380() (uintptr, uintptr) { var buf [2380]byte; use(buf[:]); return Stackguard() }
func stack2384() (uintptr, uintptr) { var buf [2384]byte; use(buf[:]); return Stackguard() }
func stack2388() (uintptr, uintptr) { var buf [2388]byte; use(buf[:]); return Stackguard() }
func stack2392() (uintptr, uintptr) { var buf [2392]byte; use(buf[:]); return Stackguard() }
func stack2396() (uintptr, uintptr) { var buf [2396]byte; use(buf[:]); return Stackguard() }
func stack2400() (uintptr, uintptr) { var buf [2400]byte; use(buf[:]); return Stackguard() }
func stack2404() (uintptr, uintptr) { var buf [2404]byte; use(buf[:]); return Stackguard() }
func stack2408() (uintptr, uintptr) { var buf [2408]byte; use(buf[:]); return Stackguard() }
func stack2412() (uintptr, uintptr) { var buf [2412]byte; use(buf[:]); return Stackguard() }
func stack2416() (uintptr, uintptr) { var buf [2416]byte; use(buf[:]); return Stackguard() }
func stack2420() (uintptr, uintptr) { var buf [2420]byte; use(buf[:]); return Stackguard() }
func stack2424() (uintptr, uintptr) { var buf [2424]byte; use(buf[:]); return Stackguard() }
func stack2428() (uintptr, uintptr) { var buf [2428]byte; use(buf[:]); return Stackguard() }
func stack2432() (uintptr, uintptr) { var buf [2432]byte; use(buf[:]); return Stackguard() }
func stack2436() (uintptr, uintptr) { var buf [2436]byte; use(buf[:]); return Stackguard() }
func stack2440() (uintptr, uintptr) { var buf [2440]byte; use(buf[:]); return Stackguard() }
func stack2444() (uintptr, uintptr) { var buf [2444]byte; use(buf[:]); return Stackguard() }
func stack2448() (uintptr, uintptr) { var buf [2448]byte; use(buf[:]); return Stackguard() }
func stack2452() (uintptr, uintptr) { var buf [2452]byte; use(buf[:]); return Stackguard() }
func stack2456() (uintptr, uintptr) { var buf [2456]byte; use(buf[:]); return Stackguard() }
func stack2460() (uintptr, uintptr) { var buf [2460]byte; use(buf[:]); return Stackguard() }
func stack2464() (uintptr, uintptr) { var buf [2464]byte; use(buf[:]); return Stackguard() }
func stack2468() (uintptr, uintptr) { var buf [2468]byte; use(buf[:]); return Stackguard() }
func stack2472() (uintptr, uintptr) { var buf [2472]byte; use(buf[:]); return Stackguard() }
func stack2476() (uintptr, uintptr) { var buf [2476]byte; use(buf[:]); return Stackguard() }
func stack2480() (uintptr, uintptr) { var buf [2480]byte; use(buf[:]); return Stackguard() }
func stack2484() (uintptr, uintptr) { var buf [2484]byte; use(buf[:]); return Stackguard() }
func stack2488() (uintptr, uintptr) { var buf [2488]byte; use(buf[:]); return Stackguard() }
func stack2492() (uintptr, uintptr) { var buf [2492]byte; use(buf[:]); return Stackguard() }
func stack2496() (uintptr, uintptr) { var buf [2496]byte; use(buf[:]); return Stackguard() }
func stack2500() (uintptr, uintptr) { var buf [2500]byte; use(buf[:]); return Stackguard() }
func stack2504() (uintptr, uintptr) { var buf [2504]byte; use(buf[:]); return Stackguard() }
func stack2508() (uintptr, uintptr) { var buf [2508]byte; use(buf[:]); return Stackguard() }
func stack2512() (uintptr, uintptr) { var buf [2512]byte; use(buf[:]); return Stackguard() }
func stack2516() (uintptr, uintptr) { var buf [2516]byte; use(buf[:]); return Stackguard() }
func stack2520() (uintptr, uintptr) { var buf [2520]byte; use(buf[:]); return Stackguard() }
func stack2524() (uintptr, uintptr) { var buf [2524]byte; use(buf[:]); return Stackguard() }
func stack2528() (uintptr, uintptr) { var buf [2528]byte; use(buf[:]); return Stackguard() }
func stack2532() (uintptr, uintptr) { var buf [2532]byte; use(buf[:]); return Stackguard() }
func stack2536() (uintptr, uintptr) { var buf [2536]byte; use(buf[:]); return Stackguard() }
func stack2540() (uintptr, uintptr) { var buf [2540]byte; use(buf[:]); return Stackguard() }
func stack2544() (uintptr, uintptr) { var buf [2544]byte; use(buf[:]); return Stackguard() }
func stack2548() (uintptr, uintptr) { var buf [2548]byte; use(buf[:]); return Stackguard() }
func stack2552() (uintptr, uintptr) { var buf [2552]byte; use(buf[:]); return Stackguard() }
func stack2556() (uintptr, uintptr) { var buf [2556]byte; use(buf[:]); return Stackguard() }
func stack2560() (uintptr, uintptr) { var buf [2560]byte; use(buf[:]); return Stackguard() }
func stack2564() (uintptr, uintptr) { var buf [2564]byte; use(buf[:]); return Stackguard() }
func stack2568() (uintptr, uintptr) { var buf [2568]byte; use(buf[:]); return Stackguard() }
func stack2572() (uintptr, uintptr) { var buf [2572]byte; use(buf[:]); return Stackguard() }
func stack2576() (uintptr, uintptr) { var buf [2576]byte; use(buf[:]); return Stackguard() }
func stack2580() (uintptr, uintptr) { var buf [2580]byte; use(buf[:]); return Stackguard() }
func stack2584() (uintptr, uintptr) { var buf [2584]byte; use(buf[:]); return Stackguard() }
func stack2588() (uintptr, uintptr) { var buf [2588]byte; use(buf[:]); return Stackguard() }
func stack2592() (uintptr, uintptr) { var buf [2592]byte; use(buf[:]); return Stackguard() }
func stack2596() (uintptr, uintptr) { var buf [2596]byte; use(buf[:]); return Stackguard() }
func stack2600() (uintptr, uintptr) { var buf [2600]byte; use(buf[:]); return Stackguard() }
func stack2604() (uintptr, uintptr) { var buf [2604]byte; use(buf[:]); return Stackguard() }
func stack2608() (uintptr, uintptr) { var buf [2608]byte; use(buf[:]); return Stackguard() }
func stack2612() (uintptr, uintptr) { var buf [2612]byte; use(buf[:]); return Stackguard() }
func stack2616() (uintptr, uintptr) { var buf [2616]byte; use(buf[:]); return Stackguard() }
func stack2620() (uintptr, uintptr) { var buf [2620]byte; use(buf[:]); return Stackguard() }
func stack2624() (uintptr, uintptr) { var buf [2624]byte; use(buf[:]); return Stackguard() }
func stack2628() (uintptr, uintptr) { var buf [2628]byte; use(buf[:]); return Stackguard() }
func stack2632() (uintptr, uintptr) { var buf [2632]byte; use(buf[:]); return Stackguard() }
func stack2636() (uintptr, uintptr) { var buf [2636]byte; use(buf[:]); return Stackguard() }
func stack2640() (uintptr, uintptr) { var buf [2640]byte; use(buf[:]); return Stackguard() }
func stack2644() (uintptr, uintptr) { var buf [2644]byte; use(buf[:]); return Stackguard() }
func stack2648() (uintptr, uintptr) { var buf [2648]byte; use(buf[:]); return Stackguard() }
func stack2652() (uintptr, uintptr) { var buf [2652]byte; use(buf[:]); return Stackguard() }
func stack2656() (uintptr, uintptr) { var buf [2656]byte; use(buf[:]); return Stackguard() }
func stack2660() (uintptr, uintptr) { var buf [2660]byte; use(buf[:]); return Stackguard() }
func stack2664() (uintptr, uintptr) { var buf [2664]byte; use(buf[:]); return Stackguard() }
func stack2668() (uintptr, uintptr) { var buf [2668]byte; use(buf[:]); return Stackguard() }
func stack2672() (uintptr, uintptr) { var buf [2672]byte; use(buf[:]); return Stackguard() }
func stack2676() (uintptr, uintptr) { var buf [2676]byte; use(buf[:]); return Stackguard() }
func stack2680() (uintptr, uintptr) { var buf [2680]byte; use(buf[:]); return Stackguard() }
func stack2684() (uintptr, uintptr) { var buf [2684]byte; use(buf[:]); return Stackguard() }
func stack2688() (uintptr, uintptr) { var buf [2688]byte; use(buf[:]); return Stackguard() }
func stack2692() (uintptr, uintptr) { var buf [2692]byte; use(buf[:]); return Stackguard() }
func stack2696() (uintptr, uintptr) { var buf [2696]byte; use(buf[:]); return Stackguard() }
func stack2700() (uintptr, uintptr) { var buf [2700]byte; use(buf[:]); return Stackguard() }
func stack2704() (uintptr, uintptr) { var buf [2704]byte; use(buf[:]); return Stackguard() }
func stack2708() (uintptr, uintptr) { var buf [2708]byte; use(buf[:]); return Stackguard() }
func stack2712() (uintptr, uintptr) { var buf [2712]byte; use(buf[:]); return Stackguard() }
func stack2716() (uintptr, uintptr) { var buf [2716]byte; use(buf[:]); return Stackguard() }
func stack2720() (uintptr, uintptr) { var buf [2720]byte; use(buf[:]); return Stackguard() }
func stack2724() (uintptr, uintptr) { var buf [2724]byte; use(buf[:]); return Stackguard() }
func stack2728() (uintptr, uintptr) { var buf [2728]byte; use(buf[:]); return Stackguard() }
func stack2732() (uintptr, uintptr) { var buf [2732]byte; use(buf[:]); return Stackguard() }
func stack2736() (uintptr, uintptr) { var buf [2736]byte; use(buf[:]); return Stackguard() }
func stack2740() (uintptr, uintptr) { var buf [2740]byte; use(buf[:]); return Stackguard() }
func stack2744() (uintptr, uintptr) { var buf [2744]byte; use(buf[:]); return Stackguard() }
func stack2748() (uintptr, uintptr) { var buf [2748]byte; use(buf[:]); return Stackguard() }
func stack2752() (uintptr, uintptr) { var buf [2752]byte; use(buf[:]); return Stackguard() }
func stack2756() (uintptr, uintptr) { var buf [2756]byte; use(buf[:]); return Stackguard() }
func stack2760() (uintptr, uintptr) { var buf [2760]byte; use(buf[:]); return Stackguard() }
func stack2764() (uintptr, uintptr) { var buf [2764]byte; use(buf[:]); return Stackguard() }
func stack2768() (uintptr, uintptr) { var buf [2768]byte; use(buf[:]); return Stackguard() }
func stack2772() (uintptr, uintptr) { var buf [2772]byte; use(buf[:]); return Stackguard() }
func stack2776() (uintptr, uintptr) { var buf [2776]byte; use(buf[:]); return Stackguard() }
func stack2780() (uintptr, uintptr) { var buf [2780]byte; use(buf[:]); return Stackguard() }
func stack2784() (uintptr, uintptr) { var buf [2784]byte; use(buf[:]); return Stackguard() }
func stack2788() (uintptr, uintptr) { var buf [2788]byte; use(buf[:]); return Stackguard() }
func stack2792() (uintptr, uintptr) { var buf [2792]byte; use(buf[:]); return Stackguard() }
func stack2796() (uintptr, uintptr) { var buf [2796]byte; use(buf[:]); return Stackguard() }
func stack2800() (uintptr, uintptr) { var buf [2800]byte; use(buf[:]); return Stackguard() }
func stack2804() (uintptr, uintptr) { var buf [2804]byte; use(buf[:]); return Stackguard() }
func stack2808() (uintptr, uintptr) { var buf [2808]byte; use(buf[:]); return Stackguard() }
func stack2812() (uintptr, uintptr) { var buf [2812]byte; use(buf[:]); return Stackguard() }
func stack2816() (uintptr, uintptr) { var buf [2816]byte; use(buf[:]); return Stackguard() }
func stack2820() (uintptr, uintptr) { var buf [2820]byte; use(buf[:]); return Stackguard() }
func stack2824() (uintptr, uintptr) { var buf [2824]byte; use(buf[:]); return Stackguard() }
func stack2828() (uintptr, uintptr) { var buf [2828]byte; use(buf[:]); return Stackguard() }
func stack2832() (uintptr, uintptr) { var buf [2832]byte; use(buf[:]); return Stackguard() }
func stack2836() (uintptr, uintptr) { var buf [2836]byte; use(buf[:]); return Stackguard() }
func stack2840() (uintptr, uintptr) { var buf [2840]byte; use(buf[:]); return Stackguard() }
func stack2844() (uintptr, uintptr) { var buf [2844]byte; use(buf[:]); return Stackguard() }
func stack2848() (uintptr, uintptr) { var buf [2848]byte; use(buf[:]); return Stackguard() }
func stack2852() (uintptr, uintptr) { var buf [2852]byte; use(buf[:]); return Stackguard() }
func stack2856() (uintptr, uintptr) { var buf [2856]byte; use(buf[:]); return Stackguard() }
func stack2860() (uintptr, uintptr) { var buf [2860]byte; use(buf[:]); return Stackguard() }
func stack2864() (uintptr, uintptr) { var buf [2864]byte; use(buf[:]); return Stackguard() }
func stack2868() (uintptr, uintptr) { var buf [2868]byte; use(buf[:]); return Stackguard() }
func stack2872() (uintptr, uintptr) { var buf [2872]byte; use(buf[:]); return Stackguard() }
func stack2876() (uintptr, uintptr) { var buf [2876]byte; use(buf[:]); return Stackguard() }
func stack2880() (uintptr, uintptr) { var buf [2880]byte; use(buf[:]); return Stackguard() }
func stack2884() (uintptr, uintptr) { var buf [2884]byte; use(buf[:]); return Stackguard() }
func stack2888() (uintptr, uintptr) { var buf [2888]byte; use(buf[:]); return Stackguard() }
func stack2892() (uintptr, uintptr) { var buf [2892]byte; use(buf[:]); return Stackguard() }
func stack2896() (uintptr, uintptr) { var buf [2896]byte; use(buf[:]); return Stackguard() }
func stack2900() (uintptr, uintptr) { var buf [2900]byte; use(buf[:]); return Stackguard() }
func stack2904() (uintptr, uintptr) { var buf [2904]byte; use(buf[:]); return Stackguard() }
func stack2908() (uintptr, uintptr) { var buf [2908]byte; use(buf[:]); return Stackguard() }
func stack2912() (uintptr, uintptr) { var buf [2912]byte; use(buf[:]); return Stackguard() }
func stack2916() (uintptr, uintptr) { var buf [2916]byte; use(buf[:]); return Stackguard() }
func stack2920() (uintptr, uintptr) { var buf [2920]byte; use(buf[:]); return Stackguard() }
func stack2924() (uintptr, uintptr) { var buf [2924]byte; use(buf[:]); return Stackguard() }
func stack2928() (uintptr, uintptr) { var buf [2928]byte; use(buf[:]); return Stackguard() }
func stack2932() (uintptr, uintptr) { var buf [2932]byte; use(buf[:]); return Stackguard() }
func stack2936() (uintptr, uintptr) { var buf [2936]byte; use(buf[:]); return Stackguard() }
func stack2940() (uintptr, uintptr) { var buf [2940]byte; use(buf[:]); return Stackguard() }
func stack2944() (uintptr, uintptr) { var buf [2944]byte; use(buf[:]); return Stackguard() }
func stack2948() (uintptr, uintptr) { var buf [2948]byte; use(buf[:]); return Stackguard() }
func stack2952() (uintptr, uintptr) { var buf [2952]byte; use(buf[:]); return Stackguard() }
func stack2956() (uintptr, uintptr) { var buf [2956]byte; use(buf[:]); return Stackguard() }
func stack2960() (uintptr, uintptr) { var buf [2960]byte; use(buf[:]); return Stackguard() }
func stack2964() (uintptr, uintptr) { var buf [2964]byte; use(buf[:]); return Stackguard() }
func stack2968() (uintptr, uintptr) { var buf [2968]byte; use(buf[:]); return Stackguard() }
func stack2972() (uintptr, uintptr) { var buf [2972]byte; use(buf[:]); return Stackguard() }
func stack2976() (uintptr, uintptr) { var buf [2976]byte; use(buf[:]); return Stackguard() }
func stack2980() (uintptr, uintptr) { var buf [2980]byte; use(buf[:]); return Stackguard() }
func stack2984() (uintptr, uintptr) { var buf [2984]byte; use(buf[:]); return Stackguard() }
func stack2988() (uintptr, uintptr) { var buf [2988]byte; use(buf[:]); return Stackguard() }
func stack2992() (uintptr, uintptr) { var buf [2992]byte; use(buf[:]); return Stackguard() }
func stack2996() (uintptr, uintptr) { var buf [2996]byte; use(buf[:]); return Stackguard() }
func stack3000() (uintptr, uintptr) { var buf [3000]byte; use(buf[:]); return Stackguard() }
func stack3004() (uintptr, uintptr) { var buf [3004]byte; use(buf[:]); return Stackguard() }
func stack3008() (uintptr, uintptr) { var buf [3008]byte; use(buf[:]); return Stackguard() }
func stack3012() (uintptr, uintptr) { var buf [3012]byte; use(buf[:]); return Stackguard() }
func stack3016() (uintptr, uintptr) { var buf [3016]byte; use(buf[:]); return Stackguard() }
func stack3020() (uintptr, uintptr) { var buf [3020]byte; use(buf[:]); return Stackguard() }
func stack3024() (uintptr, uintptr) { var buf [3024]byte; use(buf[:]); return Stackguard() }
func stack3028() (uintptr, uintptr) { var buf [3028]byte; use(buf[:]); return Stackguard() }
func stack3032() (uintptr, uintptr) { var buf [3032]byte; use(buf[:]); return Stackguard() }
func stack3036() (uintptr, uintptr) { var buf [3036]byte; use(buf[:]); return Stackguard() }
func stack3040() (uintptr, uintptr) { var buf [3040]byte; use(buf[:]); return Stackguard() }
func stack3044() (uintptr, uintptr) { var buf [3044]byte; use(buf[:]); return Stackguard() }
func stack3048() (uintptr, uintptr) { var buf [3048]byte; use(buf[:]); return Stackguard() }
func stack3052() (uintptr, uintptr) { var buf [3052]byte; use(buf[:]); return Stackguard() }
func stack3056() (uintptr, uintptr) { var buf [3056]byte; use(buf[:]); return Stackguard() }
func stack3060() (uintptr, uintptr) { var buf [3060]byte; use(buf[:]); return Stackguard() }
func stack3064() (uintptr, uintptr) { var buf [3064]byte; use(buf[:]); return Stackguard() }
func stack3068() (uintptr, uintptr) { var buf [3068]byte; use(buf[:]); return Stackguard() }
func stack3072() (uintptr, uintptr) { var buf [3072]byte; use(buf[:]); return Stackguard() }
func stack3076() (uintptr, uintptr) { var buf [3076]byte; use(buf[:]); return Stackguard() }
func stack3080() (uintptr, uintptr) { var buf [3080]byte; use(buf[:]); return Stackguard() }
func stack3084() (uintptr, uintptr) { var buf [3084]byte; use(buf[:]); return Stackguard() }
func stack3088() (uintptr, uintptr) { var buf [3088]byte; use(buf[:]); return Stackguard() }
func stack3092() (uintptr, uintptr) { var buf [3092]byte; use(buf[:]); return Stackguard() }
func stack3096() (uintptr, uintptr) { var buf [3096]byte; use(buf[:]); return Stackguard() }
func stack3100() (uintptr, uintptr) { var buf [3100]byte; use(buf[:]); return Stackguard() }
func stack3104() (uintptr, uintptr) { var buf [3104]byte; use(buf[:]); return Stackguard() }
func stack3108() (uintptr, uintptr) { var buf [3108]byte; use(buf[:]); return Stackguard() }
func stack3112() (uintptr, uintptr) { var buf [3112]byte; use(buf[:]); return Stackguard() }
func stack3116() (uintptr, uintptr) { var buf [3116]byte; use(buf[:]); return Stackguard() }
func stack3120() (uintptr, uintptr) { var buf [3120]byte; use(buf[:]); return Stackguard() }
func stack3124() (uintptr, uintptr) { var buf [3124]byte; use(buf[:]); return Stackguard() }
func stack3128() (uintptr, uintptr) { var buf [3128]byte; use(buf[:]); return Stackguard() }
func stack3132() (uintptr, uintptr) { var buf [3132]byte; use(buf[:]); return Stackguard() }
func stack3136() (uintptr, uintptr) { var buf [3136]byte; use(buf[:]); return Stackguard() }
func stack3140() (uintptr, uintptr) { var buf [3140]byte; use(buf[:]); return Stackguard() }
func stack3144() (uintptr, uintptr) { var buf [3144]byte; use(buf[:]); return Stackguard() }
func stack3148() (uintptr, uintptr) { var buf [3148]byte; use(buf[:]); return Stackguard() }
func stack3152() (uintptr, uintptr) { var buf [3152]byte; use(buf[:]); return Stackguard() }
func stack3156() (uintptr, uintptr) { var buf [3156]byte; use(buf[:]); return Stackguard() }
func stack3160() (uintptr, uintptr) { var buf [3160]byte; use(buf[:]); return Stackguard() }
func stack3164() (uintptr, uintptr) { var buf [3164]byte; use(buf[:]); return Stackguard() }
func stack3168() (uintptr, uintptr) { var buf [3168]byte; use(buf[:]); return Stackguard() }
func stack3172() (uintptr, uintptr) { var buf [3172]byte; use(buf[:]); return Stackguard() }
func stack3176() (uintptr, uintptr) { var buf [3176]byte; use(buf[:]); return Stackguard() }
func stack3180() (uintptr, uintptr) { var buf [3180]byte; use(buf[:]); return Stackguard() }
func stack3184() (uintptr, uintptr) { var buf [3184]byte; use(buf[:]); return Stackguard() }
func stack3188() (uintptr, uintptr) { var buf [3188]byte; use(buf[:]); return Stackguard() }
func stack3192() (uintptr, uintptr) { var buf [3192]byte; use(buf[:]); return Stackguard() }
func stack3196() (uintptr, uintptr) { var buf [3196]byte; use(buf[:]); return Stackguard() }
func stack3200() (uintptr, uintptr) { var buf [3200]byte; use(buf[:]); return Stackguard() }
func stack3204() (uintptr, uintptr) { var buf [3204]byte; use(buf[:]); return Stackguard() }
func stack3208() (uintptr, uintptr) { var buf [3208]byte; use(buf[:]); return Stackguard() }
func stack3212() (uintptr, uintptr) { var buf [3212]byte; use(buf[:]); return Stackguard() }
func stack3216() (uintptr, uintptr) { var buf [3216]byte; use(buf[:]); return Stackguard() }
func stack3220() (uintptr, uintptr) { var buf [3220]byte; use(buf[:]); return Stackguard() }
func stack3224() (uintptr, uintptr) { var buf [3224]byte; use(buf[:]); return Stackguard() }
func stack3228() (uintptr, uintptr) { var buf [3228]byte; use(buf[:]); return Stackguard() }
func stack3232() (uintptr, uintptr) { var buf [3232]byte; use(buf[:]); return Stackguard() }
func stack3236() (uintptr, uintptr) { var buf [3236]byte; use(buf[:]); return Stackguard() }
func stack3240() (uintptr, uintptr) { var buf [3240]byte; use(buf[:]); return Stackguard() }
func stack3244() (uintptr, uintptr) { var buf [3244]byte; use(buf[:]); return Stackguard() }
func stack3248() (uintptr, uintptr) { var buf [3248]byte; use(buf[:]); return Stackguard() }
func stack3252() (uintptr, uintptr) { var buf [3252]byte; use(buf[:]); return Stackguard() }
func stack3256() (uintptr, uintptr) { var buf [3256]byte; use(buf[:]); return Stackguard() }
func stack3260() (uintptr, uintptr) { var buf [3260]byte; use(buf[:]); return Stackguard() }
func stack3264() (uintptr, uintptr) { var buf [3264]byte; use(buf[:]); return Stackguard() }
func stack3268() (uintptr, uintptr) { var buf [3268]byte; use(buf[:]); return Stackguard() }
func stack3272() (uintptr, uintptr) { var buf [3272]byte; use(buf[:]); return Stackguard() }
func stack3276() (uintptr, uintptr) { var buf [3276]byte; use(buf[:]); return Stackguard() }
func stack3280() (uintptr, uintptr) { var buf [3280]byte; use(buf[:]); return Stackguard() }
func stack3284() (uintptr, uintptr) { var buf [3284]byte; use(buf[:]); return Stackguard() }
func stack3288() (uintptr, uintptr) { var buf [3288]byte; use(buf[:]); return Stackguard() }
func stack3292() (uintptr, uintptr) { var buf [3292]byte; use(buf[:]); return Stackguard() }
func stack3296() (uintptr, uintptr) { var buf [3296]byte; use(buf[:]); return Stackguard() }
func stack3300() (uintptr, uintptr) { var buf [3300]byte; use(buf[:]); return Stackguard() }
func stack3304() (uintptr, uintptr) { var buf [3304]byte; use(buf[:]); return Stackguard() }
func stack3308() (uintptr, uintptr) { var buf [3308]byte; use(buf[:]); return Stackguard() }
func stack3312() (uintptr, uintptr) { var buf [3312]byte; use(buf[:]); return Stackguard() }
func stack3316() (uintptr, uintptr) { var buf [3316]byte; use(buf[:]); return Stackguard() }
func stack3320() (uintptr, uintptr) { var buf [3320]byte; use(buf[:]); return Stackguard() }
func stack3324() (uintptr, uintptr) { var buf [3324]byte; use(buf[:]); return Stackguard() }
func stack3328() (uintptr, uintptr) { var buf [3328]byte; use(buf[:]); return Stackguard() }
func stack3332() (uintptr, uintptr) { var buf [3332]byte; use(buf[:]); return Stackguard() }
func stack3336() (uintptr, uintptr) { var buf [3336]byte; use(buf[:]); return Stackguard() }
func stack3340() (uintptr, uintptr) { var buf [3340]byte; use(buf[:]); return Stackguard() }
func stack3344() (uintptr, uintptr) { var buf [3344]byte; use(buf[:]); return Stackguard() }
func stack3348() (uintptr, uintptr) { var buf [3348]byte; use(buf[:]); return Stackguard() }
func stack3352() (uintptr, uintptr) { var buf [3352]byte; use(buf[:]); return Stackguard() }
func stack3356() (uintptr, uintptr) { var buf [3356]byte; use(buf[:]); return Stackguard() }
func stack3360() (uintptr, uintptr) { var buf [3360]byte; use(buf[:]); return Stackguard() }
func stack3364() (uintptr, uintptr) { var buf [3364]byte; use(buf[:]); return Stackguard() }
func stack3368() (uintptr, uintptr) { var buf [3368]byte; use(buf[:]); return Stackguard() }
func stack3372() (uintptr, uintptr) { var buf [3372]byte; use(buf[:]); return Stackguard() }
func stack3376() (uintptr, uintptr) { var buf [3376]byte; use(buf[:]); return Stackguard() }
func stack3380() (uintptr, uintptr) { var buf [3380]byte; use(buf[:]); return Stackguard() }
func stack3384() (uintptr, uintptr) { var buf [3384]byte; use(buf[:]); return Stackguard() }
func stack3388() (uintptr, uintptr) { var buf [3388]byte; use(buf[:]); return Stackguard() }
func stack3392() (uintptr, uintptr) { var buf [3392]byte; use(buf[:]); return Stackguard() }
func stack3396() (uintptr, uintptr) { var buf [3396]byte; use(buf[:]); return Stackguard() }
func stack3400() (uintptr, uintptr) { var buf [3400]byte; use(buf[:]); return Stackguard() }
func stack3404() (uintptr, uintptr) { var buf [3404]byte; use(buf[:]); return Stackguard() }
func stack3408() (uintptr, uintptr) { var buf [3408]byte; use(buf[:]); return Stackguard() }
func stack3412() (uintptr, uintptr) { var buf [3412]byte; use(buf[:]); return Stackguard() }
func stack3416() (uintptr, uintptr) { var buf [3416]byte; use(buf[:]); return Stackguard() }
func stack3420() (uintptr, uintptr) { var buf [3420]byte; use(buf[:]); return Stackguard() }
func stack3424() (uintptr, uintptr) { var buf [3424]byte; use(buf[:]); return Stackguard() }
func stack3428() (uintptr, uintptr) { var buf [3428]byte; use(buf[:]); return Stackguard() }
func stack3432() (uintptr, uintptr) { var buf [3432]byte; use(buf[:]); return Stackguard() }
func stack3436() (uintptr, uintptr) { var buf [3436]byte; use(buf[:]); return Stackguard() }
func stack3440() (uintptr, uintptr) { var buf [3440]byte; use(buf[:]); return Stackguard() }
func stack3444() (uintptr, uintptr) { var buf [3444]byte; use(buf[:]); return Stackguard() }
func stack3448() (uintptr, uintptr) { var buf [3448]byte; use(buf[:]); return Stackguard() }
func stack3452() (uintptr, uintptr) { var buf [3452]byte; use(buf[:]); return Stackguard() }
func stack3456() (uintptr, uintptr) { var buf [3456]byte; use(buf[:]); return Stackguard() }
func stack3460() (uintptr, uintptr) { var buf [3460]byte; use(buf[:]); return Stackguard() }
func stack3464() (uintptr, uintptr) { var buf [3464]byte; use(buf[:]); return Stackguard() }
func stack3468() (uintptr, uintptr) { var buf [3468]byte; use(buf[:]); return Stackguard() }
func stack3472() (uintptr, uintptr) { var buf [3472]byte; use(buf[:]); return Stackguard() }
func stack3476() (uintptr, uintptr) { var buf [3476]byte; use(buf[:]); return Stackguard() }
func stack3480() (uintptr, uintptr) { var buf [3480]byte; use(buf[:]); return Stackguard() }
func stack3484() (uintptr, uintptr) { var buf [3484]byte; use(buf[:]); return Stackguard() }
func stack3488() (uintptr, uintptr) { var buf [3488]byte; use(buf[:]); return Stackguard() }
func stack3492() (uintptr, uintptr) { var buf [3492]byte; use(buf[:]); return Stackguard() }
func stack3496() (uintptr, uintptr) { var buf [3496]byte; use(buf[:]); return Stackguard() }
func stack3500() (uintptr, uintptr) { var buf [3500]byte; use(buf[:]); return Stackguard() }
func stack3504() (uintptr, uintptr) { var buf [3504]byte; use(buf[:]); return Stackguard() }
func stack3508() (uintptr, uintptr) { var buf [3508]byte; use(buf[:]); return Stackguard() }
func stack3512() (uintptr, uintptr) { var buf [3512]byte; use(buf[:]); return Stackguard() }
func stack3516() (uintptr, uintptr) { var buf [3516]byte; use(buf[:]); return Stackguard() }
func stack3520() (uintptr, uintptr) { var buf [3520]byte; use(buf[:]); return Stackguard() }
func stack3524() (uintptr, uintptr) { var buf [3524]byte; use(buf[:]); return Stackguard() }
func stack3528() (uintptr, uintptr) { var buf [3528]byte; use(buf[:]); return Stackguard() }
func stack3532() (uintptr, uintptr) { var buf [3532]byte; use(buf[:]); return Stackguard() }
func stack3536() (uintptr, uintptr) { var buf [3536]byte; use(buf[:]); return Stackguard() }
func stack3540() (uintptr, uintptr) { var buf [3540]byte; use(buf[:]); return Stackguard() }
func stack3544() (uintptr, uintptr) { var buf [3544]byte; use(buf[:]); return Stackguard() }
func stack3548() (uintptr, uintptr) { var buf [3548]byte; use(buf[:]); return Stackguard() }
func stack3552() (uintptr, uintptr) { var buf [3552]byte; use(buf[:]); return Stackguard() }
func stack3556() (uintptr, uintptr) { var buf [3556]byte; use(buf[:]); return Stackguard() }
func stack3560() (uintptr, uintptr) { var buf [3560]byte; use(buf[:]); return Stackguard() }
func stack3564() (uintptr, uintptr) { var buf [3564]byte; use(buf[:]); return Stackguard() }
func stack3568() (uintptr, uintptr) { var buf [3568]byte; use(buf[:]); return Stackguard() }
func stack3572() (uintptr, uintptr) { var buf [3572]byte; use(buf[:]); return Stackguard() }
func stack3576() (uintptr, uintptr) { var buf [3576]byte; use(buf[:]); return Stackguard() }
func stack3580() (uintptr, uintptr) { var buf [3580]byte; use(buf[:]); return Stackguard() }
func stack3584() (uintptr, uintptr) { var buf [3584]byte; use(buf[:]); return Stackguard() }
func stack3588() (uintptr, uintptr) { var buf [3588]byte; use(buf[:]); return Stackguard() }
func stack3592() (uintptr, uintptr) { var buf [3592]byte; use(buf[:]); return Stackguard() }
func stack3596() (uintptr, uintptr) { var buf [3596]byte; use(buf[:]); return Stackguard() }
func stack3600() (uintptr, uintptr) { var buf [3600]byte; use(buf[:]); return Stackguard() }
func stack3604() (uintptr, uintptr) { var buf [3604]byte; use(buf[:]); return Stackguard() }
func stack3608() (uintptr, uintptr) { var buf [3608]byte; use(buf[:]); return Stackguard() }
func stack3612() (uintptr, uintptr) { var buf [3612]byte; use(buf[:]); return Stackguard() }
func stack3616() (uintptr, uintptr) { var buf [3616]byte; use(buf[:]); return Stackguard() }
func stack3620() (uintptr, uintptr) { var buf [3620]byte; use(buf[:]); return Stackguard() }
func stack3624() (uintptr, uintptr) { var buf [3624]byte; use(buf[:]); return Stackguard() }
func stack3628() (uintptr, uintptr) { var buf [3628]byte; use(buf[:]); return Stackguard() }
func stack3632() (uintptr, uintptr) { var buf [3632]byte; use(buf[:]); return Stackguard() }
func stack3636() (uintptr, uintptr) { var buf [3636]byte; use(buf[:]); return Stackguard() }
func stack3640() (uintptr, uintptr) { var buf [3640]byte; use(buf[:]); return Stackguard() }
func stack3644() (uintptr, uintptr) { var buf [3644]byte; use(buf[:]); return Stackguard() }
func stack3648() (uintptr, uintptr) { var buf [3648]byte; use(buf[:]); return Stackguard() }
func stack3652() (uintptr, uintptr) { var buf [3652]byte; use(buf[:]); return Stackguard() }
func stack3656() (uintptr, uintptr) { var buf [3656]byte; use(buf[:]); return Stackguard() }
func stack3660() (uintptr, uintptr) { var buf [3660]byte; use(buf[:]); return Stackguard() }
func stack3664() (uintptr, uintptr) { var buf [3664]byte; use(buf[:]); return Stackguard() }
func stack3668() (uintptr, uintptr) { var buf [3668]byte; use(buf[:]); return Stackguard() }
func stack3672() (uintptr, uintptr) { var buf [3672]byte; use(buf[:]); return Stackguard() }
func stack3676() (uintptr, uintptr) { var buf [3676]byte; use(buf[:]); return Stackguard() }
func stack3680() (uintptr, uintptr) { var buf [3680]byte; use(buf[:]); return Stackguard() }
func stack3684() (uintptr, uintptr) { var buf [3684]byte; use(buf[:]); return Stackguard() }
func stack3688() (uintptr, uintptr) { var buf [3688]byte; use(buf[:]); return Stackguard() }
func stack3692() (uintptr, uintptr) { var buf [3692]byte; use(buf[:]); return Stackguard() }
func stack3696() (uintptr, uintptr) { var buf [3696]byte; use(buf[:]); return Stackguard() }
func stack3700() (uintptr, uintptr) { var buf [3700]byte; use(buf[:]); return Stackguard() }
func stack3704() (uintptr, uintptr) { var buf [3704]byte; use(buf[:]); return Stackguard() }
func stack3708() (uintptr, uintptr) { var buf [3708]byte; use(buf[:]); return Stackguard() }
func stack3712() (uintptr, uintptr) { var buf [3712]byte; use(buf[:]); return Stackguard() }
func stack3716() (uintptr, uintptr) { var buf [3716]byte; use(buf[:]); return Stackguard() }
func stack3720() (uintptr, uintptr) { var buf [3720]byte; use(buf[:]); return Stackguard() }
func stack3724() (uintptr, uintptr) { var buf [3724]byte; use(buf[:]); return Stackguard() }
func stack3728() (uintptr, uintptr) { var buf [3728]byte; use(buf[:]); return Stackguard() }
func stack3732() (uintptr, uintptr) { var buf [3732]byte; use(buf[:]); return Stackguard() }
func stack3736() (uintptr, uintptr) { var buf [3736]byte; use(buf[:]); return Stackguard() }
func stack3740() (uintptr, uintptr) { var buf [3740]byte; use(buf[:]); return Stackguard() }
func stack3744() (uintptr, uintptr) { var buf [3744]byte; use(buf[:]); return Stackguard() }
func stack3748() (uintptr, uintptr) { var buf [3748]byte; use(buf[:]); return Stackguard() }
func stack3752() (uintptr, uintptr) { var buf [3752]byte; use(buf[:]); return Stackguard() }
func stack3756() (uintptr, uintptr) { var buf [3756]byte; use(buf[:]); return Stackguard() }
func stack3760() (uintptr, uintptr) { var buf [3760]byte; use(buf[:]); return Stackguard() }
func stack3764() (uintptr, uintptr) { var buf [3764]byte; use(buf[:]); return Stackguard() }
func stack3768() (uintptr, uintptr) { var buf [3768]byte; use(buf[:]); return Stackguard() }
func stack3772() (uintptr, uintptr) { var buf [3772]byte; use(buf[:]); return Stackguard() }
func stack3776() (uintptr, uintptr) { var buf [3776]byte; use(buf[:]); return Stackguard() }
func stack3780() (uintptr, uintptr) { var buf [3780]byte; use(buf[:]); return Stackguard() }
func stack3784() (uintptr, uintptr) { var buf [3784]byte; use(buf[:]); return Stackguard() }
func stack3788() (uintptr, uintptr) { var buf [3788]byte; use(buf[:]); return Stackguard() }
func stack3792() (uintptr, uintptr) { var buf [3792]byte; use(buf[:]); return Stackguard() }
func stack3796() (uintptr, uintptr) { var buf [3796]byte; use(buf[:]); return Stackguard() }
func stack3800() (uintptr, uintptr) { var buf [3800]byte; use(buf[:]); return Stackguard() }
func stack3804() (uintptr, uintptr) { var buf [3804]byte; use(buf[:]); return Stackguard() }
func stack3808() (uintptr, uintptr) { var buf [3808]byte; use(buf[:]); return Stackguard() }
func stack3812() (uintptr, uintptr) { var buf [3812]byte; use(buf[:]); return Stackguard() }
func stack3816() (uintptr, uintptr) { var buf [3816]byte; use(buf[:]); return Stackguard() }
func stack3820() (uintptr, uintptr) { var buf [3820]byte; use(buf[:]); return Stackguard() }
func stack3824() (uintptr, uintptr) { var buf [3824]byte; use(buf[:]); return Stackguard() }
func stack3828() (uintptr, uintptr) { var buf [3828]byte; use(buf[:]); return Stackguard() }
func stack3832() (uintptr, uintptr) { var buf [3832]byte; use(buf[:]); return Stackguard() }
func stack3836() (uintptr, uintptr) { var buf [3836]byte; use(buf[:]); return Stackguard() }
func stack3840() (uintptr, uintptr) { var buf [3840]byte; use(buf[:]); return Stackguard() }
func stack3844() (uintptr, uintptr) { var buf [3844]byte; use(buf[:]); return Stackguard() }
func stack3848() (uintptr, uintptr) { var buf [3848]byte; use(buf[:]); return Stackguard() }
func stack3852() (uintptr, uintptr) { var buf [3852]byte; use(buf[:]); return Stackguard() }
func stack3856() (uintptr, uintptr) { var buf [3856]byte; use(buf[:]); return Stackguard() }
func stack3860() (uintptr, uintptr) { var buf [3860]byte; use(buf[:]); return Stackguard() }
func stack3864() (uintptr, uintptr) { var buf [3864]byte; use(buf[:]); return Stackguard() }
func stack3868() (uintptr, uintptr) { var buf [3868]byte; use(buf[:]); return Stackguard() }
func stack3872() (uintptr, uintptr) { var buf [3872]byte; use(buf[:]); return Stackguard() }
func stack3876() (uintptr, uintptr) { var buf [3876]byte; use(buf[:]); return Stackguard() }
func stack3880() (uintptr, uintptr) { var buf [3880]byte; use(buf[:]); return Stackguard() }
func stack3884() (uintptr, uintptr) { var buf [3884]byte; use(buf[:]); return Stackguard() }
func stack3888() (uintptr, uintptr) { var buf [3888]byte; use(buf[:]); return Stackguard() }
func stack3892() (uintptr, uintptr) { var buf [3892]byte; use(buf[:]); return Stackguard() }
func stack3896() (uintptr, uintptr) { var buf [3896]byte; use(buf[:]); return Stackguard() }
func stack3900() (uintptr, uintptr) { var buf [3900]byte; use(buf[:]); return Stackguard() }
func stack3904() (uintptr, uintptr) { var buf [3904]byte; use(buf[:]); return Stackguard() }
func stack3908() (uintptr, uintptr) { var buf [3908]byte; use(buf[:]); return Stackguard() }
func stack3912() (uintptr, uintptr) { var buf [3912]byte; use(buf[:]); return Stackguard() }
func stack3916() (uintptr, uintptr) { var buf [3916]byte; use(buf[:]); return Stackguard() }
func stack3920() (uintptr, uintptr) { var buf [3920]byte; use(buf[:]); return Stackguard() }
func stack3924() (uintptr, uintptr) { var buf [3924]byte; use(buf[:]); return Stackguard() }
func stack3928() (uintptr, uintptr) { var buf [3928]byte; use(buf[:]); return Stackguard() }
func stack3932() (uintptr, uintptr) { var buf [3932]byte; use(buf[:]); return Stackguard() }
func stack3936() (uintptr, uintptr) { var buf [3936]byte; use(buf[:]); return Stackguard() }
func stack3940() (uintptr, uintptr) { var buf [3940]byte; use(buf[:]); return Stackguard() }
func stack3944() (uintptr, uintptr) { var buf [3944]byte; use(buf[:]); return Stackguard() }
func stack3948() (uintptr, uintptr) { var buf [3948]byte; use(buf[:]); return Stackguard() }
func stack3952() (uintptr, uintptr) { var buf [3952]byte; use(buf[:]); return Stackguard() }
func stack3956() (uintptr, uintptr) { var buf [3956]byte; use(buf[:]); return Stackguard() }
func stack3960() (uintptr, uintptr) { var buf [3960]byte; use(buf[:]); return Stackguard() }
func stack3964() (uintptr, uintptr) { var buf [3964]byte; use(buf[:]); return Stackguard() }
func stack3968() (uintptr, uintptr) { var buf [3968]byte; use(buf[:]); return Stackguard() }
func stack3972() (uintptr, uintptr) { var buf [3972]byte; use(buf[:]); return Stackguard() }
func stack3976() (uintptr, uintptr) { var buf [3976]byte; use(buf[:]); return Stackguard() }
func stack3980() (uintptr, uintptr) { var buf [3980]byte; use(buf[:]); return Stackguard() }
func stack3984() (uintptr, uintptr) { var buf [3984]byte; use(buf[:]); return Stackguard() }
func stack3988() (uintptr, uintptr) { var buf [3988]byte; use(buf[:]); return Stackguard() }
func stack3992() (uintptr, uintptr) { var buf [3992]byte; use(buf[:]); return Stackguard() }
func stack3996() (uintptr, uintptr) { var buf [3996]byte; use(buf[:]); return Stackguard() }
func stack4000() (uintptr, uintptr) { var buf [4000]byte; use(buf[:]); return Stackguard() }
func stack4004() (uintptr, uintptr) { var buf [4004]byte; use(buf[:]); return Stackguard() }
func stack4008() (uintptr, uintptr) { var buf [4008]byte; use(buf[:]); return Stackguard() }
func stack4012() (uintptr, uintptr) { var buf [4012]byte; use(buf[:]); return Stackguard() }
func stack4016() (uintptr, uintptr) { var buf [4016]byte; use(buf[:]); return Stackguard() }
func stack4020() (uintptr, uintptr) { var buf [4020]byte; use(buf[:]); return Stackguard() }
func stack4024() (uintptr, uintptr) { var buf [4024]byte; use(buf[:]); return Stackguard() }
func stack4028() (uintptr, uintptr) { var buf [4028]byte; use(buf[:]); return Stackguard() }
func stack4032() (uintptr, uintptr) { var buf [4032]byte; use(buf[:]); return Stackguard() }
func stack4036() (uintptr, uintptr) { var buf [4036]byte; use(buf[:]); return Stackguard() }
func stack4040() (uintptr, uintptr) { var buf [4040]byte; use(buf[:]); return Stackguard() }
func stack4044() (uintptr, uintptr) { var buf [4044]byte; use(buf[:]); return Stackguard() }
func stack4048() (uintptr, uintptr) { var buf [4048]byte; use(buf[:]); return Stackguard() }
func stack4052() (uintptr, uintptr) { var buf [4052]byte; use(buf[:]); return Stackguard() }
func stack4056() (uintptr, uintptr) { var buf [4056]byte; use(buf[:]); return Stackguard() }
func stack4060() (uintptr, uintptr) { var buf [4060]byte; use(buf[:]); return Stackguard() }
func stack4064() (uintptr, uintptr) { var buf [4064]byte; use(buf[:]); return Stackguard() }
func stack4068() (uintptr, uintptr) { var buf [4068]byte; use(buf[:]); return Stackguard() }
func stack4072() (uintptr, uintptr) { var buf [4072]byte; use(buf[:]); return Stackguard() }
func stack4076() (uintptr, uintptr) { var buf [4076]byte; use(buf[:]); return Stackguard() }
func stack4080() (uintptr, uintptr) { var buf [4080]byte; use(buf[:]); return Stackguard() }
func stack4084() (uintptr, uintptr) { var buf [4084]byte; use(buf[:]); return Stackguard() }
func stack4088() (uintptr, uintptr) { var buf [4088]byte; use(buf[:]); return Stackguard() }
func stack4092() (uintptr, uintptr) { var buf [4092]byte; use(buf[:]); return Stackguard() }
func stack4096() (uintptr, uintptr) { var buf [4096]byte; use(buf[:]); return Stackguard() }
func stack4100() (uintptr, uintptr) { var buf [4100]byte; use(buf[:]); return Stackguard() }
func stack4104() (uintptr, uintptr) { var buf [4104]byte; use(buf[:]); return Stackguard() }
func stack4108() (uintptr, uintptr) { var buf [4108]byte; use(buf[:]); return Stackguard() }
func stack4112() (uintptr, uintptr) { var buf [4112]byte; use(buf[:]); return Stackguard() }
func stack4116() (uintptr, uintptr) { var buf [4116]byte; use(buf[:]); return Stackguard() }
func stack4120() (uintptr, uintptr) { var buf [4120]byte; use(buf[:]); return Stackguard() }
func stack4124() (uintptr, uintptr) { var buf [4124]byte; use(buf[:]); return Stackguard() }
func stack4128() (uintptr, uintptr) { var buf [4128]byte; use(buf[:]); return Stackguard() }
func stack4132() (uintptr, uintptr) { var buf [4132]byte; use(buf[:]); return Stackguard() }
func stack4136() (uintptr, uintptr) { var buf [4136]byte; use(buf[:]); return Stackguard() }
func stack4140() (uintptr, uintptr) { var buf [4140]byte; use(buf[:]); return Stackguard() }
func stack4144() (uintptr, uintptr) { var buf [4144]byte; use(buf[:]); return Stackguard() }
func stack4148() (uintptr, uintptr) { var buf [4148]byte; use(buf[:]); return Stackguard() }
func stack4152() (uintptr, uintptr) { var buf [4152]byte; use(buf[:]); return Stackguard() }
func stack4156() (uintptr, uintptr) { var buf [4156]byte; use(buf[:]); return Stackguard() }
func stack4160() (uintptr, uintptr) { var buf [4160]byte; use(buf[:]); return Stackguard() }
func stack4164() (uintptr, uintptr) { var buf [4164]byte; use(buf[:]); return Stackguard() }
func stack4168() (uintptr, uintptr) { var buf [4168]byte; use(buf[:]); return Stackguard() }
func stack4172() (uintptr, uintptr) { var buf [4172]byte; use(buf[:]); return Stackguard() }
func stack4176() (uintptr, uintptr) { var buf [4176]byte; use(buf[:]); return Stackguard() }
func stack4180() (uintptr, uintptr) { var buf [4180]byte; use(buf[:]); return Stackguard() }
func stack4184() (uintptr, uintptr) { var buf [4184]byte; use(buf[:]); return Stackguard() }
func stack4188() (uintptr, uintptr) { var buf [4188]byte; use(buf[:]); return Stackguard() }
func stack4192() (uintptr, uintptr) { var buf [4192]byte; use(buf[:]); return Stackguard() }
func stack4196() (uintptr, uintptr) { var buf [4196]byte; use(buf[:]); return Stackguard() }
func stack4200() (uintptr, uintptr) { var buf [4200]byte; use(buf[:]); return Stackguard() }
func stack4204() (uintptr, uintptr) { var buf [4204]byte; use(buf[:]); return Stackguard() }
func stack4208() (uintptr, uintptr) { var buf [4208]byte; use(buf[:]); return Stackguard() }
func stack4212() (uintptr, uintptr) { var buf [4212]byte; use(buf[:]); return Stackguard() }
func stack4216() (uintptr, uintptr) { var buf [4216]byte; use(buf[:]); return Stackguard() }
func stack4220() (uintptr, uintptr) { var buf [4220]byte; use(buf[:]); return Stackguard() }
func stack4224() (uintptr, uintptr) { var buf [4224]byte; use(buf[:]); return Stackguard() }
func stack4228() (uintptr, uintptr) { var buf [4228]byte; use(buf[:]); return Stackguard() }
func stack4232() (uintptr, uintptr) { var buf [4232]byte; use(buf[:]); return Stackguard() }
func stack4236() (uintptr, uintptr) { var buf [4236]byte; use(buf[:]); return Stackguard() }
func stack4240() (uintptr, uintptr) { var buf [4240]byte; use(buf[:]); return Stackguard() }
func stack4244() (uintptr, uintptr) { var buf [4244]byte; use(buf[:]); return Stackguard() }
func stack4248() (uintptr, uintptr) { var buf [4248]byte; use(buf[:]); return Stackguard() }
func stack4252() (uintptr, uintptr) { var buf [4252]byte; use(buf[:]); return Stackguard() }
func stack4256() (uintptr, uintptr) { var buf [4256]byte; use(buf[:]); return Stackguard() }
func stack4260() (uintptr, uintptr) { var buf [4260]byte; use(buf[:]); return Stackguard() }
func stack4264() (uintptr, uintptr) { var buf [4264]byte; use(buf[:]); return Stackguard() }
func stack4268() (uintptr, uintptr) { var buf [4268]byte; use(buf[:]); return Stackguard() }
func stack4272() (uintptr, uintptr) { var buf [4272]byte; use(buf[:]); return Stackguard() }
func stack4276() (uintptr, uintptr) { var buf [4276]byte; use(buf[:]); return Stackguard() }
func stack4280() (uintptr, uintptr) { var buf [4280]byte; use(buf[:]); return Stackguard() }
func stack4284() (uintptr, uintptr) { var buf [4284]byte; use(buf[:]); return Stackguard() }
func stack4288() (uintptr, uintptr) { var buf [4288]byte; use(buf[:]); return Stackguard() }
func stack4292() (uintptr, uintptr) { var buf [4292]byte; use(buf[:]); return Stackguard() }
func stack4296() (uintptr, uintptr) { var buf [4296]byte; use(buf[:]); return Stackguard() }
func stack4300() (uintptr, uintptr) { var buf [4300]byte; use(buf[:]); return Stackguard() }
func stack4304() (uintptr, uintptr) { var buf [4304]byte; use(buf[:]); return Stackguard() }
func stack4308() (uintptr, uintptr) { var buf [4308]byte; use(buf[:]); return Stackguard() }
func stack4312() (uintptr, uintptr) { var buf [4312]byte; use(buf[:]); return Stackguard() }
func stack4316() (uintptr, uintptr) { var buf [4316]byte; use(buf[:]); return Stackguard() }
func stack4320() (uintptr, uintptr) { var buf [4320]byte; use(buf[:]); return Stackguard() }
func stack4324() (uintptr, uintptr) { var buf [4324]byte; use(buf[:]); return Stackguard() }
func stack4328() (uintptr, uintptr) { var buf [4328]byte; use(buf[:]); return Stackguard() }
func stack4332() (uintptr, uintptr) { var buf [4332]byte; use(buf[:]); return Stackguard() }
func stack4336() (uintptr, uintptr) { var buf [4336]byte; use(buf[:]); return Stackguard() }
func stack4340() (uintptr, uintptr) { var buf [4340]byte; use(buf[:]); return Stackguard() }
func stack4344() (uintptr, uintptr) { var buf [4344]byte; use(buf[:]); return Stackguard() }
func stack4348() (uintptr, uintptr) { var buf [4348]byte; use(buf[:]); return Stackguard() }
func stack4352() (uintptr, uintptr) { var buf [4352]byte; use(buf[:]); return Stackguard() }
func stack4356() (uintptr, uintptr) { var buf [4356]byte; use(buf[:]); return Stackguard() }
func stack4360() (uintptr, uintptr) { var buf [4360]byte; use(buf[:]); return Stackguard() }
func stack4364() (uintptr, uintptr) { var buf [4364]byte; use(buf[:]); return Stackguard() }
func stack4368() (uintptr, uintptr) { var buf [4368]byte; use(buf[:]); return Stackguard() }
func stack4372() (uintptr, uintptr) { var buf [4372]byte; use(buf[:]); return Stackguard() }
func stack4376() (uintptr, uintptr) { var buf [4376]byte; use(buf[:]); return Stackguard() }
func stack4380() (uintptr, uintptr) { var buf [4380]byte; use(buf[:]); return Stackguard() }
func stack4384() (uintptr, uintptr) { var buf [4384]byte; use(buf[:]); return Stackguard() }
func stack4388() (uintptr, uintptr) { var buf [4388]byte; use(buf[:]); return Stackguard() }
func stack4392() (uintptr, uintptr) { var buf [4392]byte; use(buf[:]); return Stackguard() }
func stack4396() (uintptr, uintptr) { var buf [4396]byte; use(buf[:]); return Stackguard() }
func stack4400() (uintptr, uintptr) { var buf [4400]byte; use(buf[:]); return Stackguard() }
func stack4404() (uintptr, uintptr) { var buf [4404]byte; use(buf[:]); return Stackguard() }
func stack4408() (uintptr, uintptr) { var buf [4408]byte; use(buf[:]); return Stackguard() }
func stack4412() (uintptr, uintptr) { var buf [4412]byte; use(buf[:]); return Stackguard() }
func stack4416() (uintptr, uintptr) { var buf [4416]byte; use(buf[:]); return Stackguard() }
func stack4420() (uintptr, uintptr) { var buf [4420]byte; use(buf[:]); return Stackguard() }
func stack4424() (uintptr, uintptr) { var buf [4424]byte; use(buf[:]); return Stackguard() }
func stack4428() (uintptr, uintptr) { var buf [4428]byte; use(buf[:]); return Stackguard() }
func stack4432() (uintptr, uintptr) { var buf [4432]byte; use(buf[:]); return Stackguard() }
func stack4436() (uintptr, uintptr) { var buf [4436]byte; use(buf[:]); return Stackguard() }
func stack4440() (uintptr, uintptr) { var buf [4440]byte; use(buf[:]); return Stackguard() }
func stack4444() (uintptr, uintptr) { var buf [4444]byte; use(buf[:]); return Stackguard() }
func stack4448() (uintptr, uintptr) { var buf [4448]byte; use(buf[:]); return Stackguard() }
func stack4452() (uintptr, uintptr) { var buf [4452]byte; use(buf[:]); return Stackguard() }
func stack4456() (uintptr, uintptr) { var buf [4456]byte; use(buf[:]); return Stackguard() }
func stack4460() (uintptr, uintptr) { var buf [4460]byte; use(buf[:]); return Stackguard() }
func stack4464() (uintptr, uintptr) { var buf [4464]byte; use(buf[:]); return Stackguard() }
func stack4468() (uintptr, uintptr) { var buf [4468]byte; use(buf[:]); return Stackguard() }
func stack4472() (uintptr, uintptr) { var buf [4472]byte; use(buf[:]); return Stackguard() }
func stack4476() (uintptr, uintptr) { var buf [4476]byte; use(buf[:]); return Stackguard() }
func stack4480() (uintptr, uintptr) { var buf [4480]byte; use(buf[:]); return Stackguard() }
func stack4484() (uintptr, uintptr) { var buf [4484]byte; use(buf[:]); return Stackguard() }
func stack4488() (uintptr, uintptr) { var buf [4488]byte; use(buf[:]); return Stackguard() }
func stack4492() (uintptr, uintptr) { var buf [4492]byte; use(buf[:]); return Stackguard() }
func stack4496() (uintptr, uintptr) { var buf [4496]byte; use(buf[:]); return Stackguard() }
func stack4500() (uintptr, uintptr) { var buf [4500]byte; use(buf[:]); return Stackguard() }
func stack4504() (uintptr, uintptr) { var buf [4504]byte; use(buf[:]); return Stackguard() }
func stack4508() (uintptr, uintptr) { var buf [4508]byte; use(buf[:]); return Stackguard() }
func stack4512() (uintptr, uintptr) { var buf [4512]byte; use(buf[:]); return Stackguard() }
func stack4516() (uintptr, uintptr) { var buf [4516]byte; use(buf[:]); return Stackguard() }
func stack4520() (uintptr, uintptr) { var buf [4520]byte; use(buf[:]); return Stackguard() }
func stack4524() (uintptr, uintptr) { var buf [4524]byte; use(buf[:]); return Stackguard() }
func stack4528() (uintptr, uintptr) { var buf [4528]byte; use(buf[:]); return Stackguard() }
func stack4532() (uintptr, uintptr) { var buf [4532]byte; use(buf[:]); return Stackguard() }
func stack4536() (uintptr, uintptr) { var buf [4536]byte; use(buf[:]); return Stackguard() }
func stack4540() (uintptr, uintptr) { var buf [4540]byte; use(buf[:]); return Stackguard() }
func stack4544() (uintptr, uintptr) { var buf [4544]byte; use(buf[:]); return Stackguard() }
func stack4548() (uintptr, uintptr) { var buf [4548]byte; use(buf[:]); return Stackguard() }
func stack4552() (uintptr, uintptr) { var buf [4552]byte; use(buf[:]); return Stackguard() }
func stack4556() (uintptr, uintptr) { var buf [4556]byte; use(buf[:]); return Stackguard() }
func stack4560() (uintptr, uintptr) { var buf [4560]byte; use(buf[:]); return Stackguard() }
func stack4564() (uintptr, uintptr) { var buf [4564]byte; use(buf[:]); return Stackguard() }
func stack4568() (uintptr, uintptr) { var buf [4568]byte; use(buf[:]); return Stackguard() }
func stack4572() (uintptr, uintptr) { var buf [4572]byte; use(buf[:]); return Stackguard() }
func stack4576() (uintptr, uintptr) { var buf [4576]byte; use(buf[:]); return Stackguard() }
func stack4580() (uintptr, uintptr) { var buf [4580]byte; use(buf[:]); return Stackguard() }
func stack4584() (uintptr, uintptr) { var buf [4584]byte; use(buf[:]); return Stackguard() }
func stack4588() (uintptr, uintptr) { var buf [4588]byte; use(buf[:]); return Stackguard() }
func stack4592() (uintptr, uintptr) { var buf [4592]byte; use(buf[:]); return Stackguard() }
func stack4596() (uintptr, uintptr) { var buf [4596]byte; use(buf[:]); return Stackguard() }
func stack4600() (uintptr, uintptr) { var buf [4600]byte; use(buf[:]); return Stackguard() }
func stack4604() (uintptr, uintptr) { var buf [4604]byte; use(buf[:]); return Stackguard() }
func stack4608() (uintptr, uintptr) { var buf [4608]byte; use(buf[:]); return Stackguard() }
func stack4612() (uintptr, uintptr) { var buf [4612]byte; use(buf[:]); return Stackguard() }
func stack4616() (uintptr, uintptr) { var buf [4616]byte; use(buf[:]); return Stackguard() }
func stack4620() (uintptr, uintptr) { var buf [4620]byte; use(buf[:]); return Stackguard() }
func stack4624() (uintptr, uintptr) { var buf [4624]byte; use(buf[:]); return Stackguard() }
func stack4628() (uintptr, uintptr) { var buf [4628]byte; use(buf[:]); return Stackguard() }
func stack4632() (uintptr, uintptr) { var buf [4632]byte; use(buf[:]); return Stackguard() }
func stack4636() (uintptr, uintptr) { var buf [4636]byte; use(buf[:]); return Stackguard() }
func stack4640() (uintptr, uintptr) { var buf [4640]byte; use(buf[:]); return Stackguard() }
func stack4644() (uintptr, uintptr) { var buf [4644]byte; use(buf[:]); return Stackguard() }
func stack4648() (uintptr, uintptr) { var buf [4648]byte; use(buf[:]); return Stackguard() }
func stack4652() (uintptr, uintptr) { var buf [4652]byte; use(buf[:]); return Stackguard() }
func stack4656() (uintptr, uintptr) { var buf [4656]byte; use(buf[:]); return Stackguard() }
func stack4660() (uintptr, uintptr) { var buf [4660]byte; use(buf[:]); return Stackguard() }
func stack4664() (uintptr, uintptr) { var buf [4664]byte; use(buf[:]); return Stackguard() }
func stack4668() (uintptr, uintptr) { var buf [4668]byte; use(buf[:]); return Stackguard() }
func stack4672() (uintptr, uintptr) { var buf [4672]byte; use(buf[:]); return Stackguard() }
func stack4676() (uintptr, uintptr) { var buf [4676]byte; use(buf[:]); return Stackguard() }
func stack4680() (uintptr, uintptr) { var buf [4680]byte; use(buf[:]); return Stackguard() }
func stack4684() (uintptr, uintptr) { var buf [4684]byte; use(buf[:]); return Stackguard() }
func stack4688() (uintptr, uintptr) { var buf [4688]byte; use(buf[:]); return Stackguard() }
func stack4692() (uintptr, uintptr) { var buf [4692]byte; use(buf[:]); return Stackguard() }
func stack4696() (uintptr, uintptr) { var buf [4696]byte; use(buf[:]); return Stackguard() }
func stack4700() (uintptr, uintptr) { var buf [4700]byte; use(buf[:]); return Stackguard() }
func stack4704() (uintptr, uintptr) { var buf [4704]byte; use(buf[:]); return Stackguard() }
func stack4708() (uintptr, uintptr) { var buf [4708]byte; use(buf[:]); return Stackguard() }
func stack4712() (uintptr, uintptr) { var buf [4712]byte; use(buf[:]); return Stackguard() }
func stack4716() (uintptr, uintptr) { var buf [4716]byte; use(buf[:]); return Stackguard() }
func stack4720() (uintptr, uintptr) { var buf [4720]byte; use(buf[:]); return Stackguard() }
func stack4724() (uintptr, uintptr) { var buf [4724]byte; use(buf[:]); return Stackguard() }
func stack4728() (uintptr, uintptr) { var buf [4728]byte; use(buf[:]); return Stackguard() }
func stack4732() (uintptr, uintptr) { var buf [4732]byte; use(buf[:]); return Stackguard() }
func stack4736() (uintptr, uintptr) { var buf [4736]byte; use(buf[:]); return Stackguard() }
func stack4740() (uintptr, uintptr) { var buf [4740]byte; use(buf[:]); return Stackguard() }
func stack4744() (uintptr, uintptr) { var buf [4744]byte; use(buf[:]); return Stackguard() }
func stack4748() (uintptr, uintptr) { var buf [4748]byte; use(buf[:]); return Stackguard() }
func stack4752() (uintptr, uintptr) { var buf [4752]byte; use(buf[:]); return Stackguard() }
func stack4756() (uintptr, uintptr) { var buf [4756]byte; use(buf[:]); return Stackguard() }
func stack4760() (uintptr, uintptr) { var buf [4760]byte; use(buf[:]); return Stackguard() }
func stack4764() (uintptr, uintptr) { var buf [4764]byte; use(buf[:]); return Stackguard() }
func stack4768() (uintptr, uintptr) { var buf [4768]byte; use(buf[:]); return Stackguard() }
func stack4772() (uintptr, uintptr) { var buf [4772]byte; use(buf[:]); return Stackguard() }
func stack4776() (uintptr, uintptr) { var buf [4776]byte; use(buf[:]); return Stackguard() }
func stack4780() (uintptr, uintptr) { var buf [4780]byte; use(buf[:]); return Stackguard() }
func stack4784() (uintptr, uintptr) { var buf [4784]byte; use(buf[:]); return Stackguard() }
func stack4788() (uintptr, uintptr) { var buf [4788]byte; use(buf[:]); return Stackguard() }
func stack4792() (uintptr, uintptr) { var buf [4792]byte; use(buf[:]); return Stackguard() }
func stack4796() (uintptr, uintptr) { var buf [4796]byte; use(buf[:]); return Stackguard() }
func stack4800() (uintptr, uintptr) { var buf [4800]byte; use(buf[:]); return Stackguard() }
func stack4804() (uintptr, uintptr) { var buf [4804]byte; use(buf[:]); return Stackguard() }
func stack4808() (uintptr, uintptr) { var buf [4808]byte; use(buf[:]); return Stackguard() }
func stack4812() (uintptr, uintptr) { var buf [4812]byte; use(buf[:]); return Stackguard() }
func stack4816() (uintptr, uintptr) { var buf [4816]byte; use(buf[:]); return Stackguard() }
func stack4820() (uintptr, uintptr) { var buf [4820]byte; use(buf[:]); return Stackguard() }
func stack4824() (uintptr, uintptr) { var buf [4824]byte; use(buf[:]); return Stackguard() }
func stack4828() (uintptr, uintptr) { var buf [4828]byte; use(buf[:]); return Stackguard() }
func stack4832() (uintptr, uintptr) { var buf [4832]byte; use(buf[:]); return Stackguard() }
func stack4836() (uintptr, uintptr) { var buf [4836]byte; use(buf[:]); return Stackguard() }
func stack4840() (uintptr, uintptr) { var buf [4840]byte; use(buf[:]); return Stackguard() }
func stack4844() (uintptr, uintptr) { var buf [4844]byte; use(buf[:]); return Stackguard() }
func stack4848() (uintptr, uintptr) { var buf [4848]byte; use(buf[:]); return Stackguard() }
func stack4852() (uintptr, uintptr) { var buf [4852]byte; use(buf[:]); return Stackguard() }
func stack4856() (uintptr, uintptr) { var buf [4856]byte; use(buf[:]); return Stackguard() }
func stack4860() (uintptr, uintptr) { var buf [4860]byte; use(buf[:]); return Stackguard() }
func stack4864() (uintptr, uintptr) { var buf [4864]byte; use(buf[:]); return Stackguard() }
func stack4868() (uintptr, uintptr) { var buf [4868]byte; use(buf[:]); return Stackguard() }
func stack4872() (uintptr, uintptr) { var buf [4872]byte; use(buf[:]); return Stackguard() }
func stack4876() (uintptr, uintptr) { var buf [4876]byte; use(buf[:]); return Stackguard() }
func stack4880() (uintptr, uintptr) { var buf [4880]byte; use(buf[:]); return Stackguard() }
func stack4884() (uintptr, uintptr) { var buf [4884]byte; use(buf[:]); return Stackguard() }
func stack4888() (uintptr, uintptr) { var buf [4888]byte; use(buf[:]); return Stackguard() }
func stack4892() (uintptr, uintptr) { var buf [4892]byte; use(buf[:]); return Stackguard() }
func stack4896() (uintptr, uintptr) { var buf [4896]byte; use(buf[:]); return Stackguard() }
func stack4900() (uintptr, uintptr) { var buf [4900]byte; use(buf[:]); return Stackguard() }
func stack4904() (uintptr, uintptr) { var buf [4904]byte; use(buf[:]); return Stackguard() }
func stack4908() (uintptr, uintptr) { var buf [4908]byte; use(buf[:]); return Stackguard() }
func stack4912() (uintptr, uintptr) { var buf [4912]byte; use(buf[:]); return Stackguard() }
func stack4916() (uintptr, uintptr) { var buf [4916]byte; use(buf[:]); return Stackguard() }
func stack4920() (uintptr, uintptr) { var buf [4920]byte; use(buf[:]); return Stackguard() }
func stack4924() (uintptr, uintptr) { var buf [4924]byte; use(buf[:]); return Stackguard() }
func stack4928() (uintptr, uintptr) { var buf [4928]byte; use(buf[:]); return Stackguard() }
func stack4932() (uintptr, uintptr) { var buf [4932]byte; use(buf[:]); return Stackguard() }
func stack4936() (uintptr, uintptr) { var buf [4936]byte; use(buf[:]); return Stackguard() }
func stack4940() (uintptr, uintptr) { var buf [4940]byte; use(buf[:]); return Stackguard() }
func stack4944() (uintptr, uintptr) { var buf [4944]byte; use(buf[:]); return Stackguard() }
func stack4948() (uintptr, uintptr) { var buf [4948]byte; use(buf[:]); return Stackguard() }
func stack4952() (uintptr, uintptr) { var buf [4952]byte; use(buf[:]); return Stackguard() }
func stack4956() (uintptr, uintptr) { var buf [4956]byte; use(buf[:]); return Stackguard() }
func stack4960() (uintptr, uintptr) { var buf [4960]byte; use(buf[:]); return Stackguard() }
func stack4964() (uintptr, uintptr) { var buf [4964]byte; use(buf[:]); return Stackguard() }
func stack4968() (uintptr, uintptr) { var buf [4968]byte; use(buf[:]); return Stackguard() }
func stack4972() (uintptr, uintptr) { var buf [4972]byte; use(buf[:]); return Stackguard() }
func stack4976() (uintptr, uintptr) { var buf [4976]byte; use(buf[:]); return Stackguard() }
func stack4980() (uintptr, uintptr) { var buf [4980]byte; use(buf[:]); return Stackguard() }
func stack4984() (uintptr, uintptr) { var buf [4984]byte; use(buf[:]); return Stackguard() }
func stack4988() (uintptr, uintptr) { var buf [4988]byte; use(buf[:]); return Stackguard() }
func stack4992() (uintptr, uintptr) { var buf [4992]byte; use(buf[:]); return Stackguard() }
func stack4996() (uintptr, uintptr) { var buf [4996]byte; use(buf[:]); return Stackguard() }
func stack5000() (uintptr, uintptr) { var buf [5000]byte; use(buf[:]); return Stackguard() }

// TestStackMem measures per-thread stack segment cache behavior.
// The test consumed up to 500MB in the past.
func TestStackMem(t *testing.T) {
	const (
		BatchSize      = 32
		BatchCount     = 256
		ArraySize      = 1024
		RecursionDepth = 128
	)
	if testing.Short() {
		return
	}
	defer GOMAXPROCS(GOMAXPROCS(BatchSize))
	s0 := new(MemStats)
	ReadMemStats(s0)
	for b := 0; b < BatchCount; b++ {
		c := make(chan bool, BatchSize)
		for i := 0; i < BatchSize; i++ {
			go func() {
				var f func(k int, a [ArraySize]byte)
				f = func(k int, a [ArraySize]byte) {
					if k == 0 {
						time.Sleep(time.Millisecond)
						return
					}
					f(k-1, a)
				}
				f(RecursionDepth, [ArraySize]byte{})
				c <- true
			}()
		}
		for i := 0; i < BatchSize; i++ {
			<-c
		}

		// The goroutines have signaled via c that they are ready to exit.
		// Give them a chance to exit by sleeping. If we don't wait, we
		// might not reuse them on the next batch.
		time.Sleep(10 * time.Millisecond)
	}
	s1 := new(MemStats)
	ReadMemStats(s1)
	consumed := s1.StackSys - s0.StackSys
	t.Logf("Consumed %vMB for stack mem", consumed>>20)
	estimate := uint64(8 * BatchSize * ArraySize * RecursionDepth) // 8 is to reduce flakiness.
	if consumed > estimate {
		t.Fatalf("Stack mem: want %v, got %v", estimate, consumed)
	}
	inuse := s1.StackInuse - s0.StackInuse
	t.Logf("Inuse %vMB for stack mem", inuse>>20)
	if inuse > 4<<20 {
		t.Fatalf("Stack inuse: want %v, got %v", 4<<20, inuse)
	}
}
