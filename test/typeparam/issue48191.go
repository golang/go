// compile -c=2

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I1 interface {
	int8 | int16 | int32 | int64 | int | uint
}
type I2 interface{ float32 | float64 }
type I3 interface{ string }

func F[G1 I1, G2 I2, G3 I3]() {
	var m0 map[G2]rune
	var ch0, ch1 chan bool
	var ast0, ast1 []struct{ s0 G3 }
	var ai64_2 []int64
	var m1, m2, m3 map[bool]map[int]struct {
		m0 map[G2]byte
		s1 G3
	}
	var i8_0, i8_1 G1
	var i16_0 int16
	var am3, am4 []map[float64]map[G2]*func(*byte, map[uint]int64, G3, struct{}) G2
	var pi64_0, pi64_1 *int64
	var i, i1, i2 int
	var as5, as6, as7 []G3
	var ch2, ch3, ch4 chan uint
	var m4, m5, m6 map[G1]chan bool

	if func(G2, int32) byte {
		return m1[false][30].m0[G2(28.6)] * m3[func(bool, uint) bool {
			return false
		}(false, uint(94))][31].m0[G2(185.0)] * m1[(true || true) && (false && false)][51-i2].m0[G2(278.6)]
	}(G2(672.5), int32(35)) < m3[<-m5[func(int64, int64) G1 {
		return i8_1
	}(*pi64_0, int64(50))]][15&i1^i2^i2].m0[G2(895.3)] || (func(int64, uint) uint {
		return uint(94)
	}(int64(30), uint(95))&^<-ch2^<-ch4)&<-ch2^<-ch4 == <-ch2 {
		var f0 float64
		var pf2 *float64
		var ch5, ch6 chan int16
		var fnc0 func(*int64, G2, struct {
			i8_0  G1
			m1    map[float64]bool
			i64_2 int64
		}, map[byte]func(G2, float64, *uint, float64) struct{}) complex128 = func(p0 *int64, p1 G2, p2 struct {
			i8_0  G1
			m1    map[float64]bool
			i64_2 int64
		}, p3 map[byte]func(G2, float64, *uint, float64) struct{}) complex128 {
			p0 = pi64_1
			m5 = map[G1]chan bool{(p2.i8_0 + i8_1 + i8_1 ^ i8_1) * p2.i8_0 / p2.i8_0: m4[p2.i8_0>><-ch2]}
			return (2.65i - 31.18i) * func(byte, byte) complex128 {
				return 13.12i - 32.90i + (44.15i - 70.53i - (87.16i*92.67i + (24.18i - 9.13i))) + (func(G1, int16) complex128 {
					return 55.80i
				}(G1(30), int16(80)) + 8.48i*79.18i + (37.30i*73.81i + (21.01i - 76.30i)) + func(G3, G2) complex128 {
					return 35.58i
				}(G3("2JYizeFiEMvXLkUR"), p1)*(81.59i-21.76i))
			}(m1[<-m5[G1(37)*i8_1<<i8_1%p2.i8_0]][i2].m0[p1], m1[<-ch0][55&i2/i2^i].m0[func(G3, float64) G2 {
				return G2(619.2)
			}(G3(""), 954.0)])
		}
		var m7 map[G2]int64
		var ch7 chan byte
		var fnc1 func(bool, func(chan G2, struct {
			h0 G2
		}, int64) **rune, int) map[complex128]int32 = func(p0 bool, p1 func(chan G2, struct {
			h0 G2
		}, int64) **rune, p2 int) map[complex128]int32 {
			pf2 = pf2
			as7 = as7
			return map[complex128]int32{(94.02i - 22.19i) * (fnc0(pi64_0, G2(554.1)*G2(i1), struct {
				i8_0  G1
				m1    map[float64]bool
				i64_2 int64
			}{G1(68)*i8_0 ^ i8_0, map[float64]bool{f0: <-m6[G1(33)]}, (int64(40) ^ ai64_2[77]) % *pi64_1}, map[byte]func(G2, float64, *uint, float64) struct {
			}{func(float64, float64) byte {
				return byte(32)
			}(878.2, 984.4) + m3[true][12].m0[G2(594.0)]: nil}) - (fnc0(pi64_0, G2(241.1)+G2(i2), struct {
				i8_0  G1
				m1    map[float64]bool
				i64_2 int64
			}{i8_0, map[float64]bool{904.1: false}, int64(83) + m7[G2(357.7)]}, map[byte]func(G2, float64, *uint, float64) struct {
			}{byte(85) | m1[true][99].m0[G2(372.7)]: nil}) - (fnc0(pi64_0, G2(239.9), struct {
				i8_0  G1
				m1    map[float64]bool
				i64_2 int64
			}{G1(68), map[float64]bool{555.6: false}, int64(0)}, map[byte]func(G2, float64, *uint, float64) struct {
			}{byte(18) & <-ch7: nil}) + (88.17i - 0.55i)))): int32(73) % int32(i)}
		}
		as5[54] = as6[(len(func(bool, G2) G3 {
			return G3("")
		}(false, G2(190.8)))|i1^i1)%i1-i1] + m2[68 != i || 'M'&'\xf4'|'H'&'\u1311' >= '4'&'\uab3e'>>uint(83) && (<-m6[G1(24)%i8_0] && <-ch1)][i].s1
		i = len([]G3{ast1[2].s0})
		i16_0 = <-ch6 / i16_0 & <-ch6
		i = (i1^i|i2|i2)/i + i
		m6 = m4
		am3 = am3
		m1[G2(869.6) == G2(i2)] = m2[func(float64, rune) byte {
			return func(G3, byte) byte {
				return byte(42)
			}(G3("8iDnlygG194xl"), byte(89))
		}(*pf2, '\u9cf4')/m1[func(G3, float64) bool {
			return false
		}(G3("6MbwBSHYzr9t0zD"), 774.4)][76].m0[G2(508.0)]/m2[<-m4[i8_0]][92&^i2].m0[G2(807.0)] > m3[(int32(39)|int32(i2))&^int32(i2) < int32(i2)][89*i1&i2].m0[G2(327.5)]]
		m2[<-m4[func(G1, complex128) G1 {
			return i8_1
		}(i8_0, 35.01i)] && <-m4[func(int, G1) G1 {
			return G1(0)
		}(10, G1(70))*i8_1&i8_1>><-ch2] || fnc0(pi64_0, G2(689.5), struct {
			i8_0  G1
			m1    map[float64]bool
			i64_2 int64
		}{(G1(78)*i8_1 - i8_1) / i8_1, map[float64]bool{499.2: <-m6[G1(88)^i8_0]}, int64(83) &^ ai64_2[33] & *pi64_1 * ai64_2[i1]}, map[byte]func(G2, float64, *uint, float64) struct {
		}{m1[len(G3("bNIJZq")+G3("Fri5pn1MsZzYtsaV7b")) >= i][i^i1].m0[G2(691.7)]: nil}) != 71.77i-34.84i] = map[int]struct {
			m0 map[G2]byte
			s1 G3
		}{((18+i2)&^i2%i2 ^ i) / i: m3[(G2(267.1)*G2(i1) > G2(i2) || (false || true || (true || false))) && func(int32, int64) bool {
			return <-ch0
		}(int32(63), ai64_2[61&^i1&i2])][i|i^i1]}
		i2 = 90 - i1
		_, _, _, _, _, _, _, _ = f0, pf2, ch5, ch6, fnc0, m7, ch7, fnc1
	} else {
		var m7 map[G1]chan uint
		var ch5, ch6, ch7 chan G3
		var i32_0, i32_1 int32
		var m8, m9, m10 map[bool]struct {
		}
		pi64_1 = pi64_0
		m6[func(G3, G2) G1 {
			return (G1(35) | i8_0) << i8_1 / i8_1 &^ i8_1 / i8_1
		}(G3("YBiKg"), G2(122.6))] = make(chan bool)
		ast0 = ast0
		i8_1 = (((G1(10)+i8_1)&i8_0+i8_0)&i8_0&i8_1 ^ i8_1) & i8_1
		am4 = am3
		i32_1 = int32(10) &^ i32_0
		m8[func(float64, G3) bool {
			return func(rune, int16) bool {
				return (G2(267.0)*G2(i2) == G2(i) || func(G2, G3) bool {
					return <-ch0
				}(G2(53.3), <-ch5)) && func(G2, G1) int32 {
					return int32(63)
				}(G2(804.8), G1(2))-i32_0 < i32_1
			}('\xbd', i16_0)
		}(370.9, ast0[len([]complex128{})+i-i2].s0) && (G2(245.0)-G2(i1) == G2(i1) || byte(17)&m2[false][26].m0[G2(628.5)] > m3[false][55].m0[G2(608.8)] || func(G1, G1) bool {
			return true
		}(G1(24), G1(2)) || (<-m5[G1(38)] || <-ch1) && func(int32, int) bool {
			return false && true
		}(int32(6), i1) && '\x26'&'\x27'|func(G2, G3) rune {
			return '\x13'
		}(G2(229.6), G3("ys1msVeg61uSImCDkRG3C")) <= 'V'>>uint(88)-('\xbe'+'\uafd4')) == (53.04i == 37.22i)] = m8[func(byte, int64) bool {
			return <-ch1
		}(m3[false && false][96].m0[G2(147.6)], *pi64_0) && 643.5 > float64(i1) && (<-ch0 && <-ch1)]
		i8_1 = func(byte, uint) G1 {
			return G1(68)
		}(m2[<-ch1 || <-m5[G1(96)+i8_0] || func(bool, int32) bool {
			return func(int, byte) bool {
				return m1[true][89].s1 <= G3("2ZMnHGOMQnyHSbJ")
			}(i2, m2[<-m6[G1(47)]][94].m0[G2(981.3)])
		}(<-m4[G1(0)&^i8_0&i8_0], i32_0)][i2%i&^i].m0[func(complex128, rune) G2 {
			return G2(93.1) * G2(i2)
		}(4.63i, m0[G2(975.8)])], uint(21))
		_, _, _, _, _, _, _, _, _ = m7, ch5, ch6, ch7, i32_0, i32_1, m8, m9, m10
	}

	if *pi64_0>><-ch3 <= *pi64_0 || func(bool, int32) int32 {
		return (int32(69)&^int32(i2) + int32(i2)) * int32(i2)
	}(true, int32(49))^int32(i2) >= int32(i) {
		var ai8_8, ai8_9 []G1
		var pi2, pi3, pi4 *int
		var pi8_5, pi8_6 *G1
		var i64_0, i64_1 int64
		m1[754.8*float64(i2) != float64(i) && 6.26i == 69.99i] = map[int]struct {
			m0 map[G2]byte
			s1 G3
		}{len([]G2{G2(935.9) / G2(i2), func(int64, G2) G2 {
			return G2(720.5)
		}(int64(36), G2(349.7))})&*pi2 + i2 - i1: m1[(uint(29) >= <-ch4 || int64(45)+ai64_2[18] >= *pi64_1) == (func(G2, G2) bool {
			return <-m5[G1(25)]
		}(G2(447.2), G2(946.6)) || func(int, int16) bool {
			return true
		}(40, int16(41)) && byte(51) >= m2[true][13].m0[G2(6.6)])][*pi3]}
		am4 = []map[float64]map[G2]*func(*byte, map[uint]int64, G3, struct {
		}) G2{am4[i2%*pi3]}
		pi2 = &i2
		pi64_0 = pi64_1
		ai8_8[*pi3] = *pi8_5&ai8_9[(*pi4+*pi3)%*pi3] ^ ai8_8[90+i2|*pi4]
		ai64_2 = []int64{}
		m4 = m4
		pi2 = &i1
		pi3 = &i2
		_, _, _, _, _, _, _, _, _ = ai8_8, ai8_9, pi2, pi3, pi4, pi8_5, pi8_6, i64_0, i64_1
	}

	if (true || false || int32(68) > int32(i1) || <-m5[G1(11)-i8_0] && true) && func(int, float64) bool {
		return <-m5[(G1(83)-i8_1)&^i8_1]
	}(i1, 886.6) || func(byte, int) bool {
		return 401.0/float64(i1)/float64(i1)-float64(i) == float64(i2)
	}(m1[(G1(85)^i8_1)&^i8_1 <= i8_1][72].m0[G2(617.4)], i1) || (<-m6[(G1(3)|i8_0)>><-ch2%i8_0|i8_0] || <-ch0) {
		var ch5 chan map[byte]complex128
		var fnc0 func(int32, *map[rune]complex128) complex128
		var c0 complex128
		var st0, st1, st2 struct {
		}
		var au8 []uint
		var st3, st4, st5 struct {
			ph0 *G2
			st1 struct {
				m0   map[rune]complex128
				pch1 *chan int64
				m2   map[bool]byte
				st3  struct {
					ch0 chan func(map[G1]*struct {
						pm0 *map[bool]int64
						h1  G2
					}, struct {
						u0 uint
					}, uint, float64) *struct {
						ch0 chan map[int16]G2
					}
					i1  int
					ch2 chan complex128
				}
			}
			pm2 *map[int64]struct {
				s0  G3
				pi1 *int
				st2 struct {
					m0 map[int]map[rune]int64
					r1 rune
				}
			}
		}
		var am9, am10, am11 []map[uint]int64
		m1[G3("E")+(*st4.pm2)[*pi64_0+<-*st3.st1.pch1].s0 < (*st4.pm2)[int64(46)].s0+(G3("4Jsp3pv0x")+G3("MTKt98c")+(G3("E6Nxqpl70")+G3("eXhhxb")))+(G3("siISQNeBXoQIHwGB")+G3("CzocwLRWIUD")+(G3("cDWy3E3qpeJOmw1wP9wZ")+G3("S3ZRONdtB7K1LBC"))+func(G1, uint) G3 {
			return m2[false][74].s1
		}(G1(9), uint(26)))+func(G2, int) G3 {
			return G3("WzncXvaqK4zPn")
		}(G2(291.6), i)+(ast1[(40^i1+i1)&^st4.st1.st3.i1].s0+func(byte, int64) G3 {
			return m2[207.7 == float64(i2) && (false || false)][i2].s1
		}(byte(34), am11[25][func(int32, float64) uint {
			return uint(77)
		}(int32(29), 403.1)]))] = map[int]struct {
			m0 map[G2]byte
			s1 G3
		}{st3.st1.st3.i1: m2[<-m4[i8_1]][st5.st1.st3.i1-st3.st1.st3.i1-i2]}
		st1 = struct {
		}{}
		pi64_0 = pi64_1
		m4 = m6
		as7 = as7
		m6[(i8_0+i8_0)&^i8_1&^i8_1] = m5[G1(96)^i8_1]
		st2 = struct {
		}{}
		st1 = struct {
		}{}
		am10 = []map[uint]int64{am9[len((*st4.pm2)[int64(65)].s0)+i], am11[st4.st1.st3.i1%st4.st1.st3.i1^i1]}
		i2 = st5.st1.st3.i1*i - st5.st1.st3.i1
		_, _, _, _, _, _, _, _, _, _, _, _, _ = ch5, fnc0, c0, st0, st1, st2, au8, st3, st4, st5, am9, am10, am11
	}

}

func main() {
	F[int16, float32, string]()
}
