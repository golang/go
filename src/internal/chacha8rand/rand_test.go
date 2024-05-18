// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha8rand_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"internal/byteorder"
	. "internal/chacha8rand"
	"slices"
	"testing"
)

func TestOutput(t *testing.T) {
	var s State
	s.Init(seed)
	for i := range output {
		for {
			x, ok := s.Next()
			if ok {
				if x != output[i] {
					t.Errorf("#%d: have %#x want %#x", i, x, output[i])
				}
				break
			}
			s.Refill()
		}
	}
}

func TestFillRand(t *testing.T) {
	expect := make([]byte, 0, len(output)*8)
	for _, v := range output {
		expect = byteorder.LeAppendUint64(expect, v)
	}

	cases := []struct {
		expect [][]byte
	}{
		{[][]byte{expect}},
		{[][]byte{expect[:1], expect[8:9]}},
		{[][]byte{expect[:4], expect[8:12]}},
		{[][]byte{expect[:4], expect[8:12]}},
		{[][]byte{expect[:8], expect[8:16]}},
		{[][]byte{expect[:128], expect[128:256]}},
		{[][]byte{expect[:128], expect[128:]}},
		{[][]byte{expect[:100], expect[104:]}},
	}

	var s State
	for _, tt := range cases {
		s.Init(seed)
		for _, expect := range tt.expect {
			got := make([]byte, len(expect))
			s.FillRand(got)
			if !bytes.Equal(expect, got) {
				t.Errorf("got = %#x; want = %#x", got, expect)
			}
		}
	}
}

func TestMarshal(t *testing.T) {
	var s State
	s.Init(seed)
	for i := range output {
		for {
			b := Marshal(&s)
			s = State{}
			err := Unmarshal(&s, b)
			if err != nil {
				t.Fatalf("#%d: Unmarshal: %v", i, err)
			}
			x, ok := s.Next()
			if ok {
				if x != output[i] {
					t.Fatalf("#%d: have %#x want %#x", i, x, output[i])
				}
				break
			}
			s.Refill()
		}
	}
}

func TestReseed(t *testing.T) {
	var s State
	s.Init(seed)
	old := Seed(&s)
	s.Reseed()
	if Seed(&s) == old {
		t.Errorf("Reseed did not change seed")
	}
}

func BenchmarkBlock(b *testing.B) {
	var seed [4]uint64
	var blocks [32]uint64

	for i := 0; i < b.N; i++ {
		Block(&seed, &blocks, 0)
	}
	b.SetBytes(32 * 8)
}

func TestBlockGeneric(t *testing.T) {
	var b1, b2 [32]uint64
	s := seed // byte seed
	seed := [4]uint64{
		binary.LittleEndian.Uint64(s[0*8:]),
		binary.LittleEndian.Uint64(s[1*8:]),
		binary.LittleEndian.Uint64(s[2*8:]),
		binary.LittleEndian.Uint64(s[3*8:]),
	}

	Block(&seed, &b1, 4)
	Block_generic(&seed, &b2, 4)
	if !slices.Equal(b1[:], b2[:]) {
		var out bytes.Buffer
		fmt.Fprintf(&out, "%-18s %-18s\n", "block", "block_generic")
		for i := range b1 {
			suffix := ""
			if b1[i] != b2[i] {
				suffix = " mismatch!"
			}
			fmt.Fprintf(&out, "%#016x %#016x%s\n", b1[i], b2[i], suffix)
		}
		t.Errorf("block and block_generic disagree:\n%s", out.String())
	}
}

// Golden output test to make sure algorithm never changes,
// so that its use in math/rand/v2 stays stable.
// See https://c2sp.org/chacha8rand.

var seed = [32]byte([]byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"))

var output = []uint64{
	0xb773b6063d4616a5, 0x1160af22a66abc3c, 0x8c2599d9418d287c, 0x7ee07e037edc5cd6,
	0xcfaa9ee02d1c16ad, 0x0e090eef8febea79, 0x3c82d271128b5b3e, 0x9c5addc11252a34f,
	0xdf79bb617d6ceea6, 0x36d553591f9d736a, 0xeef0d14e181ee01f, 0x089bfc760ae58436,
	0xd9e52b59cc2ad268, 0xeb2fb4444b1b8aba, 0x4f95c8a692c46661, 0xc3c6323217cae62c,
	0x91ebb4367f4e2e7e, 0x784cf2c6a0ec9bc6, 0x5c34ec5c34eabe20, 0x4f0a8f515570daa8,
	0xfc35dcb4113d6bf2, 0x5b0da44c645554bc, 0x6d963da3db21d9e1, 0xeeaefc3150e500f3,
	0x2d37923dda3750a5, 0x380d7a626d4bc8b0, 0xeeaf68ede3d7ee49, 0xf4356695883b717c,
	0x846a9021392495a4, 0x8e8510549630a61b, 0x18dc02545dbae493, 0x0f8f9ff0a65a3d43,
	0xccf065f7190ff080, 0xfd76d1aa39673330, 0x95d232936cba6433, 0x6c7456d1070cbd17,
	0x462acfdaff8c6562, 0x5bafab866d34fc6a, 0x0c862f78030a2988, 0xd39a83e407c3163d,
	0xc00a2b7b45f22ebf, 0x564307c62466b1a9, 0x257e0424b0c072d4, 0x6fb55e99496c28fe,
	0xae9873a88f5cd4e0, 0x4657362ac60d3773, 0x1c83f91ecdf23e8e, 0x6fdc0792c15387c0,
	0x36dad2a30dfd2b5c, 0xa4b593290595bdb7, 0x4de18934e4cc02c5, 0xcdc0d604f015e3a7,
	0xfba0dbf69ad80321, 0x60e8bea3d139de87, 0xd18a4d851ef48756, 0x6366447c2215f34a,
	0x05682e97d3d007ee, 0x4c0e8978c6d54ab2, 0xcf1e9f6a6712edc2, 0x061439414c80cfd3,
	0xd1a8b6e2745c0ead, 0x31a7918d45c410e8, 0xabcc61ad90216eec, 0x4040d92d2032a71a,
	0x3cd2f66ffb40cd68, 0xdcd051c07295857a, 0xeab55cbcd9ab527e, 0x18471dce781bdaac,
	0xf7f08cd144dc7252, 0x5804e0b13d7f40d1, 0x5cb1a446e4b2d35b, 0xe6d4a728d2138a06,
	0x05223e40ca60dad8, 0x2d61ec3206ac6a68, 0xab692356874c17b8, 0xc30954417676de1c,
	0x4f1ace3732225624, 0xfba9510813988338, 0x997f200f52752e11, 0x1116aaafe86221fa,
	0x07ce3b5cb2a13519, 0x2956bc72bc458314, 0x4188b7926140eb78, 0x56ca6dbfd4adea4d,
	0x7fe3c22349340ce5, 0x35c08f9c37675f8a, 0x11e1c7fbef5ed521, 0x98adc8464ec1bc75,
	0xd163b2c73d1203f8, 0x8c761ee043a2f3f3, 0x24b99d6accecd7b7, 0x793e31aa112f0370,
	0x8e87dc2a19285139, 0x4247ae04f7096e25, 0x514f3122926fe20f, 0xdc6fb3f045d2a7e9,
	0x15cb30cecdd18eba, 0xcbc7fdecf6900274, 0x3fb5c696dc8ba021, 0xd1664417c8d274e6,
	0x05f7e445ea457278, 0xf920bbca1b9db657, 0x0c1950b4da22cb99, 0xf875baf1af09e292,
	0xbed3d7b84250f838, 0xf198e8080fd74160, 0xc9eda51d9b7ea703, 0xf709ef55439bf8f6,
	0xd20c74feebf116fc, 0x305668eb146d7546, 0x829af3ec10d89787, 0x15b8f9697b551dbc,
	0xfc823c6c8e64b8c9, 0x345585e8183b40bc, 0x674b4171d6581368, 0x1234d81cd670e9f7,
	0x0e505210d8a55e19, 0xe8258d69eeeca0dc, 0x05d4c452e8baf67e, 0xe8dbe30116a45599,
	0x1cf08ce1b1176f00, 0xccf7d0a4b81ecb49, 0x303fea136b2c430e, 0x861d6c139c06c871,
	0x5f41df72e05e0487, 0x25bd7e1e1ae26b1d, 0xbe9f4004d662a41d, 0x65bf58d483188546,
	0xd1b27cff69db13cc, 0x01a6663372c1bb36, 0x578dd7577b727f4d, 0x19c78f066c083cf6,
	0xdbe014d4f9c391bb, 0x97fbb2dd1d13ffb3, 0x31c91e0af9ef8d4f, 0x094dfc98402a43ba,
	0x069bd61bea37b752, 0x5b72d762e8d986ca, 0x72ee31865904bc85, 0xd1f5fdc5cd36c33e,
	0xba9b4980a8947cad, 0xece8f05eac49ab43, 0x65fe1184abae38e7, 0x2d7cb9dea5d31452,
	0xcc71489476e467e3, 0x4c03a258a578c68c, 0x00efdf9ecb0fd8fc, 0x9924cad471e2666d,
	0x87f8668318f765e9, 0xcb4dc57c1b55f5d8, 0xd373835a86604859, 0xe526568b5540e482,
	0x1f39040f08586fec, 0xb764f3f00293f8e6, 0x049443a2f6bd50a8, 0x76fec88697d3941a,
	0x3efb70d039bae7a2, 0xe2f4611368eca8a8, 0x7c007a96e01d2425, 0xbbcce5768e69c5bf,
	0x784fb4985c42aac3, 0xf72b5091aa223874, 0x3630333fb1e62e07, 0x8e7319ebdebbb8de,
	0x2a3982bca959fa00, 0xb2b98b9f964ba9b3, 0xf7e31014adb71951, 0xebd0fca3703acc82,
	0xec654e2a2fe6419a, 0xb326132d55a52e2c, 0x2248c57f44502978, 0x32710c2f342daf16,
	0x0517b47b5acb2bec, 0x4c7a718fca270937, 0xd69142bed0bcc541, 0xe40ebcb8ff52ce88,
	0x3e44a2dbc9f828d4, 0xc74c2f4f8f873f58, 0x3dbf648eb799e45b, 0x33f22475ee0e86f8,
	0x1eb4f9ee16d47f65, 0x40f8d2b8712744e3, 0xb886b4da3cb14572, 0x2086326fbdd6f64d,
	0xcc3de5907dd882b9, 0xa2e8b49a5ee909df, 0xdbfb8e7823964c10, 0x70dd6089ef0df8d5,
	0x30141663cdd9c99f, 0x04b805325c240365, 0x7483d80314ac12d6, 0x2b271cb91aa7f5f9,
	0x97e2245362abddf0, 0x5a84f614232a9fab, 0xf71125fcda4b7fa2, 0x1ca5a61d74b27267,
	0x38cc6a9b3adbcb45, 0xdde1bb85dc653e39, 0xe9d0c8fa64f89fd4, 0x02c5fb1ecd2b4188,
	0xf2bd137bca5756e5, 0xadefe25d121be155, 0x56cd1c3c5d893a8e, 0x4c50d337beb65bb9,
	0x918c5151675cf567, 0xaba649ffcfb56a1e, 0x20c74ab26a2247cd, 0x71166bac853c08da,
	0xb07befe2e584fc5d, 0xda45ff2a588dbf32, 0xdb98b03c4d75095e, 0x60285ae1aaa65a4c,
	0xf93b686a263140b8, 0xde469752ee1c180e, 0xcec232dc04129aae, 0xeb916baa1835ea04,
	0xd49c21c8b64388ff, 0x72a82d9658864888, 0x003348ef7eac66a8, 0x7f6f67e655b209eb,
	0x532ffb0b7a941b25, 0xd940ade6128deede, 0xdf24f2a1af89fe23, 0x95aa3b4988195ae0,
	0x3da649404f94be4a, 0x692dad132c3f7e27, 0x40aee76ecaaa9eb8, 0x1294a01e09655024,
	0x6df797abdba4e4f5, 0xea2fb6024c1d7032, 0x5f4e0492295489fc, 0x57972914ea22e06a,
	0x9a8137d133aad473, 0xa2e6dd6ae7cdf2f3, 0x9f42644f18086647, 0x16d03301c170bd3e,
	0x908c416fa546656d, 0xe081503be22e123e, 0x077cf09116c4cc72, 0xcbd25cd264b7f229,
	0x3db2f468ec594031, 0x46c00e734c9badd5, 0xd0ec0ac72075d861, 0x3037cb3cf80b7630,
	0x574c3d7b3a2721c6, 0xae99906a0076824b, 0xb175a5418b532e70, 0xd8b3e251ee231ddd,
	0xb433eec25dca1966, 0x530f30dc5cff9a93, 0x9ff03d98b53cd335, 0xafc4225076558cdf,
	0xef81d3a28284402a, 0x110bdbf51c110a28, 0x9ae1b255d027e8f6, 0x7de3e0aa24688332,
	0xe483c3ecd2067ee2, 0xf829328b276137e6, 0xa413ccad57562cad, 0xe6118e8b496acb1f,
	0x8288dca6da5ec01f, 0xa53777dc88c17255, 0x8a00f1e0d5716eda, 0x618e6f47b7a720a8,
	0x9e3907b0c692a841, 0x978b42ca963f34f3, 0x75e4b0cd98a7d7ef, 0xde4dbd6e0b5f4752,
	0x0252e4153f34493f, 0x50f0e7d803734ef9, 0x237766a38ed167ee, 0x4124414001ee39a0,
	0xd08df643e535bb21, 0x34f575b5a9a80b74, 0x2c343af87297f755, 0xcd8b6d99d821f7cb,
	0xe376fd7256fc48ae, 0xe1b06e7334352885, 0xfa87b26f86c169eb, 0x36c1604665a971de,
	0xdba147c2239c8e80, 0x6b208e69fc7f0e24, 0x8795395b6f2b60c3, 0x05dabee9194907f4,
	0xb98175142f5ed902, 0x5e1701e2021ddc81, 0x0875aba2755eed08, 0x778d83289251de95,
	0x3bfbe46a039ecb31, 0xb24704fce4cbd7f9, 0x6985ffe9a7c91e3d, 0xc8efb13df249dabb,
	0xb1037e64b0f4c9f6, 0x55f69fd197d6b7c3, 0x672589d71d68a90c, 0xbebdb8224f50a77e,
	0x3f589f80007374a7, 0xd307f4635954182a, 0xcff5850c10d4fd90, 0xc6da02dfb6408e15,
	0x93daeef1e2b1a485, 0x65d833208aeea625, 0xe2b13fa13ed3b5fa, 0x67053538130fb68e,
	0xc1042f6598218fa9, 0xee5badca749b8a2e, 0x6d22a3f947dae37d, 0xb62c6d1657f4dbaf,
	0x6e007de69704c20b, 0x1af2b913fc3841d8, 0xdc0e47348e2e8e22, 0x9b1ddef1cf958b22,
	0x632ed6b0233066b8, 0xddd02d3311bed8f2, 0xf147cfe1834656e9, 0x399aaa49d511597a,
	0x6b14886979ec0309, 0x64fc4ac36b5afb97, 0xb82f78e07f7cf081, 0x10925c9a323d0e1b,
	0xf451c79ee13c63f6, 0x7c2fc180317876c7, 0x35a12bd9eecb7d22, 0x335654a539621f90,
	0xcc32a3f35db581f0, 0xc60748a80b2369cb, 0x7c4dd3b08591156b, 0xac1ced4b6de22291,
	0xa32cfa2df134def5, 0x627108918dea2a53, 0x0555b1608fcb4ff4, 0x143ee7ac43aaa33c,
	0xdae90ce7cf4fc218, 0x4d68fc2582bcf4b5, 0x37094e1849135d71, 0xf7857e09f3d49fd8,
	0x007538c503768be7, 0xedf648ba2f6be601, 0xaa347664dd72513e, 0xbe63893c6ef23b86,
	0x130b85710605af97, 0xdd765c6b1ef6ab56, 0xf3249a629a97dc6b, 0x2a114f9020fab8e5,
	0x5a69e027cfc6ad08, 0x3c4ccb36f1a5e050, 0x2e9e7d596834f0a5, 0x2430be6858fce789,
	0xe90b862f2466e597, 0x895e2884f159a9ec, 0x26ab8fa4902fcb57, 0xa6efff5c54e1fa50,
	0x333ac4e5811a8255, 0xa58d515f02498611, 0xfe5a09dcb25c6ef4, 0x03898988ab5f5818,
	0x289ff6242af6c617, 0x3d9dd59fd381ea23, 0x52d7d93d8a8aae51, 0xc76a123d511f786f,
	0xf68901edaf00c46c, 0x8c630871b590de80, 0x05209c308991e091, 0x1f809f99b4788177,
	0x11170c2eb6c19fd8, 0x44433c779062ba58, 0xc0acb51af1874c45, 0x9f2e134284809fa1,
	0xedb523bd15c619fa, 0x02d97fd53ecc23c0, 0xacaf05a34462374c, 0xddd9c6d34bffa11f,
}
