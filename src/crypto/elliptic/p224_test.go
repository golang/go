// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

var toFromBigTests = []string{
	"0",
	"1",
	"23",
	"b70e0cb46bb4bf7f321390b94a03c1d356c01122343280d6105c1d21",
	"706a46d476dcb76798e6046d89474788d164c18032d268fd10704fa6",
}

func p224AlternativeToBig(in *p224FieldElement) *big.Int {
	ret := new(big.Int)
	tmp := new(big.Int)

	for i := len(in) - 1; i >= 0; i-- {
		ret.Lsh(ret, 28)
		tmp.SetInt64(int64(in[i]))
		ret.Add(ret, tmp)
	}
	ret.Mod(ret, P224().Params().P)
	return ret
}

func TestP224ToFromBig(t *testing.T) {
	for i, test := range toFromBigTests {
		n, _ := new(big.Int).SetString(test, 16)
		var x p224FieldElement
		p224FromBig(&x, n)
		m := p224ToBig(&x)
		if n.Cmp(m) != 0 {
			t.Errorf("#%d: %x != %x", i, n, m)
		}
		q := p224AlternativeToBig(&x)
		if n.Cmp(q) != 0 {
			t.Errorf("#%d: %x != %x (alternative)", i, n, q)
		}
	}
}

// quickCheckConfig32 will make each quickcheck test run (32 * -quickchecks)
// times. The default value of -quickchecks is 100.
var quickCheckConfig32 = &quick.Config{MaxCountScale: 32}

// weirdLimbs can be combined to generate a range of edge-case field elements.
var weirdLimbs = [...]uint32{
	0, 1, (1 << 29) - 1,
	(1 << 12), (1 << 12) - 1,
	(1 << 28), (1 << 28) - 1,
}

func generateLimb(rand *rand.Rand) uint32 {
	const bottom29Bits = 0x1fffffff
	n := rand.Intn(len(weirdLimbs) + 3)
	switch n {
	case len(weirdLimbs):
		// Random value.
		return uint32(rand.Int31n(1 << 29))
	case len(weirdLimbs) + 1:
		// Sum of two values.
		k := generateLimb(rand) + generateLimb(rand)
		return k & bottom29Bits
	case len(weirdLimbs) + 2:
		// Difference of two values.
		k := generateLimb(rand) - generateLimb(rand)
		return k & bottom29Bits
	default:
		return weirdLimbs[n]
	}
}

func (p224FieldElement) Generate(rand *rand.Rand, size int) reflect.Value {
	return reflect.ValueOf(p224FieldElement{
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
		generateLimb(rand),
	})
}

func isInBounds(x *p224FieldElement) bool {
	return bits.Len32(x[0]) <= 29 &&
		bits.Len32(x[1]) <= 29 &&
		bits.Len32(x[2]) <= 29 &&
		bits.Len32(x[3]) <= 29 &&
		bits.Len32(x[4]) <= 29 &&
		bits.Len32(x[5]) <= 29 &&
		bits.Len32(x[6]) <= 29 &&
		bits.Len32(x[7]) <= 29
}

func TestP224Mul(t *testing.T) {
	mulMatchesBigInt := func(a, b, out p224FieldElement) bool {
		var tmp p224LargeFieldElement
		p224Mul(&out, &a, &b, &tmp)

		exp := new(big.Int).Mul(p224AlternativeToBig(&a), p224AlternativeToBig(&b))
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x", a)
			t.Logf("b = %x", b)
			t.Logf("p224Mul(a, b) = %x = %v", out, got)
			t.Logf("a * b = %v", exp)
			return false
		}

		return true
	}

	a := p224FieldElement{0xfffffff, 0xfffffff, 0xf00ffff, 0x20f, 0x0, 0x0, 0x0, 0x0}
	b := p224FieldElement{1, 0, 0, 0, 0, 0, 0, 0}
	if !mulMatchesBigInt(a, b, p224FieldElement{}) {
		t.Fail()
	}

	if err := quick.Check(mulMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Square(t *testing.T) {
	squareMatchesBigInt := func(a, out p224FieldElement) bool {
		var tmp p224LargeFieldElement
		p224Square(&out, &a, &tmp)

		exp := p224AlternativeToBig(&a)
		exp.Mul(exp, exp)
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x", a)
			t.Logf("p224Square(a, b) = %x = %v", out, got)
			t.Logf("a * a = %v", exp)
			return false
		}

		return true
	}

	if err := quick.Check(squareMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Add(t *testing.T) {
	addMatchesBigInt := func(a, b, out p224FieldElement) bool {
		p224Add(&out, &a, &b)

		exp := new(big.Int).Add(p224AlternativeToBig(&a), p224AlternativeToBig(&b))
		exp.Mod(exp, P224().Params().P)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 {
			t.Logf("a = %x", a)
			t.Logf("b = %x", b)
			t.Logf("p224Add(a, b) = %x = %v", out, got)
			t.Logf("a + b = %v", exp)
			return false
		}

		return true
	}

	if err := quick.Check(addMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Reduce(t *testing.T) {
	reduceMatchesBigInt := func(a p224FieldElement) bool {
		out := a
		// TODO: generate higher values for functions like p224Reduce that are
		// expected to work with higher input bounds.
		p224Reduce(&out)

		exp := p224AlternativeToBig(&a)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 || !isInBounds(&out) {
			t.Logf("a = %x = %v", a, exp)
			t.Logf("p224Reduce(a) = %x = %v", out, got)
			return false
		}

		return true
	}

	if err := quick.Check(reduceMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Contract(t *testing.T) {
	contractMatchesBigInt := func(a, out p224FieldElement) bool {
		p224Contract(&out, &a)

		exp := p224AlternativeToBig(&a)
		got := p224AlternativeToBig(&out)
		if exp.Cmp(got) != 0 {
			t.Logf("a = %x = %v", a, exp)
			t.Logf("p224Contract(a) = %x = %v", out, got)
			return false
		}

		// Check that out < P.
		for i := range p224P {
			k := 8 - i - 1
			if out[k] > p224P[k] {
				t.Logf("p224Contract(a) = %x", out)
				return false
			}
			if out[k] < p224P[k] {
				return true
			}
		}
		t.Logf("p224Contract(a) = %x", out)
		return false
	}

	if !contractMatchesBigInt(p224P, p224FieldElement{}) {
		t.Error("p224Contract(p) is broken")
	}
	pMinus1 := p224FieldElement{0, 0, 0, 0xffff000, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(pMinus1, p224FieldElement{}) {
		t.Error("p224Contract(p - 1) is broken")
	}
	// Check that we can handle input above p, but lowest limb zero.
	a := p224FieldElement{0, 1, 0, 0xffff000, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(a, p224FieldElement{}) {
		t.Error("p224Contract(p + 2²⁸) is broken")
	}
	// Check that we can handle input above p, but lowest three limbs zero.
	b := p224FieldElement{0, 0, 0, 0xffff001, 0xfffffff, 0xfffffff, 0xfffffff, 0xfffffff}
	if !contractMatchesBigInt(b, p224FieldElement{}) {
		t.Error("p224Contract(p + 2⁸⁴) is broken")
	}

	if err := quick.Check(contractMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224IsZero(t *testing.T) {
	if got := p224IsZero(&p224FieldElement{}); got != 1 {
		t.Errorf("p224IsZero(0) = %d, expected 1", got)
	}
	if got := p224IsZero((*p224FieldElement)(&p224P)); got != 1 {
		t.Errorf("p224IsZero(p) = %d, expected 1", got)
	}
	if got := p224IsZero(&p224FieldElement{1}); got != 0 {
		t.Errorf("p224IsZero(1) = %d, expected 0", got)
	}

	isZeroMatchesBigInt := func(a p224FieldElement) bool {
		isZero := p224IsZero(&a)

		big := p224AlternativeToBig(&a)
		if big.Sign() == 0 && isZero != 1 {
			return false
		}
		if big.Sign() != 0 && isZero != 0 {
			return false
		}
		return true
	}

	if err := quick.Check(isZeroMatchesBigInt, quickCheckConfig32); err != nil {
		t.Error(err)
	}
}

func TestP224Invert(t *testing.T) {
	var out p224FieldElement

	p224Invert(&out, &p224FieldElement{})
	if got := p224IsZero(&out); got != 1 {
		t.Errorf("p224Invert(0) = %x, expected 0", out)
	}

	p224Invert(&out, (*p224FieldElement)(&p224P))
	if got := p224IsZero(&out); got != 1 {
		t.Errorf("p224Invert(p) = %x, expected 0", out)
	}

	p224Invert(&out, &p224FieldElement{1})
	p224Contract(&out, &out)
	if out != (p224FieldElement{1}) {
		t.Errorf("p224Invert(1) = %x, expected 1", out)
	}

	var tmp p224LargeFieldElement
	a := p224FieldElement{1, 2, 3, 4, 5, 6, 7, 8}
	p224Invert(&out, &a)
	p224Mul(&out, &out, &a, &tmp)
	p224Contract(&out, &out)
	if out != (p224FieldElement{1}) {
		t.Errorf("p224Invert(a) * a = %x, expected 1", out)
	}
}

type baseMultTest struct {
	k    string
	x, y string
}

var p224BaseMultTests = []baseMultTest{
	{
		"1",
		"b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21",
		"bd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34",
	},
	{
		"2",
		"706a46dc76dcb76798e60e6d89474788d16dc18032d268fd1a704fa6",
		"1c2b76a7bc25e7702a704fa986892849fca629487acf3709d2e4e8bb",
	},
	{
		"3",
		"df1b1d66a551d0d31eff822558b9d2cc75c2180279fe0d08fd896d04",
		"a3f7f03cadd0be444c0aa56830130ddf77d317344e1af3591981a925",
	},
	{
		"4",
		"ae99feebb5d26945b54892092a8aee02912930fa41cd114e40447301",
		"482580a0ec5bc47e88bc8c378632cd196cb3fa058a7114eb03054c9",
	},
	{
		"5",
		"31c49ae75bce7807cdff22055d94ee9021fedbb5ab51c57526f011aa",
		"27e8bff1745635ec5ba0c9f1c2ede15414c6507d29ffe37e790a079b",
	},
	{
		"6",
		"1f2483f82572251fca975fea40db821df8ad82a3c002ee6c57112408",
		"89faf0ccb750d99b553c574fad7ecfb0438586eb3952af5b4b153c7e",
	},
	{
		"7",
		"db2f6be630e246a5cf7d99b85194b123d487e2d466b94b24a03c3e28",
		"f3a30085497f2f611ee2517b163ef8c53b715d18bb4e4808d02b963",
	},
	{
		"8",
		"858e6f9cc6c12c31f5df124aa77767b05c8bc021bd683d2b55571550",
		"46dcd3ea5c43898c5c5fc4fdac7db39c2f02ebee4e3541d1e78047a",
	},
	{
		"9",
		"2fdcccfee720a77ef6cb3bfbb447f9383117e3daa4a07e36ed15f78d",
		"371732e4f41bf4f7883035e6a79fcedc0e196eb07b48171697517463",
	},
	{
		"10",
		"aea9e17a306517eb89152aa7096d2c381ec813c51aa880e7bee2c0fd",
		"39bb30eab337e0a521b6cba1abe4b2b3a3e524c14a3fe3eb116b655f",
	},
	{
		"11",
		"ef53b6294aca431f0f3c22dc82eb9050324f1d88d377e716448e507c",
		"20b510004092e96636cfb7e32efded8265c266dfb754fa6d6491a6da",
	},
	{
		"12",
		"6e31ee1dc137f81b056752e4deab1443a481033e9b4c93a3044f4f7a",
		"207dddf0385bfdeab6e9acda8da06b3bbef224a93ab1e9e036109d13",
	},
	{
		"13",
		"34e8e17a430e43289793c383fac9774247b40e9ebd3366981fcfaeca",
		"252819f71c7fb7fbcb159be337d37d3336d7feb963724fdfb0ecb767",
	},
	{
		"14",
		"a53640c83dc208603ded83e4ecf758f24c357d7cf48088b2ce01e9fa",
		"d5814cd724199c4a5b974a43685fbf5b8bac69459c9469bc8f23ccaf",
	},
	{
		"15",
		"baa4d8635511a7d288aebeedd12ce529ff102c91f97f867e21916bf9",
		"979a5f4759f80f4fb4ec2e34f5566d595680a11735e7b61046127989",
	},
	{
		"16",
		"b6ec4fe1777382404ef679997ba8d1cc5cd8e85349259f590c4c66d",
		"3399d464345906b11b00e363ef429221f2ec720d2f665d7dead5b482",
	},
	{
		"17",
		"b8357c3a6ceef288310e17b8bfeff9200846ca8c1942497c484403bc",
		"ff149efa6606a6bd20ef7d1b06bd92f6904639dce5174db6cc554a26",
	},
	{
		"18",
		"c9ff61b040874c0568479216824a15eab1a838a797d189746226e4cc",
		"ea98d60e5ffc9b8fcf999fab1df7e7ef7084f20ddb61bb045a6ce002",
	},
	{
		"19",
		"a1e81c04f30ce201c7c9ace785ed44cc33b455a022f2acdbc6cae83c",
		"dcf1f6c3db09c70acc25391d492fe25b4a180babd6cea356c04719cd",
	},
	{
		"20",
		"fcc7f2b45df1cd5a3c0c0731ca47a8af75cfb0347e8354eefe782455",
		"d5d7110274cba7cdee90e1a8b0d394c376a5573db6be0bf2747f530",
	},
	{
		"112233445566778899",
		"61f077c6f62ed802dad7c2f38f5c67f2cc453601e61bd076bb46179e",
		"2272f9e9f5933e70388ee652513443b5e289dd135dcc0d0299b225e4",
	},
	{
		"112233445566778899112233445566778899",
		"29895f0af496bfc62b6ef8d8a65c88c613949b03668aab4f0429e35",
		"3ea6e53f9a841f2019ec24bde1a75677aa9b5902e61081c01064de93",
	},
	{
		"6950511619965839450988900688150712778015737983940691968051900319680",
		"ab689930bcae4a4aa5f5cb085e823e8ae30fd365eb1da4aba9cf0379",
		"3345a121bbd233548af0d210654eb40bab788a03666419be6fbd34e7",
	},
	{
		"13479972933410060327035789020509431695094902435494295338570602119423",
		"bdb6a8817c1f89da1c2f3dd8e97feb4494f2ed302a4ce2bc7f5f4025",
		"4c7020d57c00411889462d77a5438bb4e97d177700bf7243a07f1680",
	},
	{
		"13479971751745682581351455311314208093898607229429740618390390702079",
		"d58b61aa41c32dd5eba462647dba75c5d67c83606c0af2bd928446a9",
		"d24ba6a837be0460dd107ae77725696d211446c5609b4595976b16bd",
	},
	{
		"13479972931865328106486971546324465392952975980343228160962702868479",
		"dc9fa77978a005510980e929a1485f63716df695d7a0c18bb518df03",
		"ede2b016f2ddffc2a8c015b134928275ce09e5661b7ab14ce0d1d403",
	},
	{
		"11795773708834916026404142434151065506931607341523388140225443265536",
		"499d8b2829cfb879c901f7d85d357045edab55028824d0f05ba279ba",
		"bf929537b06e4015919639d94f57838fa33fc3d952598dcdbb44d638",
	},
	{
		"784254593043826236572847595991346435467177662189391577090",
		"8246c999137186632c5f9eddf3b1b0e1764c5e8bd0e0d8a554b9cb77",
		"e80ed8660bc1cb17ac7d845be40a7a022d3306f116ae9f81fea65947",
	},
	{
		"13479767645505654746623887797783387853576174193480695826442858012671",
		"6670c20afcceaea672c97f75e2e9dd5c8460e54bb38538ebb4bd30eb",
		"f280d8008d07a4caf54271f993527d46ff3ff46fd1190a3f1faa4f74",
	},
	{
		"205688069665150753842126177372015544874550518966168735589597183",
		"eca934247425cfd949b795cb5ce1eff401550386e28d1a4c5a8eb",
		"d4c01040dba19628931bc8855370317c722cbd9ca6156985f1c2e9ce",
	},
	{
		"13479966930919337728895168462090683249159702977113823384618282123295",
		"ef353bf5c73cd551b96d596fbc9a67f16d61dd9fe56af19de1fba9cd",
		"21771b9cdce3e8430c09b3838be70b48c21e15bc09ee1f2d7945b91f",
	},
	{
		"50210731791415612487756441341851895584393717453129007497216",
		"4036052a3091eb481046ad3289c95d3ac905ca0023de2c03ecd451cf",
		"d768165a38a2b96f812586a9d59d4136035d9c853a5bf2e1c86a4993",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368041",
		"fcc7f2b45df1cd5a3c0c0731ca47a8af75cfb0347e8354eefe782455",
		"f2a28eefd8b345832116f1e574f2c6b2c895aa8c24941f40d8b80ad1",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368042",
		"a1e81c04f30ce201c7c9ace785ed44cc33b455a022f2acdbc6cae83c",
		"230e093c24f638f533dac6e2b6d01da3b5e7f45429315ca93fb8e634",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368043",
		"c9ff61b040874c0568479216824a15eab1a838a797d189746226e4cc",
		"156729f1a003647030666054e208180f8f7b0df2249e44fba5931fff",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368044",
		"b8357c3a6ceef288310e17b8bfeff9200846ca8c1942497c484403bc",
		"eb610599f95942df1082e4f9426d086fb9c6231ae8b24933aab5db",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368045",
		"b6ec4fe1777382404ef679997ba8d1cc5cd8e85349259f590c4c66d",
		"cc662b9bcba6f94ee4ff1c9c10bd6ddd0d138df2d099a282152a4b7f",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368046",
		"baa4d8635511a7d288aebeedd12ce529ff102c91f97f867e21916bf9",
		"6865a0b8a607f0b04b13d1cb0aa992a5a97f5ee8ca1849efb9ed8678",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368047",
		"a53640c83dc208603ded83e4ecf758f24c357d7cf48088b2ce01e9fa",
		"2a7eb328dbe663b5a468b5bc97a040a3745396ba636b964370dc3352",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368048",
		"34e8e17a430e43289793c383fac9774247b40e9ebd3366981fcfaeca",
		"dad7e608e380480434ea641cc82c82cbc92801469c8db0204f13489a",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368049",
		"6e31ee1dc137f81b056752e4deab1443a481033e9b4c93a3044f4f7a",
		"df82220fc7a4021549165325725f94c3410ddb56c54e161fc9ef62ee",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368050",
		"ef53b6294aca431f0f3c22dc82eb9050324f1d88d377e716448e507c",
		"df4aefffbf6d1699c930481cd102127c9a3d992048ab05929b6e5927",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368051",
		"aea9e17a306517eb89152aa7096d2c381ec813c51aa880e7bee2c0fd",
		"c644cf154cc81f5ade49345e541b4d4b5c1adb3eb5c01c14ee949aa2",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368052",
		"2fdcccfee720a77ef6cb3bfbb447f9383117e3daa4a07e36ed15f78d",
		"c8e8cd1b0be40b0877cfca1958603122f1e6914f84b7e8e968ae8b9e",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368053",
		"858e6f9cc6c12c31f5df124aa77767b05c8bc021bd683d2b55571550",
		"fb9232c15a3bc7673a3a03b0253824c53d0fd1411b1cabe2e187fb87",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368054",
		"db2f6be630e246a5cf7d99b85194b123d487e2d466b94b24a03c3e28",
		"f0c5cff7ab680d09ee11dae84e9c1072ac48ea2e744b1b7f72fd469e",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368055",
		"1f2483f82572251fca975fea40db821df8ad82a3c002ee6c57112408",
		"76050f3348af2664aac3a8b05281304ebc7a7914c6ad50a4b4eac383",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368056",
		"31c49ae75bce7807cdff22055d94ee9021fedbb5ab51c57526f011aa",
		"d817400e8ba9ca13a45f360e3d121eaaeb39af82d6001c8186f5f866",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368057",
		"ae99feebb5d26945b54892092a8aee02912930fa41cd114e40447301",
		"fb7da7f5f13a43b81774373c879cd32d6934c05fa758eeb14fcfab38",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368058",
		"df1b1d66a551d0d31eff822558b9d2cc75c2180279fe0d08fd896d04",
		"5c080fc3522f41bbb3f55a97cfecf21f882ce8cbb1e50ca6e67e56dc",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368059",
		"706a46dc76dcb76798e60e6d89474788d16dc18032d268fd1a704fa6",
		"e3d4895843da188fd58fb0567976d7b50359d6b78530c8f62d1b1746",
	},
	{
		"26959946667150639794667015087019625940457807714424391721682722368060",
		"b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21",
		"42c89c774a08dc04b3dd201932bc8a5ea5f8b89bbb2a7e667aff81cd",
	},
}

func TestP224BaseMult(t *testing.T) {
	p224 := P224()
	for i, e := range p224BaseMultTests {
		k, ok := new(big.Int).SetString(e.k, 10)
		if !ok {
			t.Errorf("%d: bad value for k: %s", i, e.k)
		}
		x, y := p224.ScalarBaseMult(k.Bytes())
		if fmt.Sprintf("%x", x) != e.x || fmt.Sprintf("%x", y) != e.y {
			t.Errorf("%d: bad output for k=%s: got (%x, %x), want (%s, %s)", i, e.k, x, y, e.x, e.y)
		}
		if testing.Short() && i > 5 {
			break
		}
	}
}

func TestP224GenericBaseMult(t *testing.T) {
	// We use the P224 CurveParams directly in order to test the generic implementation.
	p224 := P224().Params()
	for i, e := range p224BaseMultTests {
		k, ok := new(big.Int).SetString(e.k, 10)
		if !ok {
			t.Errorf("%d: bad value for k: %s", i, e.k)
		}
		x, y := p224.ScalarBaseMult(k.Bytes())
		if fmt.Sprintf("%x", x) != e.x || fmt.Sprintf("%x", y) != e.y {
			t.Errorf("%d: bad output for k=%s: got (%x, %x), want (%s, %s)", i, e.k, x, y, e.x, e.y)
		}
		if testing.Short() && i > 5 {
			break
		}
	}
}

func TestP224Overflow(t *testing.T) {
	// This tests for a specific bug in the P224 implementation.
	p224 := P224()
	pointData, _ := hex.DecodeString("049B535B45FB0A2072398A6831834624C7E32CCFD5A4B933BCEAF77F1DD945E08BBE5178F5EDF5E733388F196D2A631D2E075BB16CBFEEA15B")
	x, y := Unmarshal(p224, pointData)
	if !p224.IsOnCurve(x, y) {
		t.Error("P224 failed to validate a correct point")
	}
}
