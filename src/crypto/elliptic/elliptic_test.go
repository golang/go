// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elliptic

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math/big"
	"testing"
)

func TestOnCurve(t *testing.T) {
	for _, curve := range [...]Curve{
		P224(), P256(), P384(), P521(), Secp256k1(),
	} {
		curveParams := curve.Params()
		t.Logf("Testing G is on curve %v", curveParams.Name)
		if !curve.IsOnCurve(curveParams.Gx, curveParams.Gy) {
			t.Errorf("FAIL")
		}
		if !curveParams.IsOnCurve(curveParams.Gx, curveParams.Gy) {
			t.Errorf("FAIL")
		}
	}
}

func TestOffCurve(t *testing.T) {
	p224 := P224()
	x, y := new(big.Int).SetInt64(1), new(big.Int).SetInt64(1)
	if p224.IsOnCurve(x, y) {
		t.Errorf("FAIL: point off curve is claimed to be on the curve")
	}
	b := Marshal(p224, x, y)
	x1, y1 := Unmarshal(p224, b)
	if x1 != nil || y1 != nil {
		t.Errorf("FAIL: unmarshaling a point not on the curve succeeded")
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

var secp256k1BaseMultTests = []baseMultTest{
	// Thanks Chuck Batson for the test vectors:
	// https://chuckbatson.wordpress.com/2014/11/26/secp256k1-test-vectors/
	{
		"1",
		"79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
		"483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
	},
	{
		"2",
		"c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
		"1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
	},
	{
		"3",
		"f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
		"388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672",
	},
	{
		"4",
		"e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd13",
		"51ed993ea0d455b75642e2098ea51448d967ae33bfbdfe40cfe97bdc47739922",
	},
	{
		"5",
		"2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4",
		"d8ac222636e5e3d6d4dba9dda6c9c426f788271bab0d6840dca87d3aa6ac62d6",
	},
	{
		"6",
		"fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556",
		"ae12777aacfbb620f3be96017f45c560de80f0f6518fe4a03c870c36b075f297",
	},
	{
		"7",
		"5cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc",
		"6aebca40ba255960a3178d6d861a54dba813d0b813fde7b5a5082628087264da",
	},
	{
		"8",
		"2f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01",
		"5c4da8a741539949293d082a132d13b4c2e213d6ba5b7617b5da2cb76cbde904",
	},
	{
		"9",
		"acd484e2f0c7f65309ad178a9f559abde09796974c57e714c35f110dfc27ccbe",
		"cc338921b0a7d9fd64380971763b61e9add888a4375f8e0f05cc262ac64f9c37",
	},
	{
		"10",
		"a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
		"893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7",
	},
	{
		"11",
		"774ae7f858a9411e5ef4246b70c65aac5649980be5c17891bbec17895da008cb",
		"d984a032eb6b5e190243dd56d7b7b365372db1e2dff9d6a8301d74c9c953c61b",
	},
	{
		"12",
		"d01115d548e7561b15c38f004d734633687cf4419620095bc5b0f47070afe85a",
		"a9f34ffdc815e0d7a8b64537e17bd81579238c5dd9a86d526b051b13f4062327",
	},
	{
		"13",
		"f28773c2d975288bc7d1d205c3748651b075fbc6610e58cddeeddf8f19405aa8",
		"ab0902e8d880a89758212eb65cdaf473a1a06da521fa91f29b5cb52db03ed81",
	},
	{
		"14",
		"499fdf9e895e719cfd64e67f07d38e3226aa7b63678949e6e49b241a60e823e4",
		"cac2f6c4b54e855190f044e4a7b3d464464279c27a3f95bcc65f40d403a13f5b",
	},
	{
		"15",
		"d7924d4f7d43ea965a465ae3095ff41131e5946f3c85f79e44adbcf8e27e080e",
		"581e2872a86c72a683842ec228cc6defea40af2bd896d3a5c504dc9ff6a26b58",
	},
	{
		"16",
		"e60fce93b59e9ec53011aabc21c23e97b2a31369b87a5ae9c44ee89e2a6dec0a",
		"f7e3507399e595929db99f34f57937101296891e44d23f0be1f32cce69616821",
	},
	{
		"17",
		"defdea4cdb677750a420fee807eacf21eb9898ae79b9768766e4faa04a2d4a34",
		"4211ab0694635168e997b0ead2a93daeced1f4a04a95c0f6cfb199f69e56eb77",
	},
	{
		"18",
		"5601570cb47f238d2b0286db4a990fa0f3ba28d1a319f5e7cf55c2a2444da7cc",
		"c136c1dc0cbeb930e9e298043589351d81d8e0bc736ae2a1f5192e5e8b061d58",
	},
	{
		"19",
		"2b4ea0a797a443d293ef5cff444f4979f06acfebd7e86d277475656138385b6c",
		"85e89bc037945d93b343083b5a1c86131a01f60c50269763b570c854e5c09b7a",
	},
	{
		"20",
		"4ce119c96e2fa357200b559b2f7dd5a5f02d5290aff74b03f3e471b273211c97",
		"12ba26dcb10ec1625da61fa10a844c676162948271d96967450288ee9233dc3a",
	},
	{
		"112233445566778899",
		"a90cc3d3f3e146daadfc74ca1372207cb4b725ae708cef713a98edd73d99ef29",
		"5a79d6b289610c68bc3b47f3d72f9788a26a06868b4d8e433e1e2ad76fb7dc76",
	},
	{
		"112233445566778899112233445566778899",
		"e5a2636bcfd412ebf36ec45b19bfb68a1bc5f8632e678132b885f7df99c5e9b3",
		"736c1ce161ae27b405cafd2a7520370153c2c861ac51d6c1d5985d9606b45f39",
	},
	{
		"28948022309329048855892746252171976963209391069768726095651290785379540373584",
		"a6b594b38fb3e77c6edf78161fade2041f4e09fd8497db776e546c41567feb3c",
		"71444009192228730cd8237a490feba2afe3d27d7cc1136bc97e439d13330d55",
	},
	{
		"57896044618658097711785492504343953926418782139537452191302581570759080747168",
		"3b78ce563f89a0ed9414f5aa28ad0d96d6795f9c63",
		"3f3979bf72ae8202983dc989aec7f2ff2ed91bdd69ce02fc0700ca100e59ddf3",
	},
	{
		"86844066927987146567678238756515930889628173209306178286953872356138621120752",
		"e24ce4beee294aa6350faa67512b99d388693ae4e7f53d19882a6ea169fc1ce1",
		"8b71e83545fc2b5872589f99d948c03108d36797c4de363ebd3ff6a9e1a95b10",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494317",
		"4ce119c96e2fa357200b559b2f7dd5a5f02d5290aff74b03f3e471b273211c97",
		"ed45d9234ef13e9da259e05ef57bb3989e9d6b7d8e269698bafd77106dcc1ff5",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494318",
		"2b4ea0a797a443d293ef5cff444f4979f06acfebd7e86d277475656138385b6c",
		"7a17643fc86ba26c4cbcf7c4a5e379ece5fe09f3afd9689c4a8f37aa1a3f60b5",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494319",
		"5601570cb47f238d2b0286db4a990fa0f3ba28d1a319f5e7cf55c2a2444da7cc",
		"3ec93e23f34146cf161d67fbca76cae27e271f438c951d5e0ae6d1a074f9ded7",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494320",
		"defdea4cdb677750a420fee807eacf21eb9898ae79b9768766e4faa04a2d4a34",
		"bdee54f96b9cae9716684f152d56c251312e0b5fb56a3f09304e660861a910b8",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494321",
		"e60fce93b59e9ec53011aabc21c23e97b2a31369b87a5ae9c44ee89e2a6dec0a",
		"81caf8c661a6a6d624660cb0a86c8efed6976e1bb2dc0f41e0cd330969e940e",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494322",
		"d7924d4f7d43ea965a465ae3095ff41131e5946f3c85f79e44adbcf8e27e080e",
		"a7e1d78d57938d597c7bd13dd733921015bf50d427692c5a3afb235f095d90d7",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494323",
		"499fdf9e895e719cfd64e67f07d38e3226aa7b63678949e6e49b241a60e823e4",
		"353d093b4ab17aae6f0fbb1b584c2b9bb9bd863d85c06a4339a0bf2afc5ebcd4",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494324",
		"f28773c2d975288bc7d1d205c3748651b075fbc6610e58cddeeddf8f19405aa8",
		"f54f6fd17277f5768a7ded149a3250b8c5e5f925ade056e0d64a34ac24fc0eae",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494325",
		"d01115d548e7561b15c38f004d734633687cf4419620095bc5b0f47070afe85a",
		"560cb00237ea1f285749bac81e8427ea86dc73a2265792ad94fae4eb0bf9d908",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494326",
		"774ae7f858a9411e5ef4246b70c65aac5649980be5c17891bbec17895da008cb",
		"267b5fcd1494a1e6fdbc22a928484c9ac8d24e1d20062957cfe28b3536ac3614",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494327",
		"a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
		"76c545bdabe643d85c4938196c5db3969086b3d127885ea6c3411ac3fc8c9358",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494328",
		"acd484e2f0c7f65309ad178a9f559abde09796974c57e714c35f110dfc27ccbe",
		"33cc76de4f5826029bc7f68e89c49e165227775bc8a071f0fa33d9d439b05ff8",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494329",
		"2f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01",
		"a3b25758beac66b6d6c2f7d5ecd2ec4b3d1dec2945a489e84a25d3479342132b",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494330",
		"5cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc",
		"951435bf45daa69f5ce8729279e5ab2457ec2f47ec02184a5af7d9d6f78d9755",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494331",
		"fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556",
		"51ed8885530449df0c4169fe80ba3a9f217f0f09ae701b5fc378f3c84f8a0998",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494332",
		"2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4",
		"2753ddd9c91a1c292b24562259363bd90877d8e454f297bf235782c459539959",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494333",
		"e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd13",
		"ae1266c15f2baa48a9bd1df6715aebb7269851cc404201bf30168422b88c630d",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494334",
		"f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
		"c77084f09cd217ebf01cc819d5c80ca99aff5666cb3ddce4934602897b4715bd",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494335",
		"c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
		"e51e970159c23cc65c3a7be6b99315110809cd9acd992f1edc9bce55af301705",
	},
	{
		"115792089237316195423570985008687907852837564279074904382605163141518161494336",
		"79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
		"b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777",
	},
}

type scalarMultTest struct {
	k          string
	xIn, yIn   string
	xOut, yOut string
}

var p256MultTests = []scalarMultTest{
	{
		"2a265f8bcbdcaf94d58519141e578124cb40d64a501fba9c11847b28965bc737",
		"023819813ac969847059028ea88a1f30dfbcde03fc791d3a252c6b41211882ea",
		"f93e4ae433cc12cf2a43fc0ef26400c0e125508224cdb649380f25479148a4ad",
		"4d4de80f1534850d261075997e3049321a0864082d24a917863366c0724f5ae3",
		"a22d2b7f7818a3563e0f7a76c9bf0921ac55e06e2e4d11795b233824b1db8cc0",
	},
	{
		"313f72ff9fe811bf573176231b286a3bdb6f1b14e05c40146590727a71c3bccd",
		"cc11887b2d66cbae8f4d306627192522932146b42f01d3c6f92bd5c8ba739b06",
		"a2f08a029cd06b46183085bae9248b0ed15b70280c7ef13a457f5af382426031",
		"831c3f6b5f762d2f461901577af41354ac5f228c2591f84f8a6e51e2e3f17991",
		"93f90934cd0ef2c698cc471c60a93524e87ab31ca2412252337f364513e43684",
	},
}

func TestBaseMult(t *testing.T) {
	type curveBaseMultTest struct {
		curve Curve
		cases []baseMultTest
	}
	for _, tt := range []curveBaseMultTest{
		{P224(), p224BaseMultTests},
		{Secp256k1(), secp256k1BaseMultTests},
	} {
		name := tt.curve.Params().Name
		for i, e := range tt.cases {
			k, ok := new(big.Int).SetString(e.k, 10)
			if !ok {
				t.Errorf("%s %d: bad value for k: %s", name, i, e.k)
			}
			x, y := tt.curve.ScalarBaseMult(k.Bytes())
			if !tt.curve.IsOnCurve(x, y) {
				t.Errorf("%s %d: (%x, %x) not on curve", name, i, x, y)
			}
			if fmt.Sprintf("%x", x) != e.x || fmt.Sprintf("%x", y) != e.y {
				t.Errorf("%s %d: bad output for k=%s: got (%x, %x), want (%s, %s)", name, i, e.k, x, y, e.x, e.y)
			}
			if testing.Short() && i > 5 {
				break
			}
		}
	}
}

func TestGenericBaseMult(t *testing.T) {
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

func TestP256BaseMult(t *testing.T) {
	p256 := P256()
	p256Generic := p256.Params()

	scalars := make([]*big.Int, 0, len(p224BaseMultTests)+1)
	for _, e := range p224BaseMultTests {
		k, _ := new(big.Int).SetString(e.k, 10)
		scalars = append(scalars, k)
	}
	k := new(big.Int).SetInt64(1)
	k.Lsh(k, 500)
	scalars = append(scalars, k)

	for i, k := range scalars {
		x, y := p256.ScalarBaseMult(k.Bytes())
		x2, y2 := p256Generic.ScalarBaseMult(k.Bytes())
		if x.Cmp(x2) != 0 || y.Cmp(y2) != 0 {
			t.Errorf("#%d: got (%x, %x), want (%x, %x)", i, x, y, x2, y2)
		}

		if testing.Short() && i > 5 {
			break
		}
	}
}

func TestP256Mult(t *testing.T) {
	p256 := P256()
	p256Generic := p256.Params()

	for i, e := range p224BaseMultTests {
		x, _ := new(big.Int).SetString(e.x, 16)
		y, _ := new(big.Int).SetString(e.y, 16)
		k, _ := new(big.Int).SetString(e.k, 10)

		xx, yy := p256.ScalarMult(x, y, k.Bytes())
		xx2, yy2 := p256Generic.ScalarMult(x, y, k.Bytes())
		if xx.Cmp(xx2) != 0 || yy.Cmp(yy2) != 0 {
			t.Errorf("#%d: got (%x, %x), want (%x, %x)", i, xx, yy, xx2, yy2)
		}
		if testing.Short() && i > 5 {
			break
		}
	}

	for i, e := range p256MultTests {
		x, _ := new(big.Int).SetString(e.xIn, 16)
		y, _ := new(big.Int).SetString(e.yIn, 16)
		k, _ := new(big.Int).SetString(e.k, 16)
		expectedX, _ := new(big.Int).SetString(e.xOut, 16)
		expectedY, _ := new(big.Int).SetString(e.yOut, 16)

		xx, yy := p256.ScalarMult(x, y, k.Bytes())
		if xx.Cmp(expectedX) != 0 || yy.Cmp(expectedY) != 0 {
			t.Errorf("#%d: got (%x, %x), want (%x, %x)", i, xx, yy, expectedX, expectedY)
		}
	}
}

func TestInfinity(t *testing.T) {
	tests := []struct {
		name  string
		curve Curve
	}{
		{"p224", P224()},
		{"p256", P256()},
	}

	for _, test := range tests {
		curve := test.curve
		x, y := curve.ScalarBaseMult(nil)
		if x.Sign() != 0 || y.Sign() != 0 {
			t.Errorf("%s: x^0 != ∞", test.name)
		}
		x.SetInt64(0)
		y.SetInt64(0)

		x2, y2 := curve.Double(x, y)
		if x2.Sign() != 0 || y2.Sign() != 0 {
			t.Errorf("%s: 2∞ != ∞", test.name)
		}

		baseX := curve.Params().Gx
		baseY := curve.Params().Gy

		x3, y3 := curve.Add(baseX, baseY, x, y)
		if x3.Cmp(baseX) != 0 || y3.Cmp(baseY) != 0 {
			t.Errorf("%s: x+∞ != x", test.name)
		}

		x4, y4 := curve.Add(x, y, baseX, baseY)
		if x4.Cmp(baseX) != 0 || y4.Cmp(baseY) != 0 {
			t.Errorf("%s: ∞+x != x", test.name)
		}
	}
}

type synthCombinedMult struct {
	Curve
}

func (s synthCombinedMult) CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int) {
	x1, y1 := s.ScalarBaseMult(baseScalar)
	x2, y2 := s.ScalarMult(bigX, bigY, scalar)
	return s.Add(x1, y1, x2, y2)
}

func TestCombinedMult(t *testing.T) {
	type combinedMult interface {
		Curve
		CombinedMult(bigX, bigY *big.Int, baseScalar, scalar []byte) (x, y *big.Int)
	}

	p256, ok := P256().(combinedMult)
	if !ok {
		p256 = &synthCombinedMult{P256()}
	}

	gx := p256.Params().Gx
	gy := p256.Params().Gy

	zero := make([]byte, 32)
	one := make([]byte, 32)
	one[31] = 1
	two := make([]byte, 32)
	two[31] = 2

	// 0×G + 0×G = ∞
	x, y := p256.CombinedMult(gx, gy, zero, zero)
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("0×G + 0×G = (%d, %d), should be ∞", x, y)
	}

	// 1×G + 0×G = G
	x, y = p256.CombinedMult(gx, gy, one, zero)
	if x.Cmp(gx) != 0 || y.Cmp(gy) != 0 {
		t.Errorf("1×G + 0×G = (%d, %d), should be (%d, %d)", x, y, gx, gy)
	}

	// 0×G + 1×G = G
	x, y = p256.CombinedMult(gx, gy, zero, one)
	if x.Cmp(gx) != 0 || y.Cmp(gy) != 0 {
		t.Errorf("0×G + 1×G = (%d, %d), should be (%d, %d)", x, y, gx, gy)
	}

	// 1×G + 1×G = 2×G
	x, y = p256.CombinedMult(gx, gy, one, one)
	ggx, ggy := p256.ScalarBaseMult(two)
	if x.Cmp(ggx) != 0 || y.Cmp(ggy) != 0 {
		t.Errorf("1×G + 1×G = (%d, %d), should be (%d, %d)", x, y, ggx, ggy)
	}

	minusOne := new(big.Int).Sub(p256.Params().N, big.NewInt(1))
	// 1×G + (-1)×G = ∞
	x, y = p256.CombinedMult(gx, gy, one, minusOne.Bytes())
	if x.Sign() != 0 || y.Sign() != 0 {
		t.Errorf("1×G + (-1)×G = (%d, %d), should be ∞", x, y)
	}
}

func BenchmarkBaseMult(b *testing.B) {
	b.ResetTimer()
	p224 := P224()
	e := p224BaseMultTests[25]
	k, _ := new(big.Int).SetString(e.k, 10)
	b.ReportAllocs()
	b.StartTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p224.ScalarBaseMult(k.Bytes())
		}
	})
}

func BenchmarkBaseMultP256(b *testing.B) {
	b.ResetTimer()
	p256 := P256()
	e := p224BaseMultTests[25]
	k, _ := new(big.Int).SetString(e.k, 10)
	b.ReportAllocs()
	b.StartTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p256.ScalarBaseMult(k.Bytes())
		}
	})
}

func BenchmarkScalarMultP256(b *testing.B) {
	b.ResetTimer()
	p256 := P256()
	_, x, y, _ := GenerateKey(p256, rand.Reader)
	priv, _, _, _ := GenerateKey(p256, rand.Reader)

	b.ReportAllocs()
	b.StartTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p256.ScalarMult(x, y, priv)
		}
	})
}

func TestMarshal(t *testing.T) {
	p224 := P224()
	_, x, y, err := GenerateKey(p224, rand.Reader)
	if err != nil {
		t.Error(err)
		return
	}
	serialized := Marshal(p224, x, y)
	xx, yy := Unmarshal(p224, serialized)
	if xx == nil {
		t.Error("failed to unmarshal")
		return
	}
	if xx.Cmp(x) != 0 || yy.Cmp(y) != 0 {
		t.Error("unmarshal returned different values")
		return
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

// See https://golang.org/issues/20482
func TestUnmarshalToLargeCoordinates(t *testing.T) {
	curve := P256()
	p := curve.Params().P

	invalidX, invalidY := make([]byte, 65), make([]byte, 65)
	invalidX[0], invalidY[0] = 4, 4 // uncompressed encoding

	// Set x to be greater than curve's parameter P – specifically, to P+5.
	// Set y to mod_sqrt(x^3 - 3x + B)) so that (x mod P = 5 , y) is on the
	// curve.
	x := new(big.Int).Add(p, big.NewInt(5))
	y, _ := new(big.Int).SetString("31468013646237722594854082025316614106172411895747863909393730389177298123724", 10)

	copy(invalidX[1:], x.Bytes())
	copy(invalidX[33:], y.Bytes())

	if X, Y := Unmarshal(curve, invalidX); X != nil || Y != nil {
		t.Errorf("Unmarshal accepts invalid X coordinate")
	}

	// This is a point on the curve with a small y value, small enough that we can add p and still be within 32 bytes.
	x, _ = new(big.Int).SetString("31931927535157963707678568152204072984517581467226068221761862915403492091210", 10)
	y, _ = new(big.Int).SetString("5208467867388784005506817585327037698770365050895731383201516607147", 10)
	y.Add(y, p)

	if p.Cmp(y) > 0 || y.BitLen() != 256 {
		t.Fatal("y not within expected range")
	}

	// marshal
	copy(invalidY[1:], x.Bytes())
	copy(invalidY[33:], y.Bytes())

	if X, Y := Unmarshal(curve, invalidY); X != nil || Y != nil {
		t.Errorf("Unmarshal accepts invalid Y coordinate")
	}
}
