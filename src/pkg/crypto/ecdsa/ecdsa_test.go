// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"big"
	"crypto/elliptic"
	"crypto/sha1"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func testKeyGeneration(t *testing.T, c *elliptic.Curve, tag string) {
	priv, err := GenerateKey(c, rand.Reader)
	if err != nil {
		t.Errorf("%s: error: %s", tag, err)
		return
	}
	if !c.IsOnCurve(priv.PublicKey.X, priv.PublicKey.Y) {
		t.Errorf("%s: public key invalid: %s", tag, err)
	}
}

func TestKeyGeneration(t *testing.T) {
	testKeyGeneration(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testKeyGeneration(t, elliptic.P256(), "p256")
	testKeyGeneration(t, elliptic.P384(), "p384")
	testKeyGeneration(t, elliptic.P521(), "p521")
}

func testSignAndVerify(t *testing.T, c *elliptic.Curve, tag string) {
	priv, _ := GenerateKey(c, rand.Reader)

	hashed := []byte("testing")
	r, s, err := Sign(rand.Reader, priv, hashed)
	if err != nil {
		t.Errorf("%s: error signing: %s", tag, err)
		return
	}

	if !Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("%s: Verify failed", tag)
	}

	hashed[0] ^= 0xff
	if Verify(&priv.PublicKey, hashed, r, s) {
		t.Errorf("%s: Verify always works!", tag)
	}
}

func TestSignAndVerify(t *testing.T) {
	testSignAndVerify(t, elliptic.P224(), "p224")
	if testing.Short() {
		return
	}
	testSignAndVerify(t, elliptic.P256(), "p256")
	testSignAndVerify(t, elliptic.P384(), "p384")
	testSignAndVerify(t, elliptic.P521(), "p521")
}

func fromHex(s string) *big.Int {
	r, ok := new(big.Int).SetString(s, 16)
	if !ok {
		panic("bad hex")
	}
	return r
}

// These test vectors were taken from
//   http://csrc.nist.gov/groups/STM/cavp/documents/dss/ecdsatestvectors.zip
var testVectors = []struct {
	msg    string
	Qx, Qy string
	r, s   string
	ok     bool
}{
	{
		"09626b45493672e48f3d1226a3aff3201960e577d33a7f72c7eb055302db8fe8ed61685dd036b554942a5737cd1512cdf811ee0c00e6dd2f08c69f08643be396e85dafda664801e772cdb7396868ac47b172245b41986aa2648cb77fbbfa562581be06651355a0c4b090f9d17d8f0ab6cced4e0c9d386cf465a516630f0231bd",
		"9504b5b82d97a264d8b3735e0568decabc4b6ca275bc53cbadfc1c40",
		"03426f80e477603b10dee670939623e3da91a94267fc4e51726009ed",
		"81d3ac609f9575d742028dd496450a58a60eea2dcf8b9842994916e1",
		"96a8c5f382c992e8f30ccce9af120b067ec1d74678fa8445232f75a5",
		false,
	},
	{
		"96b2b6536f6df29be8567a72528aceeaccbaa66c66c534f3868ca9778b02faadb182e4ed34662e73b9d52ecbe9dc8e875fc05033c493108b380689ebf47e5b062e6a0cdb3dd34ce5fe347d92768d72f7b9b377c20aea927043b509c078ed2467d7113405d2ddd458811e6faf41c403a2a239240180f1430a6f4330df5d77de37",
		"851e3100368a22478a0029353045ae40d1d8202ef4d6533cfdddafd8",
		"205302ac69457dd345e86465afa72ee8c74ca97e2b0b999aec1f10c2",
		"4450c2d38b697e990721aa2dbb56578d32b4f5aeb3b9072baa955ee0",
		"e26d4b589166f7b4ba4b1c8fce823fa47aad22f8c9c396b8c6526e12",
		false,
	},
	{
		"86778dbb4a068a01047a8d245d632f636c11d2ad350740b36fad90428b454ad0f120cb558d12ea5c8a23db595d87543d06d1ef489263d01ee529871eb68737efdb8ff85bc7787b61514bed85b7e01d6be209e0a4eb0db5c8df58a5c5bf706d76cb2bdf7800208639e05b89517155d11688236e6a47ed37d8e5a2b1e0adea338e",
		"ad5bda09d319a717c1721acd6688d17020b31b47eef1edea57ceeffc",
		"c8ce98e181770a7c9418c73c63d01494b8b80a41098c5ea50692c984",
		"de5558c257ab4134e52c19d8db3b224a1899cbd08cc508ce8721d5e9",
		"745db7af5a477e5046705c0a5eff1f52cb94a79d481f0c5a5e108ecd",
		true,
	},
	{
		"4bc6ef1958556686dab1e39c3700054a304cbd8f5928603dcd97fafd1f29e69394679b638f71c9344ce6a535d104803d22119f57b5f9477e253817a52afa9bfbc9811d6cc8c8be6b6566c6ef48b439bbb532abe30627548c598867f3861ba0b154dc1c3deca06eb28df8efd28258554b5179883a36fbb1eecf4f93ee19d41e3d",
		"cc5eea2edf964018bdc0504a3793e4d2145142caa09a72ac5fb8d3e8",
		"a48d78ae5d08aa725342773975a00d4219cf7a8029bb8cf3c17c374a",
		"67b861344b4e416d4094472faf4272f6d54a497177fbc5f9ef292836",
		"1d54f3fcdad795bf3b23408ecbac3e1321d1d66f2e4e3d05f41f7020",
		false,
	},
	{
		"bb658732acbf3147729959eb7318a2058308b2739ec58907dd5b11cfa3ecf69a1752b7b7d806fe00ec402d18f96039f0b78dbb90a59c4414fb33f1f4e02e4089de4122cd93df5263a95be4d7084e2126493892816e6a5b4ed123cb705bf930c8f67af0fb4514d5769232a9b008a803af225160ce63f675bd4872c4c97b146e5e",
		"6234c936e27bf141fc7534bfc0a7eedc657f91308203f1dcbd642855",
		"27983d87ca785ef4892c3591ef4a944b1deb125dd58bd351034a6f84",
		"e94e05b42d01d0b965ffdd6c3a97a36a771e8ea71003de76c4ecb13f",
		"1dc6464ffeefbd7872a081a5926e9fc3e66d123f1784340ba17737e9",
		false,
	},
	{
		"7c00be9123bfa2c4290be1d8bc2942c7f897d9a5b7917e3aabd97ef1aab890f148400a89abd554d19bec9d8ed911ce57b22fbcf6d30ca2115f13ce0a3f569a23bad39ee645f624c49c60dcfc11e7d2be24de9c905596d8f23624d63dc46591d1f740e46f982bfae453f107e80db23545782be23ce43708245896fc54e1ee5c43",
		"9f3f037282aaf14d4772edffff331bbdda845c3f65780498cde334f1",
		"8308ee5a16e3bcb721b6bc30000a0419bc1aaedd761be7f658334066",
		"6381d7804a8808e3c17901e4d283b89449096a8fba993388fa11dc54",
		"8e858f6b5b253686a86b757bad23658cda53115ac565abca4e3d9f57",
		false,
	},
	{
		"cffc122a44840dc705bb37130069921be313d8bde0b66201aebc48add028ca131914ef2e705d6bedd19dc6cf9459bbb0f27cdfe3c50483808ffcdaffbeaa5f062e097180f07a40ef4ab6ed03fe07ed6bcfb8afeb42c97eafa2e8a8df469de07317c5e1494c41547478eff4d8c7d9f0f484ad90fedf6e1c35ee68fa73f1691601",
		"a03b88a10d930002c7b17ca6af2fd3e88fa000edf787dc594f8d4fd4",
		"e0cf7acd6ddc758e64847fe4df9915ebda2f67cdd5ec979aa57421f5",
		"387b84dcf37dc343c7d2c5beb82f0bf8bd894b395a7b894565d296c1",
		"4adc12ce7d20a89ce3925e10491c731b15ddb3f339610857a21b53b4",
		false,
	},
	{
		"26e0e0cafd85b43d16255908ccfd1f061c680df75aba3081246b337495783052ba06c60f4a486c1591a4048bae11b4d7fec4f161d80bdc9a7b79d23e44433ed625eab280521a37f23dd3e1bdc5c6a6cfaa026f3c45cf703e76dab57add93fe844dd4cda67dc3bddd01f9152579e49df60969b10f09ce9372fdd806b0c7301866",
		"9a8983c42f2b5a87c37a00458b5970320d247f0c8a88536440173f7d",
		"15e489ec6355351361900299088cfe8359f04fe0cab78dde952be80c",
		"929a21baa173d438ec9f28d6a585a2f9abcfc0a4300898668e476dc0",
		"59a853f046da8318de77ff43f26fe95a92ee296fa3f7e56ce086c872",
		true,
	},
	{
		"1078eac124f48ae4f807e946971d0de3db3748dd349b14cca5c942560fb25401b2252744f18ad5e455d2d97ed5ae745f55ff509c6c8e64606afe17809affa855c4c4cdcaf6b69ab4846aa5624ed0687541aee6f2224d929685736c6a23906d974d3c257abce1a3fb8db5951b89ecb0cda92b5207d93f6618fd0f893c32cf6a6e",
		"d6e55820bb62c2be97650302d59d667a411956138306bd566e5c3c2b",
		"631ab0d64eaf28a71b9cbd27a7a88682a2167cee6251c44e3810894f",
		"65af72bc7721eb71c2298a0eb4eed3cec96a737cc49125706308b129",
		"bd5a987c78e2d51598dbd9c34a9035b0069c580edefdacee17ad892a",
		false,
	},
	{
		"919deb1fdd831c23481dfdb2475dcbe325b04c34f82561ced3d2df0b3d749b36e255c4928973769d46de8b95f162b53cd666cad9ae145e7fcfba97919f703d864efc11eac5f260a5d920d780c52899e5d76f8fe66936ff82130761231f536e6a3d59792f784902c469aa897aabf9a0678f93446610d56d5e0981e4c8a563556b",
		"269b455b1024eb92d860a420f143ac1286b8cce43031562ae7664574",
		"baeb6ca274a77c44a0247e5eb12ca72bdd9a698b3f3ae69c9f1aaa57",
		"cb4ec2160f04613eb0dfe4608486091a25eb12aa4dec1afe91cfb008",
		"40b01d8cd06589481574f958b98ca08ade9d2a8fe31024375c01bb40",
		false,
	},
	{
		"6e012361250dacf6166d2dd1aa7be544c3206a9d43464b3fcd90f3f8cf48d08ec099b59ba6fe7d9bdcfaf244120aed1695d8be32d1b1cd6f143982ab945d635fb48a7c76831c0460851a3d62b7209c30cd9c2abdbe3d2a5282a9fcde1a6f418dd23c409bc351896b9b34d7d3a1a63bbaf3d677e612d4a80fa14829386a64b33f",
		"6d2d695efc6b43b13c14111f2109608f1020e3e03b5e21cfdbc82fcd",
		"26a4859296b7e360b69cf40be7bd97ceaffa3d07743c8489fc47ca1b",
		"9a8cb5f2fdc288b7183c5b32d8e546fc2ed1ca4285eeae00c8b572ad",
		"8c623f357b5d0057b10cdb1a1593dab57cda7bdec9cf868157a79b97",
		true,
	},
	{
		"bf6bd7356a52b234fe24d25557200971fc803836f6fec3cade9642b13a8e7af10ab48b749de76aada9d8927f9b12f75a2c383ca7358e2566c4bb4f156fce1fd4e87ef8c8d2b6b1bdd351460feb22cdca0437ac10ca5e0abbbce9834483af20e4835386f8b1c96daaa41554ceee56730aac04f23a5c765812efa746051f396566",
		"14250131b2599939cf2d6bc491be80ddfe7ad9de644387ee67de2d40",
		"b5dc473b5d014cd504022043c475d3f93c319a8bdcb7262d9e741803",
		"4f21642f2201278a95339a80f75cc91f8321fcb3c9462562f6cbf145",
		"452a5f816ea1f75dee4fd514fa91a0d6a43622981966c59a1b371ff8",
		false,
	},
	{
		"0eb7f4032f90f0bd3cf9473d6d9525d264d14c031a10acd31a053443ed5fe919d5ac35e0be77813071b4062f0b5fdf58ad5f637b76b0b305aec18f82441b6e607b44cdf6e0e3c7c57f24e6fd565e39430af4a6b1d979821ed0175fa03e3125506847654d7e1ae904ce1190ae38dc5919e257bdac2db142a6e7cd4da6c2e83770",
		"d1f342b7790a1667370a1840255ac5bbbdc66f0bc00ae977d99260ac",
		"76416cabae2de9a1000b4646338b774baabfa3db4673790771220cdb",
		"bc85e3fc143d19a7271b2f9e1c04b86146073f3fab4dda1c3b1f35ca",
		"9a5c70ede3c48d5f43307a0c2a4871934424a3303b815df4bb0f128e",
		false,
	},
	{
		"5cc25348a05d85e56d4b03cec450128727bc537c66ec3a9fb613c151033b5e86878632249cba83adcefc6c1e35dcd31702929c3b57871cda5c18d1cf8f9650a25b917efaed56032e43b6fc398509f0d2997306d8f26675f3a8683b79ce17128e006aa0903b39eeb2f1001be65de0520115e6f919de902b32c38d691a69c58c92",
		"7e49a7abf16a792e4c7bbc4d251820a2abd22d9f2fc252a7bf59c9a6",
		"44236a8fb4791c228c26637c28ae59503a2f450d4cfb0dc42aa843b9",
		"084461b4050285a1a85b2113be76a17878d849e6bc489f4d84f15cd8",
		"079b5bddcc4d45de8dbdfd39f69817c7e5afa454a894d03ee1eaaac3",
		false,
	},
	{
		"1951533ce33afb58935e39e363d8497a8dd0442018fd96dff167b3b23d7206a3ee182a3194765df4768a3284e23b8696c199b4686e670d60c9d782f08794a4bccc05cffffbd1a12acd9eb1cfa01f7ebe124da66ecff4599ea7720c3be4bb7285daa1a86ebf53b042bd23208d468c1b3aa87381f8e1ad63e2b4c2ba5efcf05845",
		"31945d12ebaf4d81f02be2b1768ed80784bf35cf5e2ff53438c11493",
		"a62bebffac987e3b9d3ec451eb64c462cdf7b4aa0b1bbb131ceaa0a4",
		"bc3c32b19e42b710bca5c6aaa128564da3ddb2726b25f33603d2af3c",
		"ed1a719cc0c507edc5239d76fe50e2306c145ad252bd481da04180c0",
		false,
	},
}

func TestVectors(t *testing.T) {
	sha := sha1.New()

	for i, test := range testVectors {
		pub := PublicKey{
			Curve: elliptic.P224(),
			X:     fromHex(test.Qx),
			Y:     fromHex(test.Qy),
		}
		msg, _ := hex.DecodeString(test.msg)
		sha.Reset()
		sha.Write(msg)
		hashed := sha.Sum()
		r := fromHex(test.r)
		s := fromHex(test.s)
		if Verify(&pub, hashed, r, s) != test.ok {
			t.Errorf("%d: bad result", i)
		}
		if testing.Short() {
			break
		}
	}
}
