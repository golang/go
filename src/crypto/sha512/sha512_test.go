// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SHA512 hash algorithm.  See FIPS 180-4.

package sha512

import (
	"encoding/hex"
	"hash"
	"io"
	"testing"
)

type sha512Test struct {
	in     string
	out224 string
	out256 string
	out384 string
	out512 string
}

var golden = []sha512Test{
	{
		"",
		"6ed0dd02806fa89e25de060c19d3ac86cabb87d6a0ddd05c333b84f4",
		"c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a",
		"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
		"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
	},
	{
		"a",
		"d5cdb9ccc769a5121d4175f2bfdd13d6310e0d3d361ea75d82108327",
		"455e518824bc0601f9fb858ff5c37d417d67c2f8e0df2babe4808858aea830f8",
		"54a59b9f22b0b80880d8427e548b7c23abd873486e1f035dce9cd697e85175033caa88e6d57bc35efae0b5afd3145f31",
		"1f40fc92da241694750979ee6cf582f2d5d7d28e18335de05abc54d0560e0f5302860c652bf08d560252aa5e74210546f369fbbbce8c12cfc7957b2652fe9a75",
	},
	{
		"ab",
		"b35878d07bfedf39fc638af08547eb5d1072d8546319f247b442fbf5",
		"22d4d37ec6370571af7109fb12eae79673d5f7c83e6e677083faa3cfac3b2c14",
		"c7be03ba5bcaa384727076db0018e99248e1a6e8bd1b9ef58a9ec9dd4eeebb3f48b836201221175befa74ddc3d35afdd",
		"2d408a0717ec188158278a796c689044361dc6fdde28d6f04973b80896e1823975cdbf12eb63f9e0591328ee235d80e9b5bf1aa6a44f4617ff3caf6400eb172d",
	},
	{
		"abc",
		"4634270f707b6a54daae7530460842e20e37ed265ceee9a43e8924aa",
		"53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23",
		"cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
		"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
	},
	{
		"abcd",
		"0c9f157ab030fb06e957c14e3938dc5908962e5dd7b66f04a36fc534",
		"d2891c7978be0e24948f37caa415b87cb5cbe2b26b7bad9dc6391b8a6f6ddcc9",
		"1165b3406ff0b52a3d24721f785462ca2276c9f454a116c2b2ba20171a7905ea5a026682eb659c4d5f115c363aa3c79b",
		"d8022f2060ad6efd297ab73dcc5355c9b214054b0d1776a136a669d26a7d3b14f73aa0d0ebff19ee333368f0164b6419a96da49e3e481753e7e96b716bdccb6f",
	},
	{
		"abcde",
		"880e79bb0a1d2c9b7528d851edb6b8342c58c831de98123b432a4515",
		"de8322b46e78b67d4431997070703e9764e03a1237b896fd8b379ed4576e8363",
		"4c525cbeac729eaf4b4665815bc5db0c84fe6300068a727cf74e2813521565abc0ec57a37ee4d8be89d097c0d2ad52f0",
		"878ae65a92e86cac011a570d4c30a7eaec442b85ce8eca0c2952b5e3cc0628c2e79d889ad4d5c7c626986d452dd86374b6ffaa7cd8b67665bef2289a5c70b0a1",
	},
	{
		"abcdef",
		"236c829cfea4fd6d4de61ad15fcf34dca62342adaf9f2001c16f29b8",
		"e4fdcb11d1ac14e698743acd8805174cea5ddc0d312e3e47f6372032571bad84",
		"c6a4c65b227e7387b9c3e839d44869c4cfca3ef583dea64117859b808c1e3d8ae689e1e314eeef52a6ffe22681aa11f5",
		"e32ef19623e8ed9d267f657a81944b3d07adbb768518068e88435745564e8d4150a0a703be2a7d88b61e3d390c2bb97e2d4c311fdc69d6b1267f05f59aa920e7",
	},
	{
		"abcdefg",
		"4767af672b3ed107f25018dc22d6fa4b07d156e13b720971e2c4f6bf",
		"a8117f680bdceb5d1443617cbdae9255f6900075422326a972fdd2f65ba9bee3",
		"9f11fc131123f844c1226f429b6a0a6af0525d9f40f056c7fc16cdf1b06bda08e302554417a59fa7dcf6247421959d22",
		"d716a4188569b68ab1b6dfac178e570114cdf0ea3a1cc0e31486c3e41241bc6a76424e8c37ab26f096fc85ef9886c8cb634187f4fddff645fb099f1ff54c6b8c",
	},
	{
		"abcdefgh",
		"792e25e0ae286d123a38950007e037d3122e76c4ee201668c385edab",
		"a29b9645d2a02a8b582888d044199787220e316bf2e89d1422d3df26bf545bbe",
		"9000cd7cada59d1d2eb82912f7f24e5e69cc5517f68283b005fa27c285b61e05edf1ad1a8a9bded6fd29eb87d75ad806",
		"a3a8c81bc97c2560010d7389bc88aac974a104e0e2381220c6e084c4dccd1d2d17d4f86db31c2a851dc80e6681d74733c55dcd03dd96f6062cdda12a291ae6ce",
	},
	{
		"abcdefghi",
		"56b275d36127dc070cda4019baf2ce2579a25d8c67fa2bc9be61b539",
		"b955095330f9c8188d11884ec1679dc44c9c5b25ff9bda700416df9cdd39188f",
		"ef54915b60cf062b8dd0c29ae3cad69abe6310de63ac081f46ef019c5c90897caefd79b796cfa81139788a260ded52df",
		"f22d51d25292ca1d0f68f69aedc7897019308cc9db46efb75a03dd494fc7f126c010e8ade6a00a0c1a5f1b75d81e0ed5a93ce98dc9b833db7839247b1d9c24fe",
	},
	{
		"abcdefghij",
		"f809423cbb25e81a2a64aecee2cd5fdc7d91d5db583901fbf1db3116",
		"550762913d51eefbcd1a55068fcfc9b154fd11c1078b996df0d926ea59d2a68d",
		"a12070030a02d86b0ddacd0d3a5b598344513d0a051e7355053e556a0055489c1555399b03342845c4adde2dc44ff66c",
		"ef6b97321f34b1fea2169a7db9e1960b471aa13302a988087357c520be957ca119c3ba68e6b4982c019ec89de3865ccf6a3cda1fe11e59f98d99f1502c8b9745",
	},
	{
		"Discard medicine more than two years old.",
		"4c46e10b5b72204e509c3c06072cea970bc020cd45a61a0acdfa97ac",
		"690c8ad3916cefd3ad29226d9875965e3ee9ec0d4482eacc248f2ff4aa0d8e5b",
		"86f58ec2d74d1b7f8eb0c2ff0967316699639e8d4eb129de54bdf34c96cdbabe200d052149f2dd787f43571ba74670d4",
		"2210d99af9c8bdecda1b4beff822136753d8342505ddce37f1314e2cdbb488c6016bdaa9bd2ffa513dd5de2e4b50f031393d8ab61f773b0e0130d7381e0f8a1d",
	},
	{
		"He who has a shady past knows that nice guys finish last.",
		"cb0cef13c1848d91a6d02637c7c520de1914ad4a7aea824671cc328e",
		"25938ca49f7ef1178ce81620842b65e576245fcaed86026a36b516b80bb86b3b",
		"ae4a2b639ca9bfa04b1855d5a05fe7f230994f790891c6979103e2605f660c4c1262a48142dcbeb57a1914ba5f7c3fa7",
		"a687a8985b4d8d0a24f115fe272255c6afaf3909225838546159c1ed685c211a203796ae8ecc4c81a5b6315919b3a64f10713da07e341fcdbb08541bf03066ce",
	},
	{
		"I wouldn't marry him with a ten foot pole.",
		"6c7bd0f3a6544ea698006c2ea583a85f80ea2913590a186db8bb2f1b",
		"698e420c3a7038e53d8e73f4be2b02e03b93464ac1a61ebe69f557079921ef65",
		"40ae213df6436eca952aa6841886fcdb82908ef1576a99c8f49bb9dd5023169f7c53035abdda0b54c302f4974e2105e7",
		"8ddb0392e818b7d585ab22769a50df660d9f6d559cca3afc5691b8ca91b8451374e42bcdabd64589ed7c91d85f626596228a5c8572677eb98bc6b624befb7af8",
	},
	{
		"Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave",
		"981323be3eca6ccfa598e58dd74ed8cb05d5f7f6653b7604b684f904",
		"839b414d7e3900ee243aa3d1f9b6955720e64041f5ab9bedd3eb0a08da5a2ca8",
		"e7cf8b873c9bc950f06259aa54309f349cefa72c00d597aebf903e6519a50011dfe355afff064a10701c705693848df9",
		"26ed8f6ca7f8d44b6a8a54ae39640fa8ad5c673f70ee9ce074ba4ef0d483eea00bab2f61d8695d6b34df9c6c48ae36246362200ed820448bdc03a720366a87c6",
	},
	{
		"The days of the digital watch are numbered.  -Tom Stoppard",
		"e6fbf82df5138bf361e826903cadf0612cb2986649ba47a57e1bca99",
		"5625ecb9d284e54c00b257b67a8cacb25a78db2845c60ef2d29e43c84f236e8e",
		"c3d4f0f4047181c7d39d34703365f7bf70207183caf2c2f6145f04da895ef69124d9cdeb635da636c3a474e61024e29b",
		"e5a14bf044be69615aade89afcf1ab0389d5fc302a884d403579d1386a2400c089b0dbb387ed0f463f9ee342f8244d5a38cfbc0e819da9529fbff78368c9a982",
	},
	{
		"Nepal premier won't resign.",
		"6ec2cb2ecafc1a9bddaf4caf57344d853e6ded398927d5694fd7714f",
		"9b81d06bca2f985e6ad3249096ff3c0f2a9ec5bb16ef530d738d19d81e7806f2",
		"a097aab567e167d5cf93676ed73252a69f9687cb3179bb2d27c9878119e94bf7b7c4b58dc90582edfaf66e11388ed714",
		"420a1faa48919e14651bed45725abe0f7a58e0f099424c4e5a49194946e38b46c1f8034b18ef169b2e31050d1648e0b982386595f7df47da4b6fd18e55333015",
	},
	{
		"For every action there is an equal and opposite government program.",
		"7f62f36e716e0badaf4a4658da9d09bea26357a1bc6aeb8cf7c3ae35",
		"08241df8d91edfcd68bb1a1dada6e0ae1475a5c6e7b8f12d8e24ca43a38240a9",
		"5026ca45c41fc64712eb65065da92f6467541c78f8966d3fe2c8e3fb769a3ec14215f819654b47bd64f7f0eac17184f3",
		"d926a863beadb20134db07683535c72007b0e695045876254f341ddcccde132a908c5af57baa6a6a9c63e6649bba0c213dc05fadcf9abccea09f23dcfb637fbe",
	},
	{
		"His money is twice tainted: 'taint yours and 'taint mine.",
		"45adffcb86a05ee4d91263a6115dda011b805d442c60836963cb8378",
		"4ff74d9213a8117745f5d37b5353a774ec81c5dfe65c4c8986a56fc01f2c551e",
		"ac1cc0f5ac8d5f5514a7b738ac322b7fb52a161b449c3672e9b6a6ad1a5e4b26b001cf3bad24c56598676ca17d4b445a",
		"9a98dd9bb67d0da7bf83da5313dff4fd60a4bac0094f1b05633690ffa7f6d61de9a1d4f8617937d560833a9aaa9ccafe3fd24db418d0e728833545cadd3ad92d",
	},
	{
		"There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977",
		"51cb518f1f68daa901a3075a0a5e1acc755b4e5c82cb47687537f880",
		"b5baf747c307f98849ec881cf0d48605ae4edd386372aea9b26e71db517e650b",
		"722d10c5de371ec0c8c4b5247ac8a5f1d240d68c73f8da13d8b25f0166d6f309bf9561979a111a0049405771d201941a",
		"d7fde2d2351efade52f4211d3746a0780a26eec3df9b2ed575368a8a1c09ec452402293a8ea4eceb5a4f60064ea29b13cdd86918cd7a4faf366160b009804107",
	},
	{
		"It's a tiny change to the code and not completely disgusting. - Bob Manchek",
		"3b59c5e64b0da7bfc18d7017bf458d90f2c83601ff1afc6263ac0993",
		"7eef0538ebd7ecf18611d23b0e1cd26a74d65b929a2e374197dc66e755ca4944",
		"dc2d3ea18bfa10549c63bf2b75b39b5167a80c12aff0e05443168ea87ff149fb0eda5e0bd234eb5d48c7d02ffc5807f1",
		"b0f35ffa2697359c33a56f5c0cf715c7aeed96da9905ca2698acadb08fbc9e669bf566b6bd5d61a3e86dc22999bcc9f2224e33d1d4f32a228cf9d0349e2db518",
	},
	{
		"size:  a.out:  bad magic",
		"6a9525c0fac0f91b489bc4f0f539b9ec4a156a4e98bc15b655c2c881",
		"d05600964f83f55323104aadab434f32391c029718a7690d08ddb2d7e8708443",
		"1d67c969e2a945ae5346d2139760261504d4ba164c522443afe19ef3e29b152a4c52445489cfc9d7215e5a450e8e1e4e",
		"3d2e5f91778c9e66f7e061293aaa8a8fc742dd3b2e4f483772464b1144189b49273e610e5cccd7a81a19ca1fa70f16b10f1a100a4d8c1372336be8484c64b311",
	},
	{
		"The major problem is with sendmail.  -Mark Horton",
		"a1b2b2905b1527d682049c6a76e35c7d8c72551abfe7833ac1be595f",
		"53ed5f9b5c0b674ac0f3425d9f9a5d462655b07cc90f5d0f692eec093884a607",
		"5ff8e075e465646e7b73ef36d812c6e9f7d60fa6ea0e533e5569b4f73cde53cdd2cc787f33540af57cca3fe467d32fe0",
		"b2f68ff58ac015efb1c94c908b0d8c2bf06f491e4de8e6302c49016f7f8a33eac3e959856c7fddbc464de618701338a4b46f76dbfaf9a1e5262b5f40639771c7",
	},
	{
		"Give me a rock, paper and scissors and I will move the world.  CCFestoon",
		"76cf045c76a5f2e3d64d56c3cdba6a25479334611bc375460526f8c1",
		"5a0147685a44eea2435dbd582724efca7637acd9c428e5e1a05115bc3bc2a0e0",
		"5bd0a997a67c9ae1979a894eb0cde403dde003c9b6f2c03cf21925c42ff4e1176e6df1ca005381612ef18457b9b7ec3b",
		"d8c92db5fdf52cf8215e4df3b4909d29203ff4d00e9ad0b64a6a4e04dec5e74f62e7c35c7fb881bd5de95442123df8f57a489b0ae616bd326f84d10021121c57",
	},
	{
		"If the enemy is within range, then so are you.",
		"4473671daeecfdb6f6c5bc06b26374aa5e497cc37119fe14144c430c",
		"1152c9b27a99dbf4057d21438f4e63dd0cd0977d5ff12317c64d3b97fcac875a",
		"1eee6da33e7e54fc5be52ae23b94b16ba4d2a947ae4505c6a3edfc7401151ea5205ac01b669b56f27d8ef7f175ed7762",
		"19a9f8dc0a233e464e8566ad3ca9b91e459a7b8c4780985b015776e1bf239a19bc233d0556343e2b0a9bc220900b4ebf4f8bdf89ff8efeaf79602d6849e6f72e",
	},
	{
		"It's well we cannot hear the screams/That we create in others' dreams.",
		"6accb6394758523fcd453d47d37ebd10868957a0a9e81c796736abf8",
		"105e890f5d5cf1748d9a7b4cdaf58b69855779deebc2097747c2210a17b2cb51",
		"76b06e9dea66bfbb1a96029426dc0dfd7830bd297eb447ff5358d94a87cd00c88b59df2493fef56ecbb5231073892ea9",
		"00b4c41f307bde87301cdc5b5ab1ae9a592e8ecbb2021dd7bc4b34e2ace60741cc362560bec566ba35178595a91932b8d5357e2c9cec92d393b0fa7831852476",
	},
	{
		"You remind me of a TV show, but that's all right: I watch it anyway.",
		"6f173f4b6eac7f2a73eaa0833c4563752df2c869dc00b7d30219e12e",
		"74644ead770da1434365cd912656fe1aca2056d3039d39f10eb1151bddb32cf3",
		"12acaf21452cff586143e3f5db0bfdf7802c057e1adf2a619031c4e1b0ccc4208cf6cef8fe722bbaa2fb46a30d9135d8",
		"91eccc3d5375fd026e4d6787874b1dce201cecd8a27dbded5065728cb2d09c58a3d467bb1faf353bf7ba567e005245d5321b55bc344f7c07b91cb6f26c959be7",
	},
	{
		"C is as portable as Stonehedge!!",
		"db05bf4d0f73325208755f4af96cfac6cb3db5dbfc323d675d68f938",
		"50a234625de5587581883dad9ef399460928032a5ea6bd005d7dc7b68d8cc3d6",
		"0fc23d7f4183efd186f0bc4fc5db867e026e2146b06cb3d52f4bdbd57d1740122caa853b41868b197b2ac759db39df88",
		"fabbbe22180f1f137cfdc9556d2570e775d1ae02a597ded43a72a40f9b485d500043b7be128fb9fcd982b83159a0d99aa855a9e7cc4240c00dc01a9bdf8218d7",
	},
	{
		"Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley",
		"05ffa71bb02e855de1aaee1777b3bdbaf7507646f19c4c6aa29933d0",
		"a7a3846005f8a9935a0a2d43e7fd56d95132a9a3609bf3296ef80b8218acffa0",
		"bc805578a7f85d34a86a32976e1c34fe65cf815186fbef76f46ef99cda10723f971f3f1464d488243f5e29db7488598d",
		"2ecdec235c1fa4fc2a154d8fba1dddb8a72a1ad73838b51d792331d143f8b96a9f6fcb0f34d7caa351fe6d88771c4f105040e0392f06e0621689d33b2f3ba92e",
	},
	{
		"The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule",
		"3ad3c89e15b91e6273534c5d18adadbb528e7b840b288f64e81b8c6d",
		"688ff03e367680757aa9906cb1e2ad218c51f4526dc0426ea229a5ba9d002c69",
		"b23918399a12ebf4431559eec3813eaf7412e875fd7464f16d581e473330842d2e96c6be49a7ce3f9bb0b8bc0fcbe0fe",
		"7ad681f6f96f82f7abfa7ecc0334e8fa16d3dc1cdc45b60b7af43fe4075d2357c0c1d60e98350f1afb1f2fe7a4d7cd2ad55b88e458e06b73c40b437331f5dab4",
	},
	{
		"How can you write a big system without C++?  -Paul Glick",
		"e3763669d1b760c1be7bfcb6625f92300a8430419d1dbad57ec9f53c",
		"3fa46d52094b01021cff5af9a438982b887a5793f624c0a6644149b6b7c3f485",
		"1764b700eb1ead52a2fc33cc28975c2180f1b8faa5038d94cffa8d78154aab16e91dd787e7b0303948ebed62561542c8",
		"833f9248ab4a3b9e5131f745fda1ffd2dd435b30e965957e78291c7ab73605fd1912b0794e5c233ab0a12d205a39778d19b83515d6a47003f19cdee51d98c7e0",
	},
}

func testHash(t *testing.T, name, in, outHex string, oneShotResult []byte, digestFunc hash.Hash) {
	if calculated := hex.EncodeToString(oneShotResult); calculated != outHex {
		t.Errorf("one-shot result for %s(%q) = %q, but expected %q", name, in, calculated, outHex)
		return
	}

	for pass := 0; pass < 3; pass++ {
		if pass < 2 {
			io.WriteString(digestFunc, in)
		} else {
			io.WriteString(digestFunc, in[:len(in)/2])
			digestFunc.Sum(nil)
			io.WriteString(digestFunc, in[len(in)/2:])
		}

		if calculated := hex.EncodeToString(digestFunc.Sum(nil)); calculated != outHex {
			t.Errorf("%s(%q) = %q (in pass #%d), but expected %q", name, in, calculated, pass, outHex)
		}
		digestFunc.Reset()
	}
}

func TestGolden(t *testing.T) {
	for _, test := range golden {
		in := []byte(test.in)

		sum224 := Sum512_224(in)
		sum256 := Sum512_256(in)
		sum384 := Sum384(in)
		sum512 := Sum512(in)
		testHash(t, "SHA512/224", test.in, test.out224, sum224[:], New512_224())
		testHash(t, "SHA512/256", test.in, test.out256, sum256[:], New512_256())
		testHash(t, "SHA384", test.in, test.out384, sum384[:], New384())
		testHash(t, "SHA512", test.in, test.out512, sum512[:], New())
	}
}

func TestSize(t *testing.T) {
	c := New()
	if got := c.Size(); got != Size {
		t.Errorf("Size = %d; want %d", got, Size)
	}
	c = New384()
	if got := c.Size(); got != Size384 {
		t.Errorf("New384.Size = %d; want %d", got, Size384)
	}
	c = New512_224()
	if got := c.Size(); got != Size224 {
		t.Errorf("New512224.Size = %d; want %d", got, Size224)
	}
	c = New512_256()
	if got := c.Size(); got != Size256 {
		t.Errorf("New512256.Size = %d; want %d", got, Size256)
	}
}

func TestBlockSize(t *testing.T) {
	c := New()
	if got := c.BlockSize(); got != BlockSize {
		t.Errorf("BlockSize = %d; want %d", got, BlockSize)
	}
}

var bench = New()
var buf = make([]byte, 8192)

func benchmarkSize(b *testing.B, size int) {
	b.SetBytes(int64(size))
	sum := make([]byte, bench.Size())
	for i := 0; i < b.N; i++ {
		bench.Reset()
		bench.Write(buf[:size])
		bench.Sum(sum[:0])
	}
}

func BenchmarkHash8Bytes(b *testing.B) {
	benchmarkSize(b, 8)
}

func BenchmarkHash1K(b *testing.B) {
	benchmarkSize(b, 1024)
}

func BenchmarkHash8K(b *testing.B) {
	benchmarkSize(b, 8192)
}
