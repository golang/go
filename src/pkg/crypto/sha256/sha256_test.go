// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SHA256 hash algorithm.  See FIPS 180-2.

package sha256

import (
	"fmt"
	"io"
	"testing"
)

type sha256Test struct {
	out string
	in  string
}

var golden = []sha256Test{
	{"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", ""},
	{"ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb", "a"},
	{"fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603", "ab"},
	{"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc"},
	{"88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589", "abcd"},
	{"36bbe50ed96841d10443bcb670d6554f0a34b761be67ec9c4a8ad2c0c44ca42c", "abcde"},
	{"bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721", "abcdef"},
	{"7d1a54127b222502f5b79b5fb0803061152a44f92b37e23c6527baf665d4da9a", "abcdefg"},
	{"9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab", "abcdefgh"},
	{"19cc02f26df43cc571bc9ed7b0c4d29224a3ec229529221725ef76d021c8326f", "abcdefghi"},
	{"72399361da6a7754fec986dca5b7cbaf1c810a28ded4abaf56b2106d06cb78b0", "abcdefghij"},
	{"a144061c271f152da4d151034508fed1c138b8c976339de229c3bb6d4bbb4fce", "Discard medicine more than two years old."},
	{"6dae5caa713a10ad04b46028bf6dad68837c581616a1589a265a11288d4bb5c4", "He who has a shady past knows that nice guys finish last."},
	{"ae7a702a9509039ddbf29f0765e70d0001177914b86459284dab8b348c2dce3f", "I wouldn't marry him with a ten foot pole."},
	{"6748450b01c568586715291dfa3ee018da07d36bb7ea6f180c1af6270215c64f", "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{"14b82014ad2b11f661b5ae6a99b75105c2ffac278cd071cd6c05832793635774", "The days of the digital watch are numbered.  -Tom Stoppard"},
	{"7102cfd76e2e324889eece5d6c41921b1e142a4ac5a2692be78803097f6a48d8", "Nepal premier won't resign."},
	{"23b1018cd81db1d67983c5f7417c44da9deb582459e378d7a068552ea649dc9f", "For every action there is an equal and opposite government program."},
	{"8001f190dfb527261c4cfcab70c98e8097a7a1922129bc4096950e57c7999a5a", "His money is twice tainted: 'taint yours and 'taint mine."},
	{"8c87deb65505c3993eb24b7a150c4155e82eee6960cf0c3a8114ff736d69cad5", "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{"bfb0a67a19cdec3646498b2e0f751bddc41bba4b7f30081b0b932aad214d16d7", "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{"7f9a0b9bf56332e19f5a0ec1ad9c1425a153da1c624868fda44561d6b74daf36", "size:  a.out:  bad magic"},
	{"b13f81b8aad9e3666879af19886140904f7f429ef083286195982a7588858cfc", "The major problem is with sendmail.  -Mark Horton"},
	{"b26c38d61519e894480c70c8374ea35aa0ad05b2ae3d6674eec5f52a69305ed4", "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{"049d5e26d4f10222cd841a119e38bd8d2e0d1129728688449575d4ff42b842c1", "If the enemy is within range, then so are you."},
	{"0e116838e3cc1c1a14cd045397e29b4d087aa11b0853fc69ec82e90330d60949", "It's well we cannot hear the screams/That we create in others' dreams."},
	{"4f7d8eb5bcf11de2a56b971021a444aa4eafd6ecd0f307b5109e4e776cd0fe46", "You remind me of a TV show, but that's all right: I watch it anyway."},
	{"61c0cc4c4bd8406d5120b3fb4ebc31ce87667c162f29468b3c779675a85aebce", "C is as portable as Stonehedge!!"},
	{"1fb2eb3688093c4a3f80cd87a5547e2ce940a4f923243a79a2a1e242220693ac", "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{"395585ce30617b62c80b93e8208ce866d4edc811a177fdb4b82d3911d8696423", "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{"4f9b189a13d030838269dce846b16a1ce9ce81fe63e65de2f636863336a98fe6", "How can you write a big system without C++?  -Paul Glick"},
}

var golden224 = []sha256Test{
	{"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f", ""},
	{"abd37534c7d9a2efb9465de931cd7055ffdb8879563ae98078d6d6d5", "a"},
	{"db3cda86d4429a1d39c148989566b38f7bda0156296bd364ba2f878b", "ab"},
	{"23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7", "abc"},
	{"a76654d8e3550e9a2d67a0eeb6c67b220e5885eddd3fde135806e601", "abcd"},
	{"bdd03d560993e675516ba5a50638b6531ac2ac3d5847c61916cfced6", "abcde"},
	{"7043631cb415556a275a4ebecb802c74ee9f6153908e1792a90b6a98", "abcdef"},
	{"d1884e711701ad81abe0c77a3b0ea12e19ba9af64077286c72fc602d", "abcdefg"},
	{"17eb7d40f0356f8598e89eafad5f6c759b1f822975d9c9b737c8a517", "abcdefgh"},
	{"aeb35915346c584db820d2de7af3929ffafef9222a9bcb26516c7334", "abcdefghi"},
	{"d35e1e5af29ddb0d7e154357df4ad9842afee527c689ee547f753188", "abcdefghij"},
	{"19297f1cef7ddc8a7e947f5c5a341e10f7245045e425db67043988d7", "Discard medicine more than two years old."},
	{"0f10c2eb436251f777fbbd125e260d36aecf180411726c7c885f599a", "He who has a shady past knows that nice guys finish last."},
	{"4d1842104919f314cad8a3cd20b3cba7e8ed3e7abed62b57441358f6", "I wouldn't marry him with a ten foot pole."},
	{"a8ba85c6fe0c48fbffc72bbb2f03fcdbc87ae2dc7a56804d1590fb3b", "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{"5543fbab26e67e8885b1a852d567d1cb8b9bfe42e0899584c50449a9", "The days of the digital watch are numbered.  -Tom Stoppard"},
	{"65ca107390f5da9efa05d28e57b221657edc7e43a9a18fb15b053ddb", "Nepal premier won't resign."},
	{"84953962be366305a9cc9b5cd16ed019edc37ac96c0deb3e12cca116", "For every action there is an equal and opposite government program."},
	{"35a189ce987151dfd00b3577583cc6a74b9869eecf894459cb52038d", "His money is twice tainted: 'taint yours and 'taint mine."},
	{"2fc333713983edfd4ef2c0da6fb6d6415afb94987c91e4069eb063e6", "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{"cbe32d38d577a1b355960a4bc3c659c2dc4670859a19777a875842c4", "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{"a2dc118ce959e027576413a7b440c875cdc8d40df9141d6ef78a57e1", "size:  a.out:  bad magic"},
	{"d10787e24052bcff26dc484787a54ed819e4e4511c54890ee977bf81", "The major problem is with sendmail.  -Mark Horton"},
	{"62efcf16ab8a893acdf2f348aaf06b63039ff1bf55508c830532c9fb", "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{"3e9b7e4613c59f58665104c5fa86c272db5d3a2ff30df5bb194a5c99", "If the enemy is within range, then so are you."},
	{"5999c208b8bdf6d471bb7c359ac5b829e73a8211dff686143a4e7f18", "It's well we cannot hear the screams/That we create in others' dreams."},
	{"3b2d67ff54eabc4ef737b14edf87c64280ef582bcdf2a6d56908b405", "You remind me of a TV show, but that's all right: I watch it anyway."},
	{"d0733595d20e4d3d6b5c565a445814d1bbb2fd08b9a3b8ffb97930c6", "C is as portable as Stonehedge!!"},
	{"43fb8aeed8a833175c9295c1165415f98c866ef08a4922959d673507", "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{"ec18e66e93afc4fb1604bc2baedbfd20b44c43d76e65c0996d7851c6", "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{"86ed2eaa9c75ba98396e5c9fb2f679ecf0ea2ed1e0ee9ceecb4a9332", "How can you write a big system without C++?  -Paul Glick"},
}

func TestGolden(t *testing.T) {
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c := New()
		for j := 0; j < 3; j++ {
			if j < 2 {
				io.WriteString(c, g.in)
			} else {
				io.WriteString(c, g.in[0:len(g.in)/2])
				c.Sum(nil)
				io.WriteString(c, g.in[len(g.in)/2:])
			}
			s := fmt.Sprintf("%x", c.Sum(nil))
			if s != g.out {
				t.Fatalf("sha256[%d](%s) = %s want %s", j, g.in, s, g.out)
			}
			c.Reset()
		}
	}
	for i := 0; i < len(golden224); i++ {
		g := golden224[i]
		c := New224()
		for j := 0; j < 3; j++ {
			if j < 2 {
				io.WriteString(c, g.in)
			} else {
				io.WriteString(c, g.in[0:len(g.in)/2])
				c.Sum(nil)
				io.WriteString(c, g.in[len(g.in)/2:])
			}
			s := fmt.Sprintf("%x", c.Sum(nil))
			if s != g.out {
				t.Fatalf("sha224[%d](%s) = %s want %s", j, g.in, s, g.out)
			}
			c.Reset()
		}
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
