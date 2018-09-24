// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"reflect"
	"testing"
)

var aesGCMTests = []struct {
	key, nonce, plaintext, ad, result string
}{
	{
		"11754cd72aec309bf52f7687212e8957",
		"3c819d9a9bed087615030b65",
		"",
		"",
		"250327c674aaf477aef2675748cf6971",
	},
	{
		"ca47248ac0b6f8372a97ac43508308ed",
		"ffd2b598feabc9019262d2be",
		"",
		"",
		"60d20404af527d248d893ae495707d1a",
	},
	{
		"fbe3467cc254f81be8e78d765a2e6333",
		"c6697351ff4aec29cdbaabf2",
		"",
		"67",
		"3659cdc25288bf499ac736c03bfc1159",
	},
	{
		"8a7f9d80d08ad0bd5a20fb689c88f9fc",
		"88b7b27d800937fda4f47301",
		"",
		"50edd0503e0d7b8c91608eb5a1",
		"ed6f65322a4740011f91d2aae22dd44e",
	},
	{
		"051758e95ed4abb2cdc69bb454110e82",
		"c99a66320db73158a35a255d",
		"",
		"67c6697351ff4aec29cdbaabf2fbe3467cc254f81be8e78d765a2e63339f",
		"6ce77f1a5616c505b6aec09420234036",
	},
	{
		"77be63708971c4e240d1cb79e8d77feb",
		"e0e00f19fed7ba0136a797f3",
		"",
		"7a43ec1d9c0a5a78a0b16533a6213cab",
		"209fcc8d3675ed938e9c7166709dd946",
	},
	{
		"7680c5d3ca6154758e510f4d25b98820",
		"f8f105f9c3df4965780321f8",
		"",
		"c94c410194c765e3dcc7964379758ed3",
		"94dca8edfcf90bb74b153c8d48a17930",
	},
	{
		"7fddb57453c241d03efbed3ac44e371c",
		"ee283a3fc75575e33efd4887",
		"d5de42b461646c255c87bd2962d3b9a2",
		"",
		"2ccda4a5415cb91e135c2a0f78c9b2fdb36d1df9b9d5e596f83e8b7f52971cb3",
	},
	{
		"ab72c77b97cb5fe9a382d9fe81ffdbed",
		"54cc7dc2c37ec006bcc6d1da",
		"007c5e5b3e59df24a7c355584fc1518d",
		"",
		"0e1bde206a07a9c2c1b65300f8c649972b4401346697138c7a4891ee59867d0c",
	},
	{
		"fe47fcce5fc32665d2ae399e4eec72ba",
		"5adb9609dbaeb58cbd6e7275",
		"7c0e88c88899a779228465074797cd4c2e1498d259b54390b85e3eef1c02df60e743f1b840382c4bccaf3bafb4ca8429bea063",
		"88319d6e1d3ffa5f987199166c8a9b56c2aeba5a",
		"98f4826f05a265e6dd2be82db241c0fbbbf9ffb1c173aa83964b7cf5393043736365253ddbc5db8778371495da76d269e5db3e291ef1982e4defedaa2249f898556b47",
	},
	{
		"ec0c2ba17aa95cd6afffe949da9cc3a8",
		"296bce5b50b7d66096d627ef",
		"b85b3753535b825cbe5f632c0b843c741351f18aa484281aebec2f45bb9eea2d79d987b764b9611f6c0f8641843d5d58f3a242",
		"f8d00f05d22bf68599bcdeb131292ad6e2df5d14",
		"a7443d31c26bdf2a1c945e29ee4bd344a99cfaf3aa71f8b3f191f83c2adfc7a07162995506fde6309ffc19e716eddf1a828c5a890147971946b627c40016da1ecf3e77",
	},
	{
		"2c1f21cf0f6fb3661943155c3e3d8492",
		"23cb5ff362e22426984d1907",
		"42f758836986954db44bf37c6ef5e4ac0adaf38f27252a1b82d02ea949c8a1a2dbc0d68b5615ba7c1220ff6510e259f06655d8",
		"5d3624879d35e46849953e45a32a624d6a6c536ed9857c613b572b0333e701557a713e3f010ecdf9a6bd6c9e3e44b065208645aff4aabee611b391528514170084ccf587177f4488f33cfb5e979e42b6e1cfc0a60238982a7aec",
		"81824f0e0d523db30d3da369fdc0d60894c7a0a20646dd015073ad2732bd989b14a222b6ad57af43e1895df9dca2a5344a62cc57a3ee28136e94c74838997ae9823f3a",
	},
	{
		"d9f7d2411091f947b4d6f1e2d1f0fb2e",
		"e1934f5db57cc983e6b180e7",
		"73ed042327f70fe9c572a61545eda8b2a0c6e1d6c291ef19248e973aee6c312012f490c2c6f6166f4a59431e182663fcaea05a",
		"0a8a18a7150e940c3d87b38e73baee9a5c049ee21795663e264b694a949822b639092d0e67015e86363583fcf0ca645af9f43375f05fdb4ce84f411dcbca73c2220dea03a20115d2e51398344b16bee1ed7c499b353d6c597af8",
		"aaadbd5c92e9151ce3db7210b8714126b73e43436d242677afa50384f2149b831f1d573c7891c2a91fbc48db29967ec9542b2321b51ca862cb637cdd03b99a0f93b134",
	},
	{
		"fe9bb47deb3a61e423c2231841cfd1fb",
		"4d328eb776f500a2f7fb47aa",
		"f1cc3818e421876bb6b8bbd6c9",
		"",
		"b88c5c1977b35b517b0aeae96743fd4727fe5cdb4b5b42818dea7ef8c9",
	},
	{
		"6703df3701a7f54911ca72e24dca046a",
		"12823ab601c350ea4bc2488c",
		"793cd125b0b84a043e3ac67717",
		"",
		"b2051c80014f42f08735a7b0cd38e6bcd29962e5f2c13626b85a877101",
	},
	// These cases test non-standard nonce sizes.
	{
		"1672c3537afa82004c6b8a46f6f0d026",
		"05",
		"",
		"",
		"8e2ad721f9455f74d8b53d3141f27e8e",
	},
	{
		"9a4fea86a621a91ab371e492457796c0",
		"75",
		"ca6131faf0ff210e4e693d6c31c109fc5b6f54224eb120f37de31dc59ec669b6",
		"4f6e2585c161f05a9ae1f2f894e9f0ab52b45d0f",
		"5698c0a384241d30004290aac56bb3ece6fe8eacc5c4be98954deb9c3ff6aebf5d50e1af100509e1fba2a5e8a0af9670",
	},
	{
		"d0f1f4defa1e8c08b4b26d576392027c",
		"42b4f01eb9f5a1ea5b1eb73b0fb0baed54f387ecaa0393c7d7dffc6af50146ecc021abf7eb9038d4303d91f8d741a11743166c0860208bcc02c6258fd9511a2fa626f96d60b72fcff773af4e88e7a923506e4916ecbd814651e9f445adef4ad6a6b6c7290cc13b956130eef5b837c939fcac0cbbcc9656cd75b13823ee5acdac",
		"",
		"",
		"7ab49b57ddf5f62c427950111c5c4f0d",
	},
	{
		"4a0c00a3d284dea9d4bf8b8dde86685e",
		"f8cbe82588e784bcacbe092cd9089b51e01527297f635bf294b3aa787d91057ef23869789698ac960707857f163ecb242135a228ad93964f5dc4a4d7f88fd7b3b07dd0a5b37f9768fb05a523639f108c34c661498a56879e501a2321c8a4a94d7e1b89db255ac1f685e185263368e99735ebe62a7f2931b47282be8eb165e4d7",
		"6d4bf87640a6a48a50d28797b7",
		"8d8c7ffc55086d539b5a8f0d1232654c",
		"0d803ec309482f35b8e6226f2b56303239298e06b281c2d51aaba3c125",
	},
	{
		"0e18a844ac5bf38e4cd72d9b0942e506",
		"0870d4b28a2954489a0abcd5",
		"67c6697351ff4aec29cdbaabf2fbe3467cc254f81be8e78d765a2e63339fc99a66320db73158a35a255d051758e95ed4abb2cdc69bb454110e827441213ddc8770e93ea141e1fc673e017e97eadc6b968f385c2aecb03bfb32af3c54ec18db5c021afe43fbfaaa3afb29d1e6053c7c9475d8be6189f95cbba8990f95b1ebf1b3",
		"05eff700e9a13ae5ca0bcbd0484764bd1f231ea81c7b64c514735ac55e4b79633b706424119e09dcaad4acf21b10af3b33cde3504847155cbb6f2219ba9b7df50be11a1c7f23f829f8a41b13b5ca4ee8983238e0794d3d34bc5f4e77facb6c05ac86212baa1a55a2be70b5733b045cd33694b3afe2f0e49e4f321549fd824ea9",
		"cace28f4976afd72e3c5128167eb788fbf6634dda0a2f53148d00f6fa557f5e9e8f736c12e450894af56cb67f7d99e1027258c8571bd91ee3b7360e0d508aa1f382411a16115f9c05251cc326d4016f62e0eb8151c048465b0c6c8ff12558d43310e18b2cb1889eec91557ce21ba05955cf4c1d4847aadfb1b0a83f3a3b82b7efa62a5f03c5d6eda381a85dd78dbc55c",
	},
	{
		"1f6c3a3bc0542aabba4ef8f6c7169e73",
		"f3584606472b260e0dd2ebb2",
		"67c6697351ff4aec29cdbaabf2fbe3467cc254f81be8e78d765a2e63339fc99a66320db73158a35a255d051758e95ed4abb2cdc69bb454110e827441213ddc8770e93ea141e1fc673e017e97eadc6b968f385c2aecb03bfb32af3c54ec18db5c021afe43fbfaaa3afb29d1e6053c7c9475d8be6189f95cbba8990f95b1ebf1b305eff700e9a13ae5ca0bcbd0484764bd1f231ea81c7b64c514735ac55e4b79633b706424119e09dcaad4acf21b10af3b33cde3504847155cbb6f2219ba9b7df50be11a1c7f23f829f8a41b13b5ca4ee8983238e0794d3d34bc5f4e77facb6c05ac86212baa1a55a2be70b5733b045cd33694b3afe2f0e49e4f321549fd824ea90870d4b28a2954489a0abcd50e18a844ac5bf38e4cd72d9b0942e506c433afcda3847f2dadd47647de321cec4ac430f62023856cfbb20704f4ec0bb920ba86c33e05f1ecd96733b79950a3e314d3d934f75ea0f210a8f6059401beb4bc4478fa4969e623d01ada696a7e4c7e5125b34884533a94fb319990325744ee9bbce9e525cf08f5e9e25e5360aad2b2d085fa54d835e8d466826498d9a8877565705a8a3f62802944de7ca5894e5759d351adac869580ec17e485f18c0c66f17cc07cbb22fce466da610b63af62bc83b4692f3affaf271693ac071fb86d11342d8def4f89d4b66335c1c7e4248367d8ed9612ec453902d8e50af89d7709d1a596c1f41f",
		"95aa82ca6c49ae90cd1668baac7aa6f2b4a8ca99b2c2372acb08cf61c9c3805e6e0328da4cd76a19edd2d3994c798b0022569ad418d1fee4d9cd45a391c601ffc92ad91501432fee150287617c13629e69fc7281cd7165a63eab49cf714bce3a75a74f76ea7e64ff81eb61fdfec39b67bf0de98c7e4e32bdf97c8c6ac75ba43c02f4b2ed7216ecf3014df000108b67cf99505b179f8ed4980a6103d1bca70dbe9bbfab0ed59801d6e5f2d6f67d3ec5168e212e2daf02c6b963c98a1f7097de0c56891a2b211b01070dd8fd8b16c2a1a4e3cfd292d2984b3561d555d16c33ddc2bcf7edde13efe520c7e2abdda44d81881c531aeeeb66244c3b791ea8acfb6a68",
		"55864065117e07650ca650a0f0d9ef4b02aee7c58928462fddb49045bf85355b4653fa26158210a7f3ef5b3ca48612e8b7adf5c025c1b821960af770d935df1c9a1dd25077d6b1c7f937b2e20ce981b07980880214698f3fad72fa370b3b7da257ce1d0cf352bc5304fada3e0f8927bd4e5c1abbffa563bdedcb567daa64faaed748cb361732200ba3506836a3c1c82aafa14c76dc07f6c4277ff2c61325f91fdbd6c1883e745fcaadd5a6d692eeaa5ad56eead6a9d74a595d22757ed89532a4b8831e2b9e2315baea70a9b95d228f09d491a5ed5ab7076766703457e3159bbb9b17b329525669863153079448c68cd2f200c0be9d43061a60639cb59d50993d276c05caaa565db8ce633b2673e4012bebbca02b1a64d779d04066f3e949ece173825885ec816468c819a8129007cc05d8785c48077d09eb1abcba14508dde85a6f16a744bc95faef24888d53a8020515ab20307efaecbdf143a26563c67989bceedc2d6d2bb9699bb6c615d93767e4158c1124e3b6c723aaa47796e59a60d3696cd85adfae9a62f2c02c22009f80ed494bdc587f31dd892c253b5c6d6b7db078fa72d23474ee54f8144d6561182d71c862941dbc0b2cb37a4d4b23cbad5637e6be901cc73f16d5aec39c60dddee631511e57b47520b61ae1892d2d1bd2b486e30faec892f171b6de98d96108016fac805604761f8e74742b3bb7dc8a290a46bf697c3e4446e6e65832cbae7cf1aaad1",
	},
	{
		"0795d80bc7f40f4d41c280271a2e4f7f",
		"ff824c906594aff365d3cb1f",
		"1ad4e74d127f935beee57cff920665babe7ce56227377afe570ba786193ded3412d4812453157f42fafc418c02a746c1232c234a639d49baa8f041c12e2ef540027764568ce49886e0d913e28059a3a485c6eee96337a30b28e4cd5612c2961539fa6bc5de034cbedc5fa15db844013e0bef276e27ca7a4faf47a5c1093bd643354108144454d221b3737e6cb87faac36ed131959babe44af2890cfcc4e23ffa24470e689ce0894f5407bb0c8665cff536008ad2ac6f1c9ef8289abd0bd9b72f21c597bda5210cf928c805af2dd4a464d52e36819d521f967bba5386930ab5b4cf4c71746d7e6e964673457348e9d71d170d9eb560bd4bdb779e610ba816bf776231ebd0af5966f5cdab6815944032ab4dd060ad8dab880549e910f1ffcf6862005432afad",
		"98a47a430d8fd74dc1829a91e3481f8ed024d8ba34c9b903321b04864db333e558ae28653dffb2",
		"3b8f91443480e647473a0a0b03d571c622b7e70e4309a02c9bb7980053010d865e6aec161354dc9f481b2cd5213e09432b57ec4e58fbd0a8549dd15c8c4e74a6529f75fad0ce5a9e20e2beeb2f91eb638bf88999968de438d2f1cedbfb0a1c81f9e8e7362c738e0fddd963692a4f4df9276b7f040979ce874cf6fa3de26da0713784bdb25e4efcb840554ef5b38b5fe8380549a496bd8e423a7456df6f4ae78a07ebe2276a8e22fc2243ec4f78abe0c99c733fd67c8c492699fa5ee2289cdd0a8d469bf883520ee74efb854bfadc7366a49ee65ca4e894e3335e2b672618d362eee12a577dd8dc2ba55c49c1fc3ad68180e9b112d0234d4aa28f5661f1e036450ca6f18be0166676bd80f8a4890c6ddea306fabb7ff3cb2860aa32a827e3a312912a2dfa70f6bc1c07de238448f2d751bd0cf15bf7",
	},
	{
		"e2e001a36c60d2bf40d69ff5b2b1161ea218db263be16a4e",
		"84230643130d05425826641e",
		"adb034f3f4a7ca45e2993812d113a9821d50df151af978bccc6d3bc113e15bc0918fb385377dca1916022ce816d56a332649484043c0fc0f2d37d040182b00a9bbb42ef231f80b48fb3730110d9a4433e38c73264c703579a705b9c031b969ec6d98de9f90e9e78b21179c2eb1e061946cd4bbb844f031ecf6eaac27a4151311adf1b03eda97c9fbae66295f468af4b35faf6ba39f9d8f95873bbc2b51cf3dfec0ed3c9b850696336cc093b24a8765a936d14dd56edc6bf518272169f75e67b74ba452d0aae90416a997c8f31e2e9d54ffea296dc69462debc8347b3e1af6a2d53bdfdfda601134f98db42b609df0a08c9347590c8d86e845bb6373d65a26ab85f67b50569c85401a396b8ad76c2b53ff62bcfbf033e435ef47b9b591d05117c6dc681d68e",
		"d5d7316b8fdee152942148bff007c22e4b2022c6bc7be3c18c5f2e52e004e0b5dc12206bf002bd",
		"f2c39423ee630dfe961da81909159dba018ce09b1073a12a477108316af5b7a31f86be6a0548b572d604bd115ea737dde899e0bd7f7ac9b23e38910dc457551ecc15c814a9f46d8432a1a36097dc1afe2712d1ba0838fa88cb55d9f65a2e9bece0dbf8999562503989041a2c87d7eb80ef649769d2f4978ce5cf9664f2bd0849646aa81cb976e45e1ade2f17a8126219e917aadbb4bae5e2c4b3f57bbc7f13fcc807df7842d9727a1b389e0b749e5191482adacabd812627c6eae2c7a30caf0844ad2a22e08f39edddf0ae10413e47db433dfe3febbb5a5cec9ade21fbba1e548247579395880b747669a8eb7e2ec0c1bff7fed2defdb92b07a14edf07b1bde29c31ab052ff1214e6b5ebbefcb8f21b5d6f8f6e07ee57ad6e14d4e142cb3f51bb465ab3a28a2a12f01b7514ad0463f2bde0d71d221",
	},
	{
		"5394e890d37ba55ec9d5f327f15680f6a63ef5279c79331643ad0af6d2623525",
		"815e840b7aca7af3b324583f",
		"8e63067cd15359f796b43c68f093f55fdf3589fc5f2fdfad5f9d156668a617f7091d73da71cdd207810e6f71a165d0809a597df9885ca6e8f9bb4e616166586b83cc45f49917fc1a256b8bc7d05c476ab5c4633e20092619c4747b26dad3915e9fd65238ee4e5213badeda8a3a22f5efe6582d0762532026c89b4ca26fdd000eb45347a2a199b55b7790e6b1b2dba19833ce9f9522c0bcea5b088ccae68dd99ae0203c81b9f1dd3181c3e2339e83ccd1526b67742b235e872bea5111772aab574ae7d904d9b6355a79178e179b5ae8edc54f61f172bf789ea9c9af21f45b783e4251421b077776808f04972a5e801723cf781442378ce0e0568f014aea7a882dcbcb48d342be53d1c2ebfb206b12443a8a587cc1e55ca23beca385d61d0d03e9d84cbc1b0a",
		"0feccdfae8ed65fa31a0858a1c466f79e8aa658c2f3ba93c3f92158b4e30955e1c62580450beff",
		"b69a7e17bb5af688883274550a4ded0d1aff49a0b18343f4b382f745c163f7f714c9206a32a1ff012427e19431951edd0a755e5f491b0eedfd7df68bbc6085dd2888607a2f998c3e881eb1694109250db28291e71f4ad344a125624fb92e16ea9815047cd1111cabfdc9cb8c3b4b0f40aa91d31774009781231400789ed545404af6c3f76d07ddc984a7bd8f52728159782832e298cc4d529be96d17be898efd83e44dc7b0e2efc645849fd2bba61fef0ae7be0dcab233cc4e2b7ba4e887de9c64b97f2a1818aa54371a8d629dae37975f7784e5e3cc77055ed6e975b1e5f55e6bbacdc9f295ce4ada2c16113cd5b323cf78b7dde39f4a87aa8c141a31174e3584ccbd380cf5ec6d1dba539928b084fa9683e9c0953acf47cc3ac384a2c38914f1da01fb2cfd78905c2b58d36b2574b9df15535d82",
	},
	// These cases test non-standard tag sizes.
	{
		"89c54b0d3bc3c397d5039058c220685f",
		"bc7f45c00868758d62d4bb4d",
		"582670b0baf5540a3775b6615605bd05",
		"48d16cda0337105a50e2ed76fd18e114",
		"fc2d4c4eee2209ddbba6663c02765e6955e783b00156f5da0446e2970b877f",
	},
	{
		"bad6049678bf75c9087b3e3ae7e72c13",
		"a0a017b83a67d8f1b883e561",
		"a1be93012f05a1958440f74a5311f4a1",
		"f7c27b51d5367161dc2ff1e9e3edc6f2",
		"36f032f7e3dc3275ca22aedcdc68436b99a2227f8bb69d45ea5d8842cd08",
	},
	{
		"66a3c722ccf9709525650973ecc100a9",
		"1621d42d3a6d42a2d2bf9494",
		"61fa9dbbed2190fbc2ffabf5d2ea4ff8",
		"d7a9b6523b8827068a6354a6d166c6b9",
		"fef3b20f40e08a49637cc82f4c89b8603fd5c0132acfab97b5fff651c4",
	},
	{
		"562ae8aadb8d23e0f271a99a7d1bd4d1",
		"f7a5e2399413b89b6ad31aff",
		"bbdc3504d803682aa08a773cde5f231a",
		"2b9680b886b3efb7c6354b38c63b5373",
		"e2b7e5ed5ff27fc8664148f5a628a46dcbf2015184fffb82f2651c36",
	},
}

func TestAESGCM(t *testing.T) {
	for i, test := range aesGCMTests {
		key, _ := hex.DecodeString(test.key)
		aes, err := aes.NewCipher(key)
		if err != nil {
			t.Fatal(err)
		}

		nonce, _ := hex.DecodeString(test.nonce)
		plaintext, _ := hex.DecodeString(test.plaintext)
		ad, _ := hex.DecodeString(test.ad)
		tagSize := (len(test.result) - len(test.plaintext)) / 2

		var aesgcm cipher.AEAD
		switch {
		// Handle non-standard nonce sizes
		case tagSize != 16:
			aesgcm, err = cipher.NewGCMWithTagSize(aes, tagSize)
			if err != nil {
				t.Fatal(err)
			}

		// Handle non-standard tag sizes
		case len(nonce) != 12:
			aesgcm, err = cipher.NewGCMWithNonceSize(aes, len(nonce))
			if err != nil {
				t.Fatal(err)
			}

		default:
			aesgcm, err = cipher.NewGCM(aes)
			if err != nil {
				t.Fatal(err)
			}
		}

		ct := aesgcm.Seal(nil, nonce, plaintext, ad)
		if ctHex := hex.EncodeToString(ct); ctHex != test.result {
			t.Errorf("#%d: got %s, want %s", i, ctHex, test.result)
			continue
		}

		plaintext2, err := aesgcm.Open(nil, nonce, ct, ad)
		if err != nil {
			t.Errorf("#%d: Open failed", i)
			continue
		}

		if !bytes.Equal(plaintext, plaintext2) {
			t.Errorf("#%d: plaintext's don't match: got %x vs %x", i, plaintext2, plaintext)
			continue
		}

		if len(ad) > 0 {
			ad[0] ^= 0x80
			if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
				t.Errorf("#%d: Open was successful after altering additional data", i)
			}
			ad[0] ^= 0x80
		}

		nonce[0] ^= 0x80
		if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering nonce", i)
		}
		nonce[0] ^= 0x80

		ct[0] ^= 0x80
		if _, err := aesgcm.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering ciphertext", i)
		}
		ct[0] ^= 0x80
	}
}

func TestGCMInvalidTagSize(t *testing.T) {
	key, _ := hex.DecodeString("ab72c77b97cb5fe9a382d9fe81ffdbed")

	aes, _ := aes.NewCipher(key)

	for _, tagSize := range []int{0, 1, aes.BlockSize() + 1} {
		aesgcm, err := cipher.NewGCMWithTagSize(aes, tagSize)
		if aesgcm != nil || err == nil {
			t.Fatalf("NewGCMWithNonceAndTagSize was successful with an invalid %d-byte tag size", tagSize)
		}
	}
}

func TestTagFailureOverwrite(t *testing.T) {
	// The AESNI GCM code decrypts and authenticates concurrently and so
	// overwrites the output buffer before checking the authentication tag.
	// In order to be consistent across platforms, all implementations
	// should do this and this test checks that.

	key, _ := hex.DecodeString("ab72c77b97cb5fe9a382d9fe81ffdbed")
	nonce, _ := hex.DecodeString("54cc7dc2c37ec006bcc6d1db")
	ciphertext, _ := hex.DecodeString("0e1bde206a07a9c2c1b65300f8c649972b4401346697138c7a4891ee59867d0c")

	aes, _ := aes.NewCipher(key)
	aesgcm, _ := cipher.NewGCM(aes)

	dst := make([]byte, len(ciphertext)-16)
	for i := range dst {
		dst[i] = 42
	}

	result, err := aesgcm.Open(dst[:0], nonce, ciphertext, nil)
	if err == nil {
		t.Fatal("Bad Open still resulted in nil error.")
	}

	if result != nil {
		t.Fatal("Failed Open returned non-nil result.")
	}

	for i := range dst {
		if dst[i] != 0 {
			t.Fatal("Failed Open didn't zero dst buffer")
		}
	}
}

func TestGCMCounterWrap(t *testing.T) {
	// Test that the last 32-bits of the counter wrap correctly.
	tests := []struct {
		nonce, tag string
	}{
		{"0fa72e25", "37e1948cdfff09fbde0c40ad99fee4a7"},   // counter: 7eb59e4d961dad0dfdd75aaffffffff0
		{"afe05cc1", "438f3aa9fee5e54903b1927bca26bbdf"},   // counter: 75d492a7e6e6bfc979ad3a8ffffffff4
		{"9ffecbef", "7b88ca424df9703e9e8611071ec7e16e"},   // counter: c8bb108b0ecdc71747b9d57ffffffff5
		{"ffc3e5b3", "38d49c86e0abe853ac250e66da54c01a"},   // counter: 706414d2de9b36ab3b900a9ffffffff6
		{"cfdd729d", "e08402eaac36a1a402e09b1bd56500e8"},   // counter: cd0b96fe36b04e750584e56ffffffff7
		{"010ae3d486", "5405bb490b1f95d01e2ba735687154bc"}, // counter: e36c18e69406c49722808104fffffff8
		{"01b1107a9d", "939a585f342e01e17844627492d44dbf"}, // counter: e6d56eaf9127912b6d62c6dcffffffff
	}
	key, err := aes.NewCipher(make([]byte, 16))
	if err != nil {
		t.Fatal(err)
	}
	plaintext := make([]byte, 16*17+1)
	for i, test := range tests {
		nonce, _ := hex.DecodeString(test.nonce)
		want, _ := hex.DecodeString(test.tag)
		aead, err := cipher.NewGCMWithNonceSize(key, len(nonce))
		if err != nil {
			t.Fatal(err)
		}
		got := aead.Seal(nil, nonce, plaintext, nil)
		if !bytes.Equal(got[len(plaintext):], want) {
			t.Errorf("test[%v]: got: %x, want: %x", i, got[len(plaintext):], want)
		}
		_, err = aead.Open(nil, nonce, got, nil)
		if err != nil {
			t.Errorf("test[%v]: authentication failed", i)
		}
	}
}

var _ cipher.Block = (*wrapper)(nil)

type wrapper struct {
	block cipher.Block
}

func (w *wrapper) BlockSize() int          { return w.block.BlockSize() }
func (w *wrapper) Encrypt(dst, src []byte) { w.block.Encrypt(dst, src) }
func (w *wrapper) Decrypt(dst, src []byte) { w.block.Decrypt(dst, src) }

// wrap wraps the Block interface so that it does not fulfill
// any optimizing interfaces such as gcmAble.
func wrap(b cipher.Block) cipher.Block {
	return &wrapper{b}
}

func TestGCMAsm(t *testing.T) {
	// Create a new pair of AEADs, one using the assembly implementation
	// and one using the generic Go implementation.
	newAESGCM := func(key []byte) (asm, generic cipher.AEAD, err error) {
		block, err := aes.NewCipher(key[:])
		if err != nil {
			return nil, nil, err
		}
		asm, err = cipher.NewGCM(block)
		if err != nil {
			return nil, nil, err
		}
		generic, err = cipher.NewGCM(wrap(block))
		if err != nil {
			return nil, nil, err
		}
		return asm, generic, nil
	}

	// check for assembly implementation
	var key [16]byte
	asm, generic, err := newAESGCM(key[:])
	if err != nil {
		t.Fatal(err)
	}
	if reflect.TypeOf(asm) == reflect.TypeOf(generic) {
		t.Skipf("no assembly implementation of GCM")
	}

	// generate permutations
	type pair struct{ align, length int }
	lengths := []int{0, 156, 8192, 8193, 8208}
	keySizes := []int{16, 24, 32}
	alignments := []int{0, 1, 2, 3}
	if testing.Short() {
		keySizes = []int{16}
		alignments = []int{1}
	}
	perms := make([]pair, 0)
	for _, l := range lengths {
		for _, a := range alignments {
			if a != 0 && l == 0 {
				continue
			}
			perms = append(perms, pair{align: a, length: l})
		}
	}

	// run test for all permutations
	test := func(ks int, pt, ad []byte) error {
		key := make([]byte, ks)
		if _, err := io.ReadFull(rand.Reader, key); err != nil {
			return err
		}
		asm, generic, err := newAESGCM(key)
		if err != nil {
			return err
		}
		if _, err := io.ReadFull(rand.Reader, pt); err != nil {
			return err
		}
		if _, err := io.ReadFull(rand.Reader, ad); err != nil {
			return err
		}
		nonce := make([]byte, 12)
		if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
			return err
		}
		want := generic.Seal(nil, nonce, pt, ad)
		got := asm.Seal(nil, nonce, pt, ad)
		if !bytes.Equal(want, got) {
			return errors.New("incorrect Seal output")
		}
		got, err = asm.Open(nil, nonce, want, ad)
		if err != nil {
			return errors.New("authentication failed")
		}
		if !bytes.Equal(pt, got) {
			return errors.New("incorrect Open output")
		}
		return nil
	}
	for _, a := range perms {
		ad := make([]byte, a.align+a.length)
		ad = ad[a.align:]
		for _, p := range perms {
			pt := make([]byte, p.align+p.length)
			pt = pt[p.align:]
			for _, ks := range keySizes {
				if err := test(ks, pt, ad); err != nil {
					t.Error(err)
					t.Errorf("	key size: %v", ks)
					t.Errorf("	plaintext alignment: %v", p.align)
					t.Errorf("	plaintext length: %v", p.length)
					t.Errorf("	additionalData alignment: %v", a.align)
					t.Fatalf("	additionalData length: %v", a.length)
				}
			}
		}
	}
}
