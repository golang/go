// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"testing"
)

func TestPrivateKeyRead(t *testing.T) {
	packet, err := Read(readerFromHex(privKeyHex))
	if err != nil {
		t.Error(err)
		return
	}

	privKey := packet.(*PrivateKey)

	if !privKey.Encrypted {
		t.Error("private key isn't encrypted")
		return
	}

	err = privKey.Decrypt([]byte("testing"))
	if err != nil {
		t.Error(err)
		return
	}

	if privKey.CreationTime != 0x4cc349a8 || privKey.Encrypted {
		t.Errorf("failed to parse, got: %#v", privKey)
	}
}

// Generated with `gpg --export-secret-keys "Test Key 2"`
const privKeyHex = "9501fe044cc349a8010400b70ca0010e98c090008d45d1ee8f9113bd5861fd57b88bacb7c68658747663f1e1a3b5a98f32fda6472373c024b97359cd2efc88ff60f77751adfbf6af5e615e6a1408cfad8bf0cea30b0d5f53aa27ad59089ba9b15b7ebc2777a25d7b436144027e3bcd203909f147d0e332b240cf63d3395f5dfe0df0a6c04e8655af7eacdf0011010001fe0303024a252e7d475fd445607de39a265472aa74a9320ba2dac395faa687e9e0336aeb7e9a7397e511b5afd9dc84557c80ac0f3d4d7bfec5ae16f20d41c8c84a04552a33870b930420e230e179564f6d19bb153145e76c33ae993886c388832b0fa042ddda7f133924f3854481533e0ede31d51278c0519b29abc3bf53da673e13e3e1214b52413d179d7f66deee35cac8eacb060f78379d70ef4af8607e68131ff529439668fc39c9ce6dfef8a5ac234d234802cbfb749a26107db26406213ae5c06d4673253a3cbee1fcbae58d6ab77e38d6e2c0e7c6317c48e054edadb5a40d0d48acb44643d998139a8a66bb820be1f3f80185bc777d14b5954b60effe2448a036d565c6bc0b915fcea518acdd20ab07bc1529f561c58cd044f723109b93f6fd99f876ff891d64306b5d08f48bab59f38695e9109c4dec34013ba3153488ce070268381ba923ee1eb77125b36afcb4347ec3478c8f2735b06ef17351d872e577fa95d0c397c88c71b59629a36aec"
