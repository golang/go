// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package note_test

import (
	"fmt"
	"io"
	"os"

	"cmd/go/internal/note"
)

func ExampleSign() {
	skey := "PRIVATE+KEY+PeterNeumann+c74f20a3+AYEKFALVFGyNhPJEMzD1QIDr+Y7hfZx09iUvxdXHKDFz"
	text := "If you think cryptography is the answer to your problem,\n" +
		"then you don't know what your problem is.\n"

	signer, err := note.NewSigner(skey)
	if err != nil {
		fmt.Println(err)
		return
	}

	msg, err := note.Sign(&note.Note{Text: text}, signer)
	if err != nil {
		fmt.Println(err)
		return
	}
	os.Stdout.Write(msg)

	// Output:
	// If you think cryptography is the answer to your problem,
	// then you don't know what your problem is.
	//
	// — PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=
}

func ExampleOpen() {
	vkey := "PeterNeumann+c74f20a3+ARpc2QcUPDhMQegwxbzhKqiBfsVkmqq/LDE4izWy10TW"
	msg := []byte("If you think cryptography is the answer to your problem,\n" +
		"then you don't know what your problem is.\n" +
		"\n" +
		"— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=\n")

	verifier, err := note.NewVerifier(vkey)
	if err != nil {
		fmt.Println(err)
		return
	}
	verifiers := note.VerifierList(verifier)

	n, err := note.Open(msg, verifiers)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s (%08x):\n%s", n.Sigs[0].Name, n.Sigs[0].Hash, n.Text)

	// Output:
	// PeterNeumann (c74f20a3):
	// If you think cryptography is the answer to your problem,
	// then you don't know what your problem is.
}

var rand = struct {
	Reader io.Reader
}{
	zeroReader{},
}

type zeroReader struct{}

func (zeroReader) Read(buf []byte) (int, error) {
	for i := range buf {
		buf[i] = 0
	}
	return len(buf), nil
}

func ExampleSign_add_signatures() {
	vkey := "PeterNeumann+c74f20a3+ARpc2QcUPDhMQegwxbzhKqiBfsVkmqq/LDE4izWy10TW"
	msg := []byte("If you think cryptography is the answer to your problem,\n" +
		"then you don't know what your problem is.\n" +
		"\n" +
		"— PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=\n")

	verifier, err := note.NewVerifier(vkey)
	if err != nil {
		fmt.Println(err)
		return
	}
	verifiers := note.VerifierList(verifier)

	n, err := note.Open([]byte(msg), verifiers)
	if err != nil {
		fmt.Println(err)
		return
	}

	skey, vkey, err := note.GenerateKey(rand.Reader, "EnochRoot")
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = vkey // give to verifiers

	me, err := note.NewSigner(skey)
	if err != nil {
		fmt.Println(err)
		return
	}

	msg, err = note.Sign(n, me)
	if err != nil {
		fmt.Println(err)
		return
	}
	os.Stdout.Write(msg)

	// Output:
	// If you think cryptography is the answer to your problem,
	// then you don't know what your problem is.
	//
	// — PeterNeumann x08go/ZJkuBS9UG/SffcvIAQxVBtiFupLLr8pAcElZInNIuGUgYN1FFYC2pZSNXgKvqfqdngotpRZb6KE6RyyBwJnAM=
	// — EnochRoot rwz+eBzmZa0SO3NbfRGzPCpDckykFXSdeX+MNtCOXm2/5n2tiOHp+vAF1aGrQ5ovTG01oOTGwnWLox33WWd1RvMc+QQ=
}
