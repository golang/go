// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package fipstest

import (
	"bytes"
	"crypto/internal/cryptotest"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/entropy"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"flag"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var flagEntropySamples = flag.String("entropy-samples", "", "store entropy samples with the provided `suffix`")
var flagNISTSP80090B = flag.Bool("nist-sp800-90b", false, "run NIST SP 800-90B tests (requires docker)")

func TestEntropySamples(t *testing.T) {
	cryptotest.MustSupportFIPS140(t)
	now := time.Now().UTC()

	var seqSamples [1_000_000]uint8
	samplesOrTryAgain(t, seqSamples[:])
	seqSamplesName := fmt.Sprintf("entropy_samples_sequential_%s_%s_%s_%s_%s.bin", entropy.Version(),
		runtime.GOOS, runtime.GOARCH, *flagEntropySamples, now.Format("20060102T150405Z"))
	if *flagEntropySamples != "" {
		if err := os.WriteFile(seqSamplesName, seqSamples[:], 0644); err != nil {
			t.Fatalf("failed to write samples to %q: %v", seqSamplesName, err)
		}
		t.Logf("wrote %s", seqSamplesName)
	}

	var restartSamples [1000][1000]uint8
	for i := range restartSamples {
		var samples [1024]uint8
		samplesOrTryAgain(t, samples[:])
		copy(restartSamples[i][:], samples[:])
	}
	restartSamplesName := fmt.Sprintf("entropy_samples_restart_%s_%s_%s_%s_%s.bin", entropy.Version(),
		runtime.GOOS, runtime.GOARCH, *flagEntropySamples, now.Format("20060102T150405Z"))
	if *flagEntropySamples != "" {
		f, err := os.Create(restartSamplesName)
		if err != nil {
			t.Fatalf("failed to create %q: %v", restartSamplesName, err)
		}
		for i := range restartSamples {
			if _, err := f.Write(restartSamples[i][:]); err != nil {
				t.Fatalf("failed to write samples to %q: %v", restartSamplesName, err)
			}
		}
		if err := f.Close(); err != nil {
			t.Fatalf("failed to close %q: %v", restartSamplesName, err)
		}
		t.Logf("wrote %s", restartSamplesName)
	}

	if *flagNISTSP80090B {
		if *flagEntropySamples == "" {
			t.Fatalf("-nist-sp800-90b requires -entropy-samples to be set too")
		}

		// Check if the nist-sp800-90b docker image is already present,
		// and build it otherwise.
		if err := testenv.Command(t,
			"docker", "image", "inspect", "nist-sp800-90b",
		).Run(); err != nil {
			t.Logf("building nist-sp800-90b docker image")
			dockerfile := filepath.Join(t.TempDir(), "Dockerfile.SP800-90B_EntropyAssessment")
			if err := os.WriteFile(dockerfile, []byte(NISTSP80090BDockerfile), 0644); err != nil {
				t.Fatalf("failed to write Dockerfile: %v", err)
			}
			out, err := testenv.Command(t,
				"docker", "build", "-t", "nist-sp800-90b", "-f", dockerfile, "/var/empty",
			).CombinedOutput()
			if err != nil {
				t.Fatalf("failed to build nist-sp800-90b docker image: %v\n%s", err, out)
			}
		}

		pwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("failed to get current working directory: %v", err)
		}
		t.Logf("running ea_non_iid analysis")
		out, err := testenv.Command(t,
			"docker", "run", "--rm", "-v", fmt.Sprintf("%s:%s", pwd, pwd), "-w", pwd,
			"nist-sp800-90b", "ea_non_iid", seqSamplesName, "8",
		).CombinedOutput()
		if err != nil {
			t.Fatalf("ea_non_iid failed: %v\n%s", err, out)
		}
		t.Logf("\n%s", out)

		H_I := string(out)
		H_I = strings.TrimSpace(H_I[strings.LastIndexByte(H_I, ' ')+1:])
		t.Logf("running ea_restart analysis with H_I = %s", H_I)
		out, err = testenv.Command(t,
			"docker", "run", "--rm", "-v", fmt.Sprintf("%s:%s", pwd, pwd), "-w", pwd,
			"nist-sp800-90b", "ea_restart", restartSamplesName, "8", H_I,
		).CombinedOutput()
		if err != nil {
			t.Fatalf("ea_restart failed: %v\n%s", err, out)
		}
		t.Logf("\n%s", out)
	}
}

var NISTSP80090BDockerfile = `
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y build-essential git \
    libbz2-dev libdivsufsort-dev libjsoncpp-dev libgmp-dev libmpfr-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/usnistgov/SP800-90B_EntropyAssessment.git
RUN cd SP800-90B_EntropyAssessment && git checkout 8924f158c97e7b805e0f95247403ad4c44b9cd6f
WORKDIR ./SP800-90B_EntropyAssessment/cpp/
RUN make all
RUN cd selftest && ./selftest
RUN cp ea_non_iid ea_restart /usr/local/bin/
`

var memory entropy.ScratchBuffer

// samplesOrTryAgain calls entropy.Samples up to 10 times until it succeeds.
// Samples has a non-negligible chance of failing the health tests, as required
// by SP 800-90B.
func samplesOrTryAgain(t *testing.T, samples []uint8) {
	t.Helper()
	for range 10 {
		if err := entropy.Samples(samples, &memory); err != nil {
			t.Logf("entropy.Samples() failed: %v", err)
			continue
		}
		return
	}
	t.Fatal("entropy.Samples() failed 10 times in a row")
}

func TestEntropySHA384(t *testing.T) {
	var input [1024]uint8
	for i := range input {
		input[i] = uint8(i)
	}
	want := sha512.Sum384(input[:])
	got := entropy.SHA384(&input)
	if got != want {
		t.Errorf("SHA384() = %x, want %x", got, want)
	}

	for l := range 1024*3 + 1 {
		input := make([]byte, l)
		rand.Read(input)
		want := sha512.Sum384(input)
		got := entropy.TestingOnlySHA384(input)
		if got != want {
			t.Errorf("TestingOnlySHA384(%d bytes) = %x, want %x", l, got, want)
		}
	}
}

func TestEntropyRepetitionCountTest(t *testing.T) {
	good := bytes.Repeat(append(bytes.Repeat([]uint8{42}, 40), 1), 100)
	if err := entropy.RepetitionCountTest(good); err != nil {
		t.Errorf("RepetitionCountTest(good) = %v, want nil", err)
	}

	bad := bytes.Repeat([]uint8{0}, 40)
	bad = append(bad, bytes.Repeat([]uint8{1}, 40)...)
	bad = append(bad, bytes.Repeat([]uint8{42}, 41)...)
	bad = append(bad, bytes.Repeat([]uint8{2}, 40)...)
	if err := entropy.RepetitionCountTest(bad); err == nil {
		t.Error("RepetitionCountTest(bad) = nil, want error")
	}

	bad = bytes.Repeat([]uint8{42}, 41)
	if err := entropy.RepetitionCountTest(bad); err == nil {
		t.Error("RepetitionCountTest(bad) = nil, want error")
	}
}

func TestEntropyAdaptiveProportionTest(t *testing.T) {
	good := bytes.Repeat([]uint8{0}, 409)
	good = append(good, bytes.Repeat([]uint8{1}, 512-409)...)
	good = append(good, bytes.Repeat([]uint8{0}, 409)...)
	if err := entropy.AdaptiveProportionTest(good); err != nil {
		t.Errorf("AdaptiveProportionTest(good) = %v, want nil", err)
	}

	// These fall out of the window.
	bad := bytes.Repeat([]uint8{1}, 100)
	bad = append(bad, bytes.Repeat([]uint8{1, 2, 3, 4, 5, 6}, 100)...)
	// These are in the window.
	bad = append(bad, bytes.Repeat([]uint8{42}, 410)...)
	if err := entropy.AdaptiveProportionTest(bad[:len(bad)-1]); err != nil {
		t.Errorf("AdaptiveProportionTest(bad[:len(bad)-1]) = %v, want nil", err)
	}
	if err := entropy.AdaptiveProportionTest(bad); err == nil {
		t.Error("AdaptiveProportionTest(bad) = nil, want error")
	}
}

func TestEntropyUnchanged(t *testing.T) {
	testenv.MustHaveSource(t)

	h := sha256.New()
	root := os.DirFS("../fips140/entropy")
	if err := fs.WalkDir(root, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		data, err := fs.ReadFile(root, path)
		if err != nil {
			return err
		}
		t.Logf("Hashing %s (%d bytes)", path, len(data))
		fmt.Fprintf(h, "%s %d\n", path, len(data))
		h.Write(data)
		return nil
	}); err != nil {
		t.Fatalf("WalkDir: %v", err)
	}

	// The crypto/internal/fips140/entropy package is certified as a FIPS 140-3
	// entropy source through the Entropy Source Validation program,
	// independently of the FIPS 140-3 module. It must not change even across
	// FIPS 140-3 module versions, in order to reuse the ESV certificate.
	exp := "1b68d4c091ef66c6006602e4ed3ac10f8a82ad193708ec99d63b145e3baa3e6c"
	if got := hex.EncodeToString(h.Sum(nil)); got != exp {
		t.Errorf("hash of crypto/internal/fips140/entropy = %s, want %s", got, exp)
	}
}

func TestEntropyRace(t *testing.T) {
	// Check that concurrent calls to Seed don't trigger the race detector.
	for range 2 {
		go func() {
			_, _ = entropy.Seed(&memory)
		}()
	}
	// Same, with the higher-level DRBG. More concurrent calls to hit the Pool.
	for range 16 {
		go func() {
			var b [64]byte
			drbg.Read(b[:])
		}()
	}
}

var sink byte

func BenchmarkEntropySeed(b *testing.B) {
	for b.Loop() {
		seed, err := entropy.Seed(&memory)
		if err != nil {
			b.Fatalf("entropy.Seed() failed: %v", err)
		}
		sink ^= seed[0]
	}
}
