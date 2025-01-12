package testing

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"testing"
	"time"

	"github.com/dterei/gotsc"
	"github.com/open-quantum-safe/liboqs-go/oqs"
)

var algorithms = []string{
	"ML-DSA-44",
	"ML-DSA-65",
	"ML-DSA-87",
	"cross-rsdp-128-small",
	"cross-rsdp-128-fast",
	"cross-rsdp-192-small",
	"cross-rsdp-256-small",
}

var duration time.Duration

func init() {
	// Define the duration flag and set its default value
	flag.DurationVar(&duration, "duration", 3*time.Second, "duration for each test")
}

func TestMain(m *testing.M) {
	flag.Parse()
	fmt.Printf("Duration set to: %v\n", duration)
	m.Run()
}

var hash = sha256.Sum256([]byte("Test message for post-quantum signature"))

func TestKeygenPQC(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			signer := oqs.Signature{}
			if err := signer.Init(alg, nil); err != nil {
				t.Fatalf("Failed to initialize algorithm %s: %v", alg, err)
			}
			defer signer.Clean()

			tsc := gotsc.TSCOverhead()

			start := time.Now()
			startCycles := gotsc.BenchStart()
			var keyGenCPU []int64
			var keyGenTime []int64

			iterations := 0
			for time.Since(start) < duration {
				_, err := signer.GenerateKeyPair()
				if err != nil {
					t.Fatalf("Failed to generate keys for %s: %v", alg, err)
				}

				keyGenTime = append(keyGenTime, time.Since(start).Microseconds())
				keyGenCPU = append(keyGenCPU, int64(gotsc.BenchEnd()-startCycles-tsc))

				iterations++
			}

			mean := func(data []int64) int64 {
				var sum int64
				for _, v := range data {
					sum += v
				}
				return sum / int64(len(data))
			}

			opsPerS := float64(iterations) / duration.Seconds()
			meanCPU := mean(keyGenCPU)
			meanTime := mean(keyGenTime)

			fmt.Printf("TESTING KEYGEN - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}

func TestSignPQC(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			signer := oqs.Signature{}
			if err := signer.Init(alg, nil); err != nil {
				t.Fatalf("Failed to initialize algorithm %s: %v", alg, err)
			}
			defer signer.Clean()

			_, err := signer.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Failed to generate keys for %s: %v", alg, err)
			}

			tsc := gotsc.TSCOverhead()

			start := time.Now()
			startCycles := gotsc.BenchStart()
			var signCPU []int64
			var signTime []int64

			iterations := 0
			for time.Since(start) < duration {
				_, err := signer.Sign(hash[:])
				if err != nil {
					t.Fatalf("Failed to sign message for %s: %v", alg, err)
				}

				signTime = append(signTime, time.Since(start).Microseconds())
				signCPU = append(signCPU, int64(gotsc.BenchEnd()-startCycles-tsc))

				iterations++
			}

			mean := func(data []int64) int64 {
				var sum int64
				for _, v := range data {
					sum += v
				}
				return sum / int64(len(data))
			}

			opsPerS := float64(iterations) / duration.Seconds()
			meanCPU := mean(signCPU)
			meanTime := mean(signTime)

			fmt.Printf("TESTING SIGN - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}

func TestVerifyPQC(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			signer := oqs.Signature{}
			if err := signer.Init(alg, nil); err != nil {
				t.Fatalf("Failed to initialize algorithm %s: %v", alg, err)
			}
			defer signer.Clean()

			publicKey, err := signer.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Failed to generate keys for %s: %v", alg, err)
			}

			signature, err := signer.Sign(hash[:])
			if err != nil {
				t.Fatalf("Failed to sign message for %s: %v", alg, err)
			}

			tsc := gotsc.TSCOverhead()

			start := time.Now()
			startCycles := gotsc.BenchStart()
			var verifyCPU []int64
			var verifyTime []int64

			iterations := 0
			for time.Since(start) < duration {
				valid, err := signer.Verify(hash[:], signature, publicKey)
				if err != nil || !valid {
					t.Fatalf("Failed to verify signature for %s", alg)
				}

				verifyTime = append(verifyTime, time.Since(start).Microseconds())
				verifyCPU = append(verifyCPU, int64(gotsc.BenchEnd()-startCycles-tsc))

				iterations++
			}

			mean := func(data []int64) int64 {
				var sum int64
				for _, v := range data {
					sum += v
				}
				return sum / int64(len(data))
			}

			opsPerS := float64(iterations) / duration.Seconds()
			meanCPU := mean(verifyCPU)
			meanTime := mean(verifyTime)

			fmt.Printf("TESTING VERIFY - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}
