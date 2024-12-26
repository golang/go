package hybrid

import (
	"crypto/sha256"
	"flag"
  "fmt"
	"testing"
  "time"
	"os"

  "github.com/dterei/gotsc"
)

var algorithms = []OID{
	MAYO1_ED25519,
	MAYO2_ED25519,
	MAYO3_ED25519,
	MAYO5_ED25519,
	CROSS_128_SMALL_ED25519,
	CROSS_128_FAST_ED25519,
	CROSS_192_SMALL_ED25519,
	CROSS_256_SMALL_ED25519,
  ML_DSA_44_P256,
  ML_DSA_65_P384,
  ML_DSA_87_P521,
	ML_DSA_65_ED25519,
}

var (
	duration time.Duration
)

func init() {
	// Define the duration flag and set its default value
	flag.DurationVar(&duration, "duration", 3*time.Second, "duration for each test")
}

// TestMain processes flags before running tests
func TestMain(m *testing.M) {
	flag.Parse()
	fmt.Printf("Duration set to: %v\n", duration)
	os.Exit(m.Run())
}

var hash = sha256.Sum256([]byte("Test message for hybrid signature"))

func TestKeygenHybrid(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(string(alg), func(t *testing.T) {
			tsc := gotsc.TSCOverhead()

			start := time.Now()
			startCycles := gotsc.BenchStart()
			var keyGenCPU []int64
			var keyGenTime []int64

			iterations := 0
			for time.Since(start) < duration {
				// Test key generation
				_, err := GenerateKey(alg)
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

			// Print results
      fmt.Printf("TESTING KEYGEN - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}

func TestSignHybrid(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(string(alg), func(t *testing.T) {

      privKey, err := GenerateKey(alg)
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
				// Test signing
				_, err = privKey.Sign(hash[:])
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

			// Print results
			fmt.Printf("TESTING SIGN - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}

func TestVerifyHybrid(t *testing.T) {
	for _, alg := range algorithms {
		t.Run(string(alg), func(t *testing.T) {
      privKey, err := GenerateKey(alg)
      if err != nil {
        t.Fatalf("Failed to generate keys for %s: %v", alg, err)
      }
      pubKey := privKey.ExportPublicKey()

      signature, err := privKey.Sign(hash[:])
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
				// Test verification
				if !VerifyHybrid(pubKey, hash[:], signature) {
					t.Errorf("Signature verification failed for %s", alg)
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

			// Print results
			fmt.Printf("TESTING VERIFY - Algorithm: %s | Iterations: %d | Mean CPU Cycles: %d | Mean Time: %d µs | Operations/S: %f s\n", alg, iterations, meanCPU, meanTime, opsPerS)
		})
	}
}
